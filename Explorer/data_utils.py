"""
Data loading, preprocessing, and scoring helpers.

All functions that touch large numpy arrays live here so they can be
imported from any tab module without duplicating logic.

Two-stage loading pipeline
--------------------------
_load_file_cached  — Stage 1: file I/O + spatial/temporal filter only.
                     Cached by (file, parameter, bounds). Changing preprocessing
                     flags does NOT cause a re-read.

load_data_cached   — Stage 2: preprocessing on top of Stage 1 cache.
                     Copies the raw data, then applies the requested transforms.
                     Cached per unique combination of preprocessing flags.

compute_seasonal_cycle_cached — derives a (365, H, W) climatology from Stage 1
                     data without storing a second full (T, H, W) array.
                     Returns (seasonal_cycle, md_index, noleap_orig_idx) so
                     callers can compute per-event anomalies on the fly.
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
import streamlit as st

import src.preprocessing as pp
import src.rarity_scoring_base
import src.rarity_scoring_exclusion
import src.rarity_scoring_interval


# ── Stage 1: file I/O ───────────────────────────────────────────────────────
# Cached by file/bounds only. max_entries=2 covers different spatial subsets
# of the same file. The arrays are shared objects — never mutate them.

@st.cache_resource(show_spinner=False, max_entries=1)
def _load_file_cached(
    file_path, parameter,
    lon_min, lon_max, lat_min, lat_max,
    start_year, end_year,
):
    """Read the NetCDF file and apply spatial/temporal filters.
    Returns raw data (leap days still present). Do NOT mutate the result.
    """
    ds = xr.open_dataset(file_path)

    if start_year is not None and end_year is not None:
        ds = ds.sel(time=slice(f"{int(start_year)}-01-01", f"{int(end_year)}-12-31"))

    lon_dim = "lon" if "lon" in ds.dims else "longitude"
    lat_dim = "lat" if "lat" in ds.dims else "latitude"

    if lon_min is not None and lon_max is not None:
        ds = ds.sel({lon_dim: slice(float(lon_min), float(lon_max))})
    if lat_min is not None and lat_max is not None:
        ds = ds.sel({lat_dim: slice(float(lat_max), float(lat_min))})

    lat = ds[lat_dim].values.astype(np.float32)
    lon = ds[lon_dim].values.astype(np.float32)
    data = ds[parameter].transpose("time", lat_dim, lon_dim).values.astype(np.float32) / 100.0
    times = pd.to_datetime(ds["time"].values)
    ds.close()

    return data, times, lat, lon


def load_stage1(load_params):
    """Public accessor for Stage 1 cached data (raw, may include leap days)."""
    return _load_file_cached(**load_params)


# ── Stage 2: preprocessing ──────────────────────────────────────────────────
# Reuses Stage 1. Copies the raw data before modifying it so the Stage 1
# cache entry is never mutated. max_entries=4 covers typical combinations:
# scoring preproc, cluster preproc, cluster with detrend.

# @st.cache_resource(show_spinner=False, max_entries=1)
def load_data_cached(
    file_path, parameter,
    lon_min, lon_max, lat_min, lat_max,
    start_year, end_year,
    remove_leap, remove_sc, cos_lat, pxstd, detrend=False,
):
    """Apply preprocessing to the Stage 1 cached data.
    The returned arrays are owned by this cache entry — do NOT mutate.
    """
    data, times, lat, lon = _load_file_cached(
        file_path, parameter,
        lon_min, lon_max, lat_min, lat_max,
        start_year, end_year,
    )
    # data = raw_data.copy()
    # times = raw_times.copy()
    data = torch.from_numpy(data) 

    if remove_leap:
        data, times = pp.remove_bisex_dailydata(data, times)
    if remove_sc:
        data = pp.remove_seasonal_cycle365(data, times)
    if cos_lat:
        data = pp.cos_lat_weighting(data, lat, data.shape[2])
    if pxstd:
        shape = data.shape
        data = data.reshape(shape[0], -1)
        data = pp.pixelwise_standardize(data)
        data = data.reshape(shape)

    return data, times, lat, lon


# ── Seasonal cycle ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False, max_entries=2)
def compute_seasonal_cycle_cached(
    file_path, parameter,
    lon_min, lon_max, lat_min, lat_max,
    start_year, end_year,
):
    """Compute the 365-day climatology from Stage 1 data.

    Returns
    -------
    seasonal_cycle : np.ndarray, shape (365, H, W)
        Mean field for each day-of-year (leap days excluded).
    md_index : np.ndarray, shape (T_noleap,), dtype int16
        0-based day-of-year index for each time step in the leap-removed series.
    noleap_orig_idx : np.ndarray, shape (T_noleap,), dtype int64
        Indices into the Stage 1 raw array (which may contain leap days) that
        correspond to the leap-removed series. Use
        ``raw_data[noleap_orig_idx[i]]`` to retrieve the raw frame for event i.
    """
    raw_data, raw_times, _, _ = _load_file_cached(
        file_path, parameter,
        lon_min, lon_max, lat_min, lat_max,
        start_year, end_year,
    )

    times = pd.to_datetime(raw_times)
    noleap_mask = ~((times.month == 2) & (times.day == 29))
    noleap_orig_idx = np.where(noleap_mask)[0]

    data_nl = raw_data[noleap_orig_idx]   # (T_noleap, H, W) — temporary view
    times_nl = times[noleap_mask]

    month_cumsum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    md_index = (
        month_cumsum[times_nl.month.values - 1] + (times_nl.day.values - 1)
    ).astype(np.int16)

    seasonal_cycle = np.zeros((365, *data_nl.shape[1:]), dtype=np.float32)
    counts = np.zeros(365, dtype=np.int32)
    np.add.at(seasonal_cycle, md_index, data_nl)
    np.add.at(counts, md_index, 1)
    seasonal_cycle[counts > 0] /= counts[counts > 0, None, None]

    return seasonal_cycle, md_index, noleap_orig_idx


# ── Scoring ──────────────────────────────────────────────────────────────────

def run_scoring(data, traj_length, k, exclusion_zone, algorithm, device_str, use_pca=False):
    """Run trajectory-kNN rarity scoring. Returns a plain numpy array."""
    algo_map = {
        "base": src.rarity_scoring_base.compute_distances_and_scores,
        "exclusion": src.rarity_scoring_exclusion.compute_distances_and_scores,
        "interval": src.rarity_scoring_interval.compute_distances_and_scores,
    }
    dev = None if device_str == "auto" else device_str
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(
        q_batch=1024, r_chunk=1024,
        device=dev, dtype=torch.float32,
        exclusion_zone=exclusion_zone,
    )

    scores = algo_map[algorithm](data, traj_length, k, use_pca=use_pca, **kwargs)
    return scores.numpy()


def load_scores_from_npz(uploaded):
    """Load scores and times from a .npz file object."""
    npz = np.load(uploaded)
    scores = npz["scores"]
    times = pd.to_datetime(npz["times"])
    return scores, times
