import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy.signal import detrend
import cftime
from scipy.io import loadmat

def remove_daily_climatology(data, time):
    """
    Remove the average per day of year from the data, keep the spatial dimension.
    Parameters:
    - data: np.ndarray of shape (T, H, W)
    - time: np.ndarray of shape (T,) with datetime values
    Returns:
    - data: np.ndarray of shape (T, H, W)
    """

    day_of_year = pd.to_datetime(time).dayofyear
    daily_climatology = np.array(
        [data[day_of_year == doy].mean(axis=0) for doy in range(1, 367)])
    data = data - daily_climatology[day_of_year - 1]

    return data



def save_subset(nc_path):
    ds = xr.open_dataset(nc_path)
    n_time = 365 * 5
    ds = ds.isel(time=slice(0, n_time))
    ds.to_netcdf(f"Data/era5_msl_daily_eu_small.nc")

def validate_z500_anomalies(
    original_ds_path,
    anomalies_path="Data/z500_anom_daily_eu.nc",
    *,
    var_name="z500",
    groupby_key="auto",       # "auto", "mmdd", or "dayofyear"
    daily_mean_atol=1e-6,     # tighten/loosen as needed; 1e-3 is common for geopotential in SI
    recon_atol=1e-6,
    slope_check_points=3,
    slope_rel_tol=0.1,
    slope_abs_atol=1e-8,
):
    """
    Validate a 365-day anomaly file. If groupby_key="auto", tries both "%m-%d" and dayofyear
    and uses the one with smaller residual for the zero-mean check & reconstruction.
    """
    import numpy as np
    import xarray as xr
    from collections import namedtuple

    anom = xr.open_dataset(anomalies_path)[var_name]
    t = anom.time

    # 1) Time axis checks
    assert anom.indexes["time"].is_unique, "Duplicate timestamps in anomalies."
    dt = np.diff(t.values.astype("datetime64[ns]"))
    assert (dt > np.timedelta64(0, "ns")).all(), "Anomalies 'time' is not strictly increasing."

    # 2) No Feb 29
    assert not (((t.dt.month == 2) & (t.dt.day == 29)).any().item()), "Found Feb 29 in anomalies."

    # --- Helper: compute residuals under a given key ---
    KeyRes = namedtuple("KeyRes", "name key_labels n_unique max_abs_daily_mean daily_mean")
    def _res_for(keyname):
        if keyname == "mmdd":
            labels = t.dt.strftime("%m-%d")
        elif keyname == "dayofyear":
            labels = t.dt.dayofyear
        else:
            raise ValueError("keyname must be 'mmdd' or 'dayofyear'")
        n_unique = labels.to_series().nunique()
        dm = anom.groupby(labels).mean("time", skipna=True)
        max_abs = float(np.nanmax(np.abs(dm.values)))
        return KeyRes(keyname, labels, int(n_unique), max_abs, dm)

    # 3) Decide grouping key
    if groupby_key == "auto":
        r_mmdd = _res_for("mmdd")
        r_doy  = _res_for("dayofyear")
        # Prefer 365 keys if available; otherwise choose smaller residual
        if r_mmdd.n_unique == 365 and (r_doy.n_unique != 365 or r_mmdd.max_abs_daily_mean <= r_doy.max_abs_daily_mean):
            chosen = r_mmdd
        else:
            chosen = r_doy
    elif groupby_key == "mmdd":
        chosen = _res_for("mmdd")
    elif groupby_key == "dayofyear":
        chosen = _res_for("dayofyear")
    else:
        raise ValueError("groupby_key must be 'auto', 'mmdd', or 'dayofyear'")

    # 4) Enforce 365-day grouping (your requirement)
    assert chosen.n_unique == 365, f"Expected 365 groups, got {chosen.n_unique} using {chosen.name}."

    # 5) Daily mean ~ 0 under the chosen key
    if chosen.max_abs_daily_mean >= daily_mean_atol:
        # Give a small diagnostic of worst days
        # (works for both mmdd and dayofyear; convert to a readable label)
        dm = chosen.daily_mean
        # collapse spatial dims to a scalar per day to rank (use mean of abs)
        reduce_dims = [d for d in dm.dims if d != dm.dims[0]]  # first dim is the grouping dim
        perday = np.abs(dm).mean(reduce_dims, skipna=True)
        # pick top 5
        worst_idx = perday.argsort()[-5:].values
        worst_days = [str(dm[dm.dims[0]].values[i]) for i in worst_idx]
        raise AssertionError(
            f"Daily anomaly mean not ~0 (max abs={chosen.max_abs_daily_mean}) "
            f"using key '{chosen.name}'. Worst days: {worst_days}"
        )

    # 6) Trend check vs original
    ds0 = xr.open_dataset(original_ds_path)
    if "plev" in ds0.dims:
        ds0 = ds0.squeeze("plev", drop=True) if ds0.sizes["plev"] == 1 else ds0.isel(plev=0, drop=True)
    orig = ds0[var_name]
    orig = orig.sel(time=~((orig.time.dt.month == 2) & (orig.time.dt.day == 29)))
    orig, anom = xr.align(orig, anom, join="inner")

    def _slope(time_values, y):
        x = time_values.astype("datetime64[D]").astype(float)
        y = np.asarray(y, dtype=float)
        if np.all(np.isnan(y)) or len(y) < 2:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    dims = set(orig.dims)
    points = []
    if {"lat", "lon"} <= dims:
        lat_n, lon_n = orig.sizes["lat"], orig.sizes["lon"]
        points = [(0,0), (lat_n//2, lon_n//2), (lat_n-1, lon_n-1)][:max(1, slope_check_points)]
    else:
        points = [None]

    worst_rel = 0.0
    worst_abs = 0.0
    for pt in points:
        y_orig = orig.values if pt is None else orig.isel(lat=pt[0], lon=pt[1]).values
        y_anom = anom.values if pt is None else anom.isel(lat=pt[0], lon=pt[1]).values
        m_orig = abs(_slope(orig.time.values, y_orig))
        m_anom = abs(_slope(anom.time.values, y_anom))
        if not (np.isnan(m_orig) or np.isnan(m_anom)):
            worst_rel = max(worst_rel, (m_anom / m_orig) if m_orig > 0 else 0.0)
            worst_abs = max(worst_abs, m_anom)
            assert (m_anom < m_orig * slope_rel_tol) or (m_anom < slope_abs_atol), (
                f"Trend not adequately removed (post={m_anom}, pre={m_orig})."
            )

    # 7) Reconstruction under the chosen key
    key_labels = chosen.key_labels
    clim_from_orig = orig.groupby(key_labels).mean("time", skipna=True)
    recon = orig.groupby(key_labels) - clim_from_orig
    recon, anom_aligned = xr.align(recon, anom, join="inner")
    max_diff = float(np.nanmax(np.abs((recon - anom_aligned).values)))
    assert max_diff < recon_atol, f"Anomalies mismatch reconstruction (max diff={max_diff})."

    # 8) Dtype / NaNs / round-trip
    assert anom.dtype.kind in "f", f"Anomalies dtype should be float, got {anom.dtype}."
    total = np.prod(anom.shape) or 1
    nan_count = int(np.isnan(anom.values).sum())
    frac_nan = nan_count / total
    assert np.isfinite(frac_nan) and (frac_nan < 0.01 or frac_nan == 0.0), (
        f"Unexpected NaNs in anomalies (fraction={frac_nan})."
    )
    tmp = anomalies_path + ".__roundtrip__.nc"
    anom.to_netcdf(tmp)
    _ = xr.open_dataset(tmp)[var_name].isel(time=0).load()

    return {
        "groupby_key_used": chosen.name,
        "n_unique_groups": chosen.n_unique,
        "max_abs_daily_mean": chosen.max_abs_daily_mean,
        "reconstruction_max_abs_diff": max_diff,
        "nan_fraction": frac_nan,
        "slope_worst_relative": worst_rel,
        "slope_worst_abs": worst_abs,
    }



def pp_geopot(filepath):

    ds = xr.open_dataset(filepath)

    # Drop singleton plev safely (or pick a level if multiple)
    if "plev" in ds.dims:
        if ds.sizes["plev"] == 1:
            ds = ds.squeeze("plev", drop=True)
        else:
            ds = ds.isel(plev=0, drop=True)

    z500 = ds["z500"]

    # Detrend along time (ensure your detrend returns same-length array)
    z500 = xr.apply_ufunc(
        detrend,
        z500,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[z500.dtype],
    )

    # Remove Feb 29
    z500 = z500.sel(time=~((z500.time.dt.month == 2) & (z500.time.dt.day == 29)))

    # Group by mm-dd (365 unique keys) to build climatology
    mmdd = z500.time.dt.strftime("%m-%d").rename("mmdd")  # labels like '01-01', '12-31'
    clim = z500.groupby(mmdd).mean("time", skipna=True)

    # Anomalies w.r.t. that 365-day climatology
    z500_anom = z500.groupby(mmdd) - clim

    # Drop the grouping coord
    z500_anom = z500_anom.reset_coords("mmdd", drop=True)

    z500_anom.to_netcdf("Data/z500_anom_daily_eu.nc")

from datetime import datetime, timedelta

def mat_to_netcdf(mat_path, out_path='output.nc'):
    mat = loadmat(mat_path)
    hgt = np.squeeze(mat['hgt'])
    lat = np.squeeze(mat['lat'])
    lon = np.squeeze(mat['lon'])

    ntime = hgt.shape[0]
    assert ntime == 72 * 365, "Expected 72 years (1950–2021) of daily data, no leap days."

    # Build a no-leap time vector
    years = np.arange(1950, 2022)
    time = []
    for y in years:
        for d in range(365):
            time.append(datetime(y, 1, 1) + timedelta(days=d))
    time = np.array(time[:ntime])

    ds = xr.Dataset(
        data_vars={'hgt': (('time', 'lat', 'lon'), hgt, {'units': 'm'})},
        coords={
            'time': ('time', time),
            'lat': ('lat', lat, {'units': 'degrees_north'}),
            'lon': ('lon', lon, {'units': 'degrees_east'}),
        },
        attrs={'title': 'Geopotential height 1950–2021 (no-leap)'}
    )

    enc = {
        'time': {'units': 'days since 1950-01-01', 'calendar': 'standard', 'dtype': 'int32'},
        'hgt': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': np.nan},
        'lat': {'dtype': 'float32'},
        'lon': {'dtype': 'float32'},
    }

    ds.to_netcdf(out_path, format='NETCDF4', encoding=enc)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    filepath = "Data/era5_z500_daily_eu.nc"
    # pp_geopot(filepath)

    mat_path = "Data/ERA5_hgt_1950_2021_anomalies_withoutbsx.mat"
    out_path = "Data/hgt_anom_daily_eu.nc"
    mat_to_netcdf(mat_path, out_path)