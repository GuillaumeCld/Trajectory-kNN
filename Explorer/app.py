"""
TRAKNN Explorer — Interactive Streamlit app for spatiotemporal anomaly detection
and clustering case study.

Run with:
    streamlit run Explorer/app.py
"""

import sys
import os

# Ensure the project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.preprocessing as pp
import src.rarity_scoring_base
import src.rarity_scoring_exclusion
import src.rarity_scoring_interval

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# ============================================================
# Config
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="TRAKNN Explorer",
    page_icon=":cyclone:",
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ============================================================
# Session state init
# ============================================================
for key in ["scores", "times", "data", "lat", "lon",
            "cluster_results", "all_labels", "fields_embed",
            "top_idx_cluster", "run_params"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# Helper: load & preprocess data (cached by parameters)
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess(
    file_path, parameter,
    lon_min, lon_max, lat_min, lat_max,
    start_year, end_year,
    remove_leap, remove_seasonal_cycle,
    cos_lat_w, pixelwise_std,
):
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

    data = ds[parameter].transpose("time", lat_dim, lon_dim).values.astype(np.float32)
    times = pd.to_datetime(ds["time"].values)
    ds.close()

    if remove_leap:
        data, times = pp.remove_bisex_dailydata(data, times)
    if remove_seasonal_cycle:
        data = pp.remove_seasonal_cycle365(data, times)
    if cos_lat_w:
        data = pp.cos_lat_weighting(data, lat, len(lon))
    if pixelwise_std:
        shape = data.shape
        data_2d = data.reshape(shape[0], -1)
        data_2d = pp.pixelwise_standardize(data_2d)
        data = data_2d.reshape(shape)

    return data, times, lat, lon


# ============================================================
# Helper: compute scores (slow — not cached, use st.spinner)
# ============================================================
def run_scoring(data, traj_length, k, exclusion_zone, algorithm, device_str):
    algo_map = {
        "base": src.rarity_scoring_base.compute_distances_and_scores,
        "exclusion": src.rarity_scoring_exclusion.compute_distances_and_scores,
        "interval": src.rarity_scoring_interval.compute_distances_and_scores,
    }
    dev = None if device_str == "auto" else device_str
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    scores = algo_map[algorithm](
        data,
        traj_length,
        k,
        q_batch=1024,
        r_chunk=1024,
        device=dev,
        dtype=torch.float32,
        exclusion_zone=exclusion_zone,
    )
    return scores.numpy()


# ============================================================
# Helper: load existing scores from NPZ
# ============================================================
def load_scores_from_npz(uploaded):
    npz = np.load(uploaded)
    scores = npz["scores"]
    times = pd.to_datetime(npz["times"])
    return scores, times


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("TRAKNN Explorer")
st.sidebar.caption("Spatiotemporal Anomaly Detection")
st.sidebar.divider()

st.sidebar.header("Data")
file_path = st.sidebar.text_input(
    "NetCDF File Path",
    value="Data/era5_msl_daily_eu.nc",
    help="Path to the NetCDF file relative to the project root.",
)
parameter = st.sidebar.text_input(
    "Variable name",
    value="msl",
    help="Name of the variable in the dataset (e.g. msl, t2m, z500).",
)

with st.sidebar.expander("Spatial filter", expanded=False):
    c1, c2 = st.columns(2)
    lon_min_val = c1.number_input("Lon min", value=None, placeholder="—", format="%.1f")
    lon_max_val = c2.number_input("Lon max", value=None, placeholder="—", format="%.1f")
    lat_min_val = c1.number_input("Lat min", value=None, placeholder="—", format="%.1f")
    lat_max_val = c2.number_input("Lat max", value=None, placeholder="—", format="%.1f")

with st.sidebar.expander("Temporal filter", expanded=False):
    c1, c2 = st.columns(2)
    start_year_val = c1.number_input("Start year", value=None, step=1, placeholder="—", format="%d")
    end_year_val = c2.number_input("End year", value=None, step=1, placeholder="—", format="%d")

st.sidebar.divider()
st.sidebar.header("Preprocessing")
remove_leap_val = st.sidebar.checkbox("Remove leap days", value=True)
remove_sc_val = st.sidebar.checkbox("Remove seasonal cycle", value=True)
cos_lat_val = st.sidebar.checkbox("Cosine latitude weighting", value=True)
pxstd_val = st.sidebar.checkbox("Pixel-wise standardisation", value=False)

st.sidebar.divider()
st.sidebar.header("Algorithm")
traj_length_val = st.sidebar.slider("Trajectory length (days)", 1, 15, 5)
k_val = st.sidebar.slider("Neighbours (k)", 1, 100, 10)
excl_val = st.sidebar.slider("Exclusion zone (days)", 0, 30, 5)
algo_val = st.sidebar.selectbox("Algorithm variant", ["base", "exclusion", "interval"])

device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
device_val = st.sidebar.selectbox("Device", device_options)

# ============================================================
# MAIN AREA — TABS
# ============================================================
st.title("TRAKNN Explorer: Rare Meteorological Pattern Discovery")

tab_score, tab_anomaly, tab_cluster = st.tabs([
    "Scoring Pipeline",
    "Anomaly Explorer",
    "Cluster Analysis",
])

# ============================================================
# TAB 1 — SCORING PIPELINE
# ============================================================
with tab_score:
    st.header("Scoring Pipeline")
    st.markdown(
        "Load a NetCDF file, apply preprocessing, and compute trajectory-kNN rarity scores. "
        "Configure parameters in the sidebar."
    )

    col_btn, col_or, col_upload = st.columns([2, 0.3, 2])

    with col_btn:
        run_btn = st.button(
            "Run Scoring",
            type="primary",
            use_container_width=True,
            help="Load data and compute scores from scratch.",
        )

    with col_or:
        st.markdown("<div style='text-align:center;margin-top:8px'>or</div>", unsafe_allow_html=True)

    with col_upload:
        uploaded_npz = st.file_uploader(
            "Upload pre-computed scores (.npz)",
            type=["npz"],
            help="A .npz file with 'scores' and 'times' arrays (output of score.py).",
        )

    if uploaded_npz is not None:
        try:
            sc, ti = load_scores_from_npz(uploaded_npz)
            st.session_state.scores = sc
            st.session_state.times = ti
            st.success(f"Loaded {len(sc)} scores from uploaded file.")
        except Exception as e:
            st.error(f"Failed to load NPZ: {e}")

    if run_btn:
        if not os.path.exists(file_path):
            st.error(f"File not found: `{file_path}`")
        else:
            try:
                with st.spinner("Loading & preprocessing data..."):
                    data, times, lat, lon = load_and_preprocess(
                        file_path, parameter,
                        lon_min_val, lon_max_val,
                        lat_min_val, lat_max_val,
                        start_year_val, end_year_val,
                        remove_leap_val, remove_sc_val,
                        cos_lat_val, pxstd_val,
                    )
                T, H, W = data.shape
                st.info(
                    f"Data loaded: **{T}** time steps, **{H}×{W}** grid "
                    f"({lat.min():.1f}N–{lat.max():.1f}N, {lon.min():.1f}E–{lon.max():.1f}E)"
                )
                st.session_state.data = data
                st.session_state.times = times
                st.session_state.lat = lat
                st.session_state.lon = lon

                progress_bar = st.progress(0, text="Computing trajectory distances...")
                scores = run_scoring(
                    data, traj_length_val, k_val, excl_val, algo_val, device_val
                )
                progress_bar.progress(100, text="Done.")
                st.session_state.scores = scores
                st.session_state.run_params = dict(
                    traj_length=traj_length_val, k=k_val,
                    exclusion_zone=excl_val, algorithm=algo_val,
                )
                st.success(f"Computed **{len(scores)}** rarity scores.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    # Show summary if scores are available
    if st.session_state.scores is not None:
        scores = st.session_state.scores
        times = st.session_state.times
        score_times = times[:len(scores)]

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time steps", len(scores))
        c2.metric("Max score", f"{scores.max():.4f}")
        c3.metric("Mean score", f"{scores.mean():.4f}")
        c4.metric("Min score", f"{scores.min():.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = px.histogram(
                x=scores, nbins=100,
                labels={"x": "Rarity score", "y": "Count"},
                title="Score distribution",
                color_discrete_sequence=["steelblue"],
            )
            fig_hist.update_layout(height=320, margin=dict(t=40, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            df_ts = pd.DataFrame({"time": score_times, "score": scores})
            # monthly z-score (relative anomaly)
            df_ts["month"] = pd.to_datetime(score_times).month
            stats = df_ts.groupby("month")["score"].agg(["mean", "std"])
            df_ts = df_ts.merge(stats, on="month")
            df_ts["z_score"] = (df_ts["score"] - df_ts["mean"]) / df_ts["std"].replace(0, np.nan)

            fig_ts = px.line(
                df_ts, x="time", y="score",
                title="Rarity score time series",
                labels={"score": "Score", "time": "Date"},
                color_discrete_sequence=["#2c7fb8"],
            )
            fig_ts.update_layout(height=320, margin=dict(t=40, b=30))
            st.plotly_chart(fig_ts, use_container_width=True)

        # Export
        npz_buf = None
        st.download_button(
            "Download scores as CSV",
            pd.DataFrame({"time": score_times, "score": scores}).to_csv(index=False),
            file_name="scores.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 2 — ANOMALY EXPLORER
# ============================================================
with tab_anomaly:
    st.header("Anomaly Explorer")

    if st.session_state.scores is None:
        st.info("Run the scoring pipeline (or upload a .npz) to explore anomalies.")
    else:
        scores = st.session_state.scores
        times = st.session_state.times
        data = st.session_state.data
        lat = st.session_state.lat
        lon = st.session_state.lon
        score_times = times[:len(scores)]

        c1, c2 = st.columns([1, 3])
        with c1:
            n_top = st.number_input(
                "Top-N events", min_value=5, max_value=500, value=100, step=5,
            )
            score_type = st.radio(
                "Ranking by",
                ["Absolute score", "Monthly z-score"],
                help="Monthly z-score standardises scores relative to the seasonal cycle.",
            )

        top_idx = np.argsort(-scores)[:int(n_top)]
        if score_type == "Monthly z-score":
            df_temp = pd.DataFrame({"time": score_times, "score": scores})
            df_temp["month"] = pd.to_datetime(score_times).month
            stats = df_temp.groupby("month")["score"].agg(["mean", "std"])
            df_temp = df_temp.merge(stats, on="month")
            df_temp["z"] = (df_temp["score"] - df_temp["mean"]) / df_temp["std"].replace(0, np.nan)
            z_scores = df_temp["z"].values
            top_idx = np.argsort(-np.nan_to_num(z_scores))[:int(n_top)]
            rank_values = z_scores[top_idx]
            rank_label = "Monthly z-score"
        else:
            rank_values = scores[top_idx]
            rank_label = "Rarity score"

        top_times = score_times[top_idx]
        top_raw_scores = scores[top_idx]

        # Time series with top events highlighted
        with c2:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=score_times, y=scores,
                mode="lines", name="Score",
                line=dict(color="#aec7e8", width=1),
            ))
            fig_ts.add_trace(go.Scatter(
                x=top_times, y=top_raw_scores,
                mode="markers", name=f"Top {n_top}",
                marker=dict(color="#d62728", size=5, symbol="circle"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.4f}<extra></extra>",
            ))
            fig_ts.update_layout(
                title="Rarity score time series",
                xaxis_title="Date", yaxis_title="Score",
                height=280, margin=dict(t=40, b=20, l=40, r=20),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.divider()

        # Table + field map side-by-side
        col_tbl, col_map = st.columns([1, 2])

        with col_tbl:
            df_top = pd.DataFrame({
                "Rank": range(1, int(n_top) + 1),
                "Date": [t.strftime("%Y-%m-%d") for t in top_times],
                rank_label: np.round(rank_values, 4),
            })
            selected_rows = st.dataframe(
                df_top,
                use_container_width=True,
                height=420,
                on_select="rerun",
                selection_mode="single-row",
            )
            st.download_button(
                "Download top events CSV",
                df_top.to_csv(index=False),
                file_name="top_events.csv",
                mime="text/csv",
            )

        with col_map:
            try:
                sel = selected_rows.selection.rows
            except AttributeError:
                sel = []
            rank_sel = int(sel[0]) + 1 if sel else 1
            event_idx = top_idx[rank_sel - 1]
            event_date = score_times[event_idx]

            if data is not None and lat is not None and lon is not None:
                field = data[event_idx]
                vmax = float(np.nanpercentile(np.abs(field), 98))
                fig_map = px.imshow(
                    field,
                    x=lon, y=lat,
                    color_continuous_scale="RdBu_r",
                    zmin=-vmax, zmax=vmax,
                    origin="lower",
                    labels={"x": "Longitude", "y": "Latitude", "color": parameter},
                    title=(
                        f"Rank {rank_sel} — {event_date.strftime('%Y-%m-%d')} "
                        f"| score = {scores[event_idx]:.4f}"
                    ),
                )
                fig_map.update_layout(
                    height=420, margin=dict(t=50, b=20, l=10, r=10),
                    coloraxis_colorbar=dict(title=parameter),
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(
                    "Field maps are only available when scores were computed "
                    "from a file in this session (not from an uploaded NPZ)."
                )


# ============================================================
# TAB 3 — CLUSTER ANALYSIS
# ============================================================
with tab_cluster:
    st.header("Cluster Analysis")
    st.markdown(
        "Extract the top-N anomalous events, optionally reduce dimensionality with PCA, "
        "then run K-means clustering across a range of K. Composites and temporal "
        "distributions are shown interactively."
    )

    if st.session_state.scores is None:
        st.info("Run the scoring pipeline (or upload a .npz) first.")
    else:
        scores = st.session_state.scores
        times = st.session_state.times
        data = st.session_state.data
        lat = st.session_state.lat
        lon = st.session_state.lon
        score_times = times[:len(scores)]

        # --- Cluster parameters ---
        with st.expander("Clustering parameters", expanded=True):
            cc1, cc2, cc3, cc4 = st.columns(4)
            n_top_cl = cc1.number_input("Top-N events", 10, 500, 100, step=10)
            k_min_cl = int(cc2.number_input("K min", 2, 15, 2, step=1))
            k_max_cl = int(cc3.number_input("K max", 2, 20, 10, step=1))
            n_init_cl = int(cc4.number_input("K-means n_init", 10, 200, 50, step=10))
            use_pca_cl = st.checkbox("PCA preprocessing (retain 95% variance)", value=True)

        run_cl_btn = st.button("Run Cluster Analysis", type="primary")

        if run_cl_btn:
            if data is None:
                st.error(
                    "Raw data not available. Re-run the scoring pipeline from a file "
                    "(uploaded NPZ files do not include the raw fields)."
                )
            else:
                try:
                    with st.spinner("Extracting top events..."):
                        top_idx_cl = np.sort(np.argsort(-scores)[:int(n_top_cl)])
                        fields = data[top_idx_cl]        # (n, H, W)
                        n_ev, nlat_c, nlon_c = fields.shape
                        base_fields = fields.copy()

                        fields_2d = fields.reshape(n_ev, nlat_c * nlon_c)
                        valid_cols = ~np.isnan(fields_2d).any(axis=0)
                        fields_2d = fields_2d[:, valid_cols]

                    with st.spinner("PCA..." if use_pca_cl else "Preparing features..."):
                        if use_pca_cl:
                            pca_model = PCA()
                            score_pca = pca_model.fit_transform(fields_2d)
                            cumvar = np.cumsum(pca_model.explained_variance_ratio_)
                            npc = max(int(np.argmax(cumvar >= 0.95)) + 1, 3)
                            fields_embed = score_pca[:, :npc]
                            st.info(f"PCA: retained **{npc}** components (≥95% variance)")
                        else:
                            fields_embed = fields_2d

                    kmax_actual = min(k_max_cl, n_ev - 1)
                    kmin_actual = k_min_cl

                    all_labels_map = {}
                    avg_sil_map = {}
                    inertias_map = {}

                    progress = st.progress(0, text="K-means sweep...")
                    k_range = list(range(kmin_actual, kmax_actual + 1))
                    for step_i, K in enumerate(k_range):
                        km = KMeans(
                            n_clusters=K, n_init=n_init_cl,
                            max_iter=1000, random_state=0, init="k-means++",
                        )
                        labels = km.fit_predict(fields_embed)
                        all_labels_map[K] = labels
                        sil = silhouette_samples(fields_embed, labels)
                        avg_sil_map[K] = float(sil.mean())
                        inertias_map[K] = float(km.inertia_)
                        progress.progress(
                            int((step_i + 1) / len(k_range) * 100),
                            text=f"K={K} — silhouette={avg_sil_map[K]:.3f}",
                        )

                    best_k = max(avg_sil_map, key=avg_sil_map.get)
                    progress.progress(100, text=f"Done. Best K = {best_k}")

                    st.session_state.cluster_results = {
                        "best_k": best_k,
                        "avg_sil": avg_sil_map,
                        "inertias": inertias_map,
                        "base_fields": base_fields,
                        "top_idx": top_idx_cl,
                        "n_ev": n_ev,
                        "nlat": nlat_c,
                        "nlon": nlon_c,
                        "kmin": kmin_actual,
                        "kmax": kmax_actual,
                    }
                    st.session_state.all_labels = all_labels_map
                    st.session_state.fields_embed = fields_embed
                    st.session_state.top_idx_cluster = top_idx_cl
                    st.success(
                        f"Best K = **{best_k}** "
                        f"(mean silhouette = {avg_sil_map[best_k]:.3f})"
                    )

                except Exception as e:
                    st.error(f"Clustering error: {e}")
                    st.exception(e)

        # --- Results ---
        if st.session_state.cluster_results is not None:
            cr = st.session_state.cluster_results
            all_labels_map = st.session_state.all_labels
            fields_embed = st.session_state.fields_embed
            top_idx_cl = st.session_state.top_idx_cluster

            best_k = cr["best_k"]
            avg_sil_map = cr["avg_sil"]
            inertias_map = cr["inertias"]
            base_fields = cr["base_fields"]
            nlat_c = cr["nlat"]
            nlon_c = cr["nlon"]
            kmin_cl = cr["kmin"]
            kmax_cl = cr["kmax"]

            event_times = score_times[top_idx_cl]
            ks = list(range(kmin_cl, kmax_cl + 1))

            st.divider()
            st.subheader("K Selection")

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=ks, y=[avg_sil_map[k] for k in ks],
                    mode="lines+markers",
                    line=dict(color="#2c3e50", width=2.5),
                    marker=dict(size=8),
                    name="Silhouette score",
                ))
                fig_sil.add_vline(
                    x=best_k, line_color="#e74c3c", line_dash="dash",
                    annotation_text=f"Best K={best_k}",
                    annotation_position="top right",
                )
                fig_sil.update_layout(
                    title="Mean Silhouette Score vs K",
                    xaxis_title="K", yaxis_title="Mean silhouette",
                    height=320, margin=dict(t=40, b=30),
                )
                st.plotly_chart(fig_sil, use_container_width=True)

            with pcol2:
                fig_inertia = go.Figure()
                fig_inertia.add_trace(go.Scatter(
                    x=ks, y=[inertias_map[k] for k in ks],
                    mode="lines+markers",
                    line=dict(color="#2c3e50", width=2.5),
                    marker=dict(size=8),
                    name="Inertia",
                ))
                fig_inertia.update_layout(
                    title="Inertia (Elbow Method) vs K",
                    xaxis_title="K", yaxis_title="Inertia",
                    height=320, margin=dict(t=40, b=30),
                )
                st.plotly_chart(fig_inertia, use_container_width=True)

            st.divider()

            # K selector
            selected_k = st.selectbox(
                "View results for K =",
                ks,
                index=ks.index(best_k),
                format_func=lambda k: f"K={k}" + (" (best)" if k == best_k else ""),
            )
            view_cl = all_labels_map[selected_k]

            # --- Composites ---
            st.subheader(f"Cluster Composites (K={selected_k})")

            composites = []
            counts = []
            for k_i in range(selected_k):
                members = np.where(view_cl == k_i)[0]
                counts.append(len(members))
                composites.append(base_fields[members].mean(axis=0) if len(members) > 0
                                  else np.zeros((nlat_c, nlon_c)))

            n_cols = min(selected_k, 3)
            comp_cols = st.columns(n_cols)
            for k_i in range(selected_k):
                col_idx = k_i % n_cols
                with comp_cols[col_idx]:
                    vmax = float(np.nanpercentile(np.abs(composites[k_i]), 95)) if counts[k_i] > 0 else 1.0
                    if lat is not None and lon is not None:
                        fig_comp = px.imshow(
                            composites[k_i],
                            x=lon, y=lat,
                            color_continuous_scale="RdBu_r",
                            zmin=-vmax, zmax=vmax,
                            origin="lower",
                            labels={"x": "Lon", "y": "Lat", "color": parameter},
                            title=f"Cluster {k_i + 1}  (n={counts[k_i]})",
                        )
                    else:
                        fig_comp = px.imshow(
                            composites[k_i],
                            color_continuous_scale="RdBu_r",
                            zmin=-vmax, zmax=vmax,
                            title=f"Cluster {k_i + 1}  (n={counts[k_i]})",
                        )
                    fig_comp.update_layout(
                        height=280, margin=dict(t=40, b=5, l=5, r=5),
                        coloraxis_showscale=(k_i == n_cols - 1),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Cluster sizes
            fig_sizes = px.bar(
                x=[f"Cluster {k_i + 1}" for k_i in range(selected_k)],
                y=counts,
                labels={"x": "Cluster", "y": "Count"},
                title="Cluster Membership",
                color_discrete_sequence=["steelblue"],
                text=counts,
            )
            fig_sizes.update_traces(textposition="outside")
            fig_sizes.update_layout(height=250, margin=dict(t=40, b=20))
            st.plotly_chart(fig_sizes, use_container_width=True)

            # --- Temporal distributions ---
            st.divider()
            st.subheader("Temporal Distributions")

            cl_tabs = st.tabs([f"Cluster {k_i + 1}" for k_i in range(selected_k)])
            for k_i, cl_tab in enumerate(cl_tabs):
                with cl_tab:
                    members = np.where(view_cl == k_i)[0]
                    member_times = pd.to_datetime(event_times[members])
                    member_scores = scores[top_idx_cl[members]]

                    tcol1, tcol2 = st.columns(2)
                    with tcol1:
                        month_counts = (
                            pd.Series(member_times.month)
                            .value_counts()
                            .reindex(range(1, 13), fill_value=0)
                            .sort_index()
                        )
                        fig_month = px.bar(
                            x=month_counts.index, y=month_counts.values,
                            labels={"x": "Month", "y": "Count"},
                            title=f"Cluster {k_i + 1}: Monthly distribution",
                            color_discrete_sequence=["teal"],
                        )
                        fig_month.update_xaxes(
                            tickvals=list(range(1, 13)),
                            ticktext=MONTH_LABELS,
                        )
                        fig_month.update_layout(height=300, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_month, use_container_width=True)

                    with tcol2:
                        year_counts = (
                            pd.Series(member_times.year)
                            .value_counts()
                            .sort_index()
                        )
                        fig_year = px.bar(
                            x=year_counts.index, y=year_counts.values,
                            labels={"x": "Year", "y": "Count"},
                            title=f"Cluster {k_i + 1}: Yearly distribution",
                            color_discrete_sequence=["coral"],
                        )
                        fig_year.update_layout(height=300, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_year, use_container_width=True)

                    # Silhouette per member
                    if len(members) > 1 and fields_embed is not None:
                        sil_all = silhouette_samples(fields_embed, view_cl)
                        sil_members = sil_all[members]
                        sort_idx = np.argsort(sil_members)
                        fig_sil_m = go.Figure()
                        fig_sil_m.add_trace(go.Bar(
                            x=sil_members[sort_idx],
                            y=list(range(len(members))),
                            orientation="h",
                            marker_color="steelblue",
                            name="Silhouette",
                        ))
                        fig_sil_m.add_vline(x=0, line_color="black", line_width=1)
                        fig_sil_m.update_layout(
                            title=f"Cluster {k_i + 1}: per-event silhouette values",
                            xaxis_title="Silhouette value",
                            yaxis_title="Event (sorted)",
                            height=max(200, 12 * len(members)),
                            margin=dict(t=40, b=20),
                        )
                        st.plotly_chart(fig_sil_m, use_container_width=True)

                    # Date table
                    df_cl = pd.DataFrame({
                        "Date": [t.strftime("%Y-%m-%d") for t in member_times],
                        "Rarity score": np.round(member_scores, 4),
                    }).sort_values("Rarity score", ascending=False)
                    st.dataframe(df_cl, use_container_width=True, height=250)

            # --- Download ---
            st.divider()
            df_export = pd.DataFrame({
                "date": [t.strftime("%Y-%m-%d") for t in event_times],
                "cluster": view_cl + 1,
                "rarity_score": scores[top_idx_cl],
            })
            st.download_button(
                "Download cluster assignments CSV",
                df_export.to_csv(index=False),
                file_name=f"cluster_K{selected_k}_assignments.csv",
                mime="text/csv",
            )
