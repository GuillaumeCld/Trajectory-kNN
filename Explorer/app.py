"""
TRAKNN Explorer — Interactive Streamlit app for spatiotemporal anomaly detection
and clustering case study.

Run with:
    streamlit run Explorer/app.py
"""

import sys
import os

# Project root → enables `import src.*`
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Explorer dir → enables `import tab_scoring`, `import data_utils`, etc.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, _here)

import torch
import streamlit as st

# ── Page config must be the very first st.* call ────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="TRAKNN Explorer",
    page_icon=":cyclone:",
)

# Tab modules are imported after set_page_config.
# plot_utils (imported transitively) sets matplotlib.use("Agg").
import tab_scoring
import tab_anomaly
import tab_cluster
import tab_analogue

# ============================================================
# Session state init
# ============================================================
for key in ["scores", "times", "lat", "lon",
            "load_params", "preproc_params",
            "cluster_results", "all_labels", "fields_embed",
            "top_idx_cluster", "run_params",
            "analogue_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
remove_sc_val   = st.sidebar.checkbox("Remove seasonal cycle", value=True)
cos_lat_val     = st.sidebar.checkbox("Cosine latitude weighting", value=True)
pxstd_val       = st.sidebar.checkbox("Pixel-wise standardisation", value=False)
use_pca_val     = st.sidebar.checkbox(
    "PCA (reduce spatial dims)",
    value=False,
    help="Apply PCA to the spatial fields before computing distances (base algorithm only). "
         "Also used as the default embedding step for cluster analysis.",
)

st.sidebar.divider()
st.sidebar.header("Algorithm")
traj_length_val = st.sidebar.slider("Trajectory length (days)", 1, 15, 5)
k_val           = st.sidebar.slider("Neighbours (k)", 1, 100, 10)
excl_val        = st.sidebar.slider("Exclusion zone (days)", 0, 30, 5)
algo_val        = st.sidebar.selectbox("Algorithm variant", ["base", "exclusion", "interval"])

device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
device_val = st.sidebar.selectbox("Device", device_options)

# ============================================================
# Sidebar params dict — passed to each tab's render function
# ============================================================
sidebar_params = dict(
    file_path=file_path,
    parameter=parameter,
    lon_min=lon_min_val,
    lon_max=lon_max_val,
    lat_min=lat_min_val,
    lat_max=lat_max_val,
    start_year=start_year_val,
    end_year=end_year_val,
    remove_leap=remove_leap_val,
    remove_sc=remove_sc_val,
    cos_lat=cos_lat_val,
    pxstd=pxstd_val,
    use_pca=use_pca_val,
    traj_length=traj_length_val,
    k=k_val,
    exclusion_zone=excl_val,
    algorithm=algo_val,
    device=device_val,
)

# ============================================================
# MAIN AREA — TABS
# ============================================================
st.title("TRAKNN Explorer: Rare Meteorological Pattern Discovery")

tab_score_t, tab_anomaly_t, tab_cluster_t, tab_analogue_t = st.tabs([
    "Scoring Pipeline",
    "Anomaly Explorer",
    "Cluster Analysis",
    "Analogue Search",
])

with tab_score_t:
    tab_scoring.render(sidebar_params)

with tab_anomaly_t:
    tab_anomaly.render(sidebar_params)

with tab_cluster_t:
    tab_cluster.render(sidebar_params)

with tab_analogue_t:
    tab_analogue.render(sidebar_params)
