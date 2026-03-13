"""
Tab 1 — Scoring Pipeline.

Provides a render(p) function that draws the full tab content.
`p` is the sidebar_params dict assembled by app.py.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data_utils import load_data_cached, run_scoring, load_scores_from_npz


def render(p):
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
            width="stretch",
            help="Load data and compute scores from scratch.",
        )

    # with col_or:
    #     st.markdown(
    #         "<div style='text-align:center;margin-top:8px'>or</div>",
    #         unsafe_allow_html=True,
    #     )

    # with col_upload:
    #     uploaded_npz = st.file_uploader(
    #         "Upload pre-computed scores (.npz)",
    #         type=["npz"],
    #         help="A .npz file with 'scores' and 'times' arrays (output of score.py).",
    #     )

    # if uploaded_npz is not None:
    #     try:
    #         sc, ti = load_scores_from_npz(uploaded_npz)
    #         sc = (sc - sc.min()) / max(float(sc.max() - sc.min()), 1e-12)
    #         st.session_state.scores = sc
    #         st.session_state.times = ti
    #         st.success(f"Loaded {len(sc)} scores from uploaded file.")
    #     except Exception as e:
    #         st.error(f"Failed to load NPZ: {e}")

    if run_btn:
        if not os.path.exists(p["file_path"]):
            st.error(f"File not found: `{p['file_path']}`")
        else:
            try:
                with st.spinner("Loading and preprocessing data..."):
                    data, times, lat, lon = load_data_cached(
                        p["file_path"], p["parameter"],
                        p["lon_min"], p["lon_max"],
                        p["lat_min"], p["lat_max"],
                        p["start_year"], p["end_year"],
                        p["remove_leap"], p["remove_sc"],
                        p["cos_lat"], p["pxstd"],
                    )
                T, H, W = data.shape
                st.info(
                    f"Data loaded: **{T}** time steps, **{H}×{W}** grid "
                    f"({lat.min():.1f}N–{lat.max():.1f}N, "
                    f"{lon.min():.1f}E–{lon.max():.1f}E)"
                )
                st.session_state.times = times
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.load_params = dict(
                    file_path=p["file_path"], parameter=p["parameter"],
                    lon_min=p["lon_min"], lon_max=p["lon_max"],
                    lat_min=p["lat_min"], lat_max=p["lat_max"],
                    start_year=p["start_year"], end_year=p["end_year"],
                )
                st.session_state.preproc_params = dict(
                    remove_leap=p["remove_leap"],
                    remove_sc=p["remove_sc"],
                    cos_lat=p["cos_lat"],
                    pxstd=p["pxstd"],
                )

                progress_bar = st.progress(0, text="Computing trajectory distances...")
                scores = run_scoring(
                    data,
                    p["traj_length"], p["k"], p["exclusion_zone"],
                    p["algorithm"], p["device"],
                    use_pca=p["use_pca"],
                )
                progress_bar.progress(100, text="Done.")
                scores = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
                st.session_state.scores = scores
                st.session_state.run_params = dict(
                    traj_length=p["traj_length"], k=p["k"],
                    exclusion_zone=p["exclusion_zone"],
                    algorithm=p["algorithm"],
                    use_pca=p["use_pca"],
                )
                st.success(f"Computed **{len(scores)}** rarity scores.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    # --- Summary (shown whenever scores exist) ---
    if st.session_state.scores is not None:
        scores = st.session_state.scores
        times = st.session_state.times
        score_times = times[:len(scores)]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Time steps", len(scores))
        c2.metric("Mean score", f"{scores.mean():.2f}")
        c3.metric(f"Percentiles 25 | 50 | 75 | 95", f" {np.percentile(scores,25):.2f} | {np.percentile(scores,50):.2f} | {np.percentile(scores,75):.2f} | {np.percentile(scores,95):.2f}")
        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = px.histogram(
                x=scores, nbins=100,
                labels={"x": "Rarity score", "y": "Count"},
                title="Score distribution",
                color_discrete_sequence=["steelblue"],
            )
            fig_hist.update_layout(height=320, margin=dict(t=40, b=30))
            st.plotly_chart(fig_hist, width="stretch")

        with col_r:
            df_ts = pd.DataFrame({"time": score_times, "score": scores})
            df_ts["month"] = pd.to_datetime(score_times).month
            stats = df_ts.groupby("month")["score"].agg(["mean", "std"])
            df_ts = df_ts.merge(stats, on="month")
            df_ts["z_score"] = (
                (df_ts["score"] - df_ts["mean"]) / df_ts["std"].replace(0, np.nan)
            )
            fig_ts = px.line(
                df_ts, x="time", y="score",
                title="Rarity score time series",
                labels={"score": "Score", "time": "Date"},
                color_discrete_sequence=["#2c7fb8"],
            )
            fig_ts.update_layout(height=320, margin=dict(t=40, b=30))
            st.plotly_chart(fig_ts, width="stretch")

        st.download_button(
            "Download scores as CSV",
            pd.DataFrame({"time": score_times, "score": scores}).to_csv(index=False),
            file_name="scores.csv",
            mime="text/csv",
        )
