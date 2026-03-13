"""
Tab 2 — Anomaly Explorer.

Provides a render(p) function that draws the full tab content.
`p` is the sidebar_params dict assembled by app.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from data_utils import compute_seasonal_cycle_cached, load_stage1
from plot_utils import make_cartopy_fig


_RARITY_COLOR = {"common": "green", "rare": "orange", "extreme": "red"}


def _rarity_label(F_hat, tau_rare, tau_extreme):
    """Return (label, streamlit_color) for an empirical CDF value."""
    if F_hat >= tau_extreme:
        return "extreme", "red"
    elif F_hat >= tau_rare:
        return "rare", "orange"
    else:
        return "common", "green"


def render(p):
    st.header("Anomaly Explorer")

    if st.session_state.scores is None:
        st.info("Run the scoring pipeline (or upload a .npz) to explore anomalies.")
        return

    scores = st.session_state.scores
    times = st.session_state.times
    lat = st.session_state.lat
    lon = st.session_state.lon
    load_params = st.session_state.load_params
    preproc_params = st.session_state.preproc_params
    score_times = times[:len(scores)]

    # Seasonal cycle: (365, H, W) + index arrays — no extra (T, H, W) allocation
    sc_available = False
    if load_params is not None:
        seasonal_cycle, md_index, noleap_orig_idx = compute_seasonal_cycle_cached(
            **load_params
        )
        raw_stage1, _, lat, lon = load_stage1(load_params)
        sc_available = True

    c1, c2 = st.columns([1, 3])
    with c1:
        n_top = st.number_input(
            "Top-N events", min_value=5, max_value=500, value=100, step=5,
        )
        score_type = st.radio(
            "Ranking by",
            ["Absolute score"],
            help="Monthly z-score standardises scores relative to the seasonal cycle.",
        )

    tau_rare, tau_extreme = 0.90, 0.99

    top_idx = np.argsort(-scores)[:int(n_top)]
    if score_type == "Monthly z-score":
        df_temp = pd.DataFrame({"time": score_times, "score": scores})
        df_temp["month"] = pd.to_datetime(score_times).month
        stats = df_temp.groupby("month")["score"].agg(["mean", "std"])
        df_temp = df_temp.merge(stats, on="month")
        df_temp["z"] = (
            (df_temp["score"] - df_temp["mean"]) / df_temp["std"].replace(0, np.nan)
        )
        z_scores = df_temp["z"].values
        top_idx = np.argsort(-np.nan_to_num(z_scores))[:int(n_top)]
        rank_values = z_scores[top_idx]
        rank_label = "Monthly z-score"
    else:
        rank_values = scores[top_idx]
        rank_label = "Rarity score"

    top_times = score_times[top_idx]
    top_raw_scores = scores[top_idx]

    # Vectorised empirical CDF for the top events
    sorted_scores = np.sort(scores)
    top_F_hat = np.searchsorted(sorted_scores, top_raw_scores, side="right") / len(scores)
    top_rarity = [_rarity_label(f, tau_rare, tau_extreme)[0] for f in top_F_hat]

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
        st.plotly_chart(fig_ts, width="stretch")

    st.divider()

    # Table + field map side-by-side
    col_tbl, col_map = st.columns([1, 2])

    with col_tbl:
        df_top = pd.DataFrame({
            "Rank": range(1, int(n_top) + 1),
            "Date": [t.strftime("%Y-%m-%d") for t in top_times],
            rank_label: np.round(rank_values, 4),
            "Rarity": top_rarity,
        })
        selected_rows = st.dataframe(
            df_top,
            width="stretch",
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
        # Sync selectbox default with dataframe row click
        try:
            sel_rows = selected_rows.selection.rows
        except AttributeError:
            sel_rows = []
        default_rank = int(sel_rows[0]) if sel_rows else 0

        event_labels = [
            f"Rank {r}  —  {d}  |  {v:.4f}"
            for r, d, v in zip(
                df_top["Rank"], df_top["Date"], np.round(rank_values, 4)
            )
        ]
        rank_sel_idx = st.selectbox(
            "Select event to display",
            options=range(len(event_labels)),
            index=default_rank,
            format_func=lambda i: event_labels[i],
        )

        event_idx = top_idx[rank_sel_idx]
        event_date = score_times[event_idx]
        event_F_hat = top_F_hat[rank_sel_idx]
        event_label, event_color = _rarity_label(event_F_hat, tau_rare, tau_extreme)

        if sc_available and lat is not None and lon is not None:
            # Anomaly = raw frame (no leap) minus its day-of-year climatology
            raw_frame = raw_stage1[noleap_orig_idx[event_idx]]
            display_field = raw_frame - seasonal_cycle[md_index[event_idx]]
            fig_map = make_cartopy_fig(
                display_field, lat, lon,
                title=(
                    f"Rank {rank_sel_idx + 1} — {event_date.strftime('%Y-%m-%d')} "
                    f"| score = {scores[event_idx]:.4f}"
                ),
                cbar_label=p["parameter"],
                figsize=(7, 4),
            )
            st.pyplot(fig_map, width="stretch")
            plt.close(fig_map)
        else:
            st.info(
                "Field maps are only available when scores were computed "
                "from a file in this session (not from an uploaded NPZ)."
            )

        st.markdown(
            f":{event_color}[**{event_label.upper()}**]  "
            f"— {event_F_hat * 100:.1f}th percentile"
        )
