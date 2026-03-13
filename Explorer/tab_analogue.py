"""
Tab 4 — Analogue Search.

Provides a render(p) function that draws the full tab content.
`p` is the sidebar_params dict assembled by app.py.

Given a query trajectory (start date + length), finds the k most similar
historical trajectories using blocked L2² distance on the preprocessed data
already loaded in memory via the Stage-2 cache.
"""

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from src.rarity_scoring_base import blocked_norm_compute

from data_utils import load_data_cached, compute_seasonal_cycle_cached, load_stage1
from plot_utils import make_cartopy_fig


def _rarity_label(F_hat, tau_rare, tau_extreme):
    """Return (label, streamlit_color) for an empirical CDF value."""
    if F_hat >= tau_extreme:
        return "extreme", "red"
    elif F_hat >= tau_rare:
        return "rare", "orange"
    else:
        return "common", "green"


# ── Core analogue search ─────────────────────────────────────────────────────

@torch.no_grad()
def _run_analogue_search(data, query_idx, traj_length, k, exclusion_zone, device_str):
    """Compute trajectory-distance analogues for a given query window.

    Parameters
    ----------
    data : np.ndarray, shape (T, H, W)
        Preprocessed spatiotemporal data.
    query_idx : int
        Start index of the query trajectory in `data`.
    traj_length : int
        Number of time steps per trajectory.
    k : int
        Number of analogues to return.
    exclusion_zone : int
        Minimum separation (in days) between the start of analogue
        trajectories and the query window edges.
    device_str : str
        "cpu", "cuda", or "auto".

    Returns
    -------
    analogue_idx : np.ndarray, shape (k,)
        Start indices of the top-k best analogues, sorted by distance.
    distances : np.ndarray, shape (k,)
        Corresponding summed L2² trajectory distances.
    """
    T, H, W = data.shape
    D = H * W
    N = T - traj_length + 1  # number of valid trajectory start positions

    if query_idx + traj_length > T:
        raise ValueError(
            f"Query window [{query_idx}, {query_idx + traj_length}) exceeds data length {T}."
        )
    if N < 1:
        raise ValueError("Data too short for the requested trajectory length.")

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_str)

    # Move full dataset to device (float32)
    data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    X = data.reshape(T, D).to(torch.float32).to(dev)  # (T, D)
    X_query = X[query_idx: query_idx + traj_length]                      # (L, D)

    # Precompute squared norms
    r_chunk = 1024

    norms_all = blocked_norm_compute(X, r_chunk, dev)# (T,)
    norms_query = (X_query * X_query).sum(dim=1) # (L,)

    # Accumulate trajectory distances
    distances_traj = torch.zeros(N, dtype=torch.float32, device=dev)

    for t_offset in range(traj_length):
        q_row = X_query[t_offset]          # (D,)
        q_norm = norms_query[t_offset]     # scalar

        # Historical frames for this offset: indices [t_offset, t_offset + N)
        hist_slice = X[t_offset: t_offset + N]       # (N, D)
        hist_norms = norms_all[t_offset: t_offset + N]  # (N,)

        for cs in range(0, N, r_chunk):
            ce = min(cs + r_chunk, N)
            cols = hist_slice[cs:ce]
            col_norms = hist_norms[cs:ce]
            dists = q_norm + col_norms - 2.0 * (q_row @ cols.T)
            distances_traj[cs:ce] += dists.clamp(min=0.0)

    distances_np = distances_traj.clamp(min=0.0).cpu().numpy()

    # Apply exclusion zone: exclude start indices too close to the query
    excl_lo = query_idx - traj_length - exclusion_zone + 1
    excl_hi = query_idx + traj_length + exclusion_zone - 1
    for i in range(N):
        if excl_lo <= i <= excl_hi:
            distances_np[i] = np.inf

    # Top-k smallest (finite) distances
    finite_idx = np.where(np.isfinite(distances_np))[0]
    if len(finite_idx) == 0:
        raise ValueError("All trajectories excluded — try reducing the exclusion zone.")
    k_eff = min(k, len(finite_idx))
    part = np.argpartition(distances_np[finite_idx], k_eff - 1)[:k_eff]
    top_idx = finite_idx[part]
    top_idx = top_idx[np.argsort(distances_np[top_idx])]

    return top_idx, np.sqrt(distances_np[top_idx])



# ── Tab render ───────────────────────────────────────────────────────────────

def render(p):
    st.header("Analogue Search")
    st.markdown(
        "Select a query period, then find the *k* most similar historical trajectories "
        "based on summed L2² distance over the preprocessed field."
    )

    load_params = st.session_state.load_params
    preproc_params = st.session_state.preproc_params
    times = st.session_state.times

    if load_params is None or preproc_params is None or times is None:
        st.info(
            "Run the scoring pipeline from a file first "
            "(NPZ uploads do not include the raw fields required for analogue search)."
        )
        return

    # Seasonal cycle for anomaly maps — mirrors the anomaly tab approach
    seasonal_cycle, md_index, noleap_orig_idx = compute_seasonal_cycle_cached(**load_params)
    raw_stage1, _, lat, lon = load_stage1(load_params)

    # ── Query definition ─────────────────────────────────────────────────────
    t_min = pd.Timestamp(times[0]).date()
    t_max = pd.Timestamp(times[-1]).date()

    qc1, qc2, qc3 = st.columns([2, 1, 2])

    query_date = qc1.date_input(
        "Query start date",
        value=t_min,
        min_value=t_min,
        max_value=t_max,
        help="First day of the query window.",
    )
    traj_len_an = int(qc2.number_input(
        "Duration (days)",
        min_value=1, max_value=60,
        value=p["traj_length"], step=1,
        help="Number of consecutive days in the query window.",
    ))

    # Compute and display the end date
    query_end = query_date + pd.Timedelta(days=traj_len_an - 1)
    end_clamped = query_end > t_max
    end_label = f":red[{query_end}  ⚠ beyond data]" if end_clamped else str(query_end)
    qc3.markdown(f"**Query end date**\n\n{end_label}", help="Inclusive last day of the query window.")

    with st.expander("Search parameters", expanded=False):
        sc1, sc2 = st.columns(2)
        k_an = int(sc1.number_input(
            "Analogues (k)",
            min_value=1, max_value=200,
            value=p["k"], step=1,
        ))
        excl_an = int(sc2.number_input(
            "Exclusion zone (days)",
            min_value=0, max_value=365,
            value=p["exclusion_zone"], step=1,
            help="Trajectories whose start is within this many days of the query are excluded.",
        ))
        rc1, rc2 = st.columns(2)
        tau_rare = rc1.slider(
            "Rare threshold", 0.80, 0.99, 0.90, step=0.01, format="%.2f",
            help="Empirical percentile above which an event is labelled *rare*.",
        )
        tau_extreme = rc2.slider(
            "Extreme threshold", 0.90, 1.00, 0.99, step=0.01, format="%.2f",
            help="Empirical percentile above which an event is labelled *extreme*.",
        )
        tau_extreme = max(tau_extreme, tau_rare + 0.01)

    run_an_btn = st.button(
        "Search analogues", type="primary",
        disabled=bool(end_clamped),
    )

    if run_an_btn:
        # Map query_date → index in times array
        times_pd = pd.to_datetime(times)
        query_ts = pd.Timestamp(query_date)
        idx_arr = np.where(times_pd.normalize() == query_ts.normalize())[0]

        if len(idx_arr) == 0:
            st.error(
                f"Date {query_date} not found in the loaded time series. "
                "Try a different date."
            )
        else:
            query_idx = int(idx_arr[0])

            # Guard: query window must fit in data
            if query_idx + traj_len_an > len(times):
                st.error(
                    f"Query window extends beyond the end of the data "
                    f"({pd.Timestamp(times[-1]).date()}). "
                    "Reduce trajectory length or choose an earlier start date."
                )
            else:
                try:
                    with st.spinner("Loading preprocessed data..."):
                        data, _, lat, lon = load_data_cached(
                            **load_params, **preproc_params
                        )

                    with st.spinner("Searching for analogues..."):
                        analogue_idx, dists = _run_analogue_search(
                            data, query_idx, traj_len_an,
                            k_an, excl_an, p["device"],
                        )

                    st.session_state.analogue_results = {
                        "query_idx": query_idx,
                        "query_date": query_date,
                        "traj_length": traj_len_an,
                        "analogue_idx": analogue_idx,
                        "distances": dists,
                    }
                    st.success(
                        f"Found **{len(analogue_idx)}** analogues for "
                        f"{query_date} (traj length = {traj_len_an})."
                    )

                except Exception as e:
                    st.error(f"Analogue search error: {e}")
                    st.exception(e)

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.get("analogue_results") is None:
        return

    ar = st.session_state.analogue_results
    query_idx = ar["query_idx"]
    query_date = ar["query_date"]
    traj_len_an = ar["traj_length"]
    analogue_idx = ar["analogue_idx"]
    dists = ar["distances"]

    times_pd = pd.to_datetime(times)
    analogue_start_times = times_pd[analogue_idx]

    # Build result table
    df_an = pd.DataFrame({
        "Rank": range(1, len(analogue_idx) + 1),
        "Start date": [t.strftime("%Y-%m-%d") for t in analogue_start_times],
        "Trajectory distance": np.round(dists, 2),
    })

    st.divider()

    # ── Query score percentile & rarity ──────────────────────────────────────
    scores = st.session_state.scores
    if scores is not None and query_idx < len(scores):
        query_score = float(scores[query_idx])
        F_hat = float(np.mean(scores <= query_score))
        label, color = _rarity_label(F_hat, tau_rare, tau_extreme)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Query rarity score", f"{query_score:.4f}")
        mc2.metric("Empirical percentile", f"{F_hat * 100:.1f}%",
                   help="Fraction of all time steps with a score ≤ the query score.")
        mc3.markdown(
            f"**Classification**\n\n:{color}[**{label.upper()}**]",
            help=f"rare ≥ {tau_rare:.0%}, extreme ≥ {tau_extreme:.0%}",
        )
        mc4.metric("Analogues found", len(analogue_idx))
    else:
        st.metric("Analogues found", len(analogue_idx))

    st.subheader("Ranked analogues")

    tbl_col, map_col = st.columns([1, 2])

    with tbl_col:
        selected_rows = st.dataframe(
            df_an,
            width="stretch",
            height=420,
            on_select="rerun",
            selection_mode="single-row",
        )
        st.download_button(
            "Download analogues CSV",
            df_an.to_csv(index=False),
            file_name="analogues.csv",
            mime="text/csv",
        )

    with map_col:
        try:
            sel_rows = selected_rows.selection.rows
        except AttributeError:
            sel_rows = []
        default_rank = int(sel_rows[0]) if sel_rows else 0

        rank_labels = [
            f"Rank {r}  —  {d}  |  dist={v:.2f}"
            for r, d, v in zip(df_an["Rank"], df_an["Start date"],
                               df_an["Trajectory distance"])
        ]
        rank_sel_idx = st.selectbox(
            "Select analogue to display",
            options=range(len(rank_labels)),
            index=default_rank,
            format_func=lambda i: rank_labels[i],
        )

        sel_an_start = analogue_idx[rank_sel_idx]
        sel_an_time = analogue_start_times[rank_sel_idx]

        if lat is not None and lon is not None:
            # Query: mean anomaly composite over trajectory window
            q_sl = slice(query_idx, query_idx + traj_len_an)
            query_field = (
                raw_stage1[noleap_orig_idx[q_sl]]# - seasonal_cycle[md_index[q_sl]]
            ).mean(axis=0)

            a_sl = slice(sel_an_start, sel_an_start + traj_len_an)
            analogue_field = (
                raw_stage1[noleap_orig_idx[a_sl]]# - seasonal_cycle[md_index[a_sl]]
            ).mean(axis=0)

            vmax = float(np.nanpercentile(
                np.abs(np.concatenate([query_field.ravel(), analogue_field.ravel()])),
                98,
            ))
            vmax = max(vmax, 1e-6)

            map1, map2 = st.columns(2)
            with map1:
                fig_q = make_cartopy_fig(
                    query_field, lat, lon,
                    title=f"Query anomaly: {query_date} (mean over {traj_len_an}d)",
                    cbar_label=f"{p['parameter']} raw value",
                    vmax=vmax, figsize=(5, 3.5),
                )
                st.pyplot(fig_q, width="stretch")
                plt.close(fig_q)
            with map2:
                fig_a = make_cartopy_fig(
                    analogue_field, lat, lon,
                    title=(
                        f"Analogue #{rank_sel_idx + 1}: "
                        f"{sel_an_time.strftime('%Y-%m-%d')} "
                        f"(mean over {traj_len_an}d)"
                    ),
                    cbar_label=f"{p['parameter']} raw value",
                    vmax=vmax, figsize=(5, 3.5),
                )
                st.pyplot(fig_a, width="stretch")
                plt.close(fig_a)
        else:
            st.info("Field maps require data loaded from a file (not NPZ).")

    # ── Temporal distributions ────────────────────────────────────────────────
    st.divider()
    st.subheader("Temporal Distributions")

    MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        month_counts = (
            pd.Series(analogue_start_times.month)
            .value_counts()
            .reindex(range(1, 13), fill_value=0)
            .sort_index()
        )
        fig_month = px.bar(
            x=month_counts.index, y=month_counts.values,
            labels={"x": "Month", "y": "Count"},
            title="Monthly distribution of analogues",
            color_discrete_sequence=["teal"],
        )
        fig_month.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
        fig_month.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_month, width="stretch")

    with tcol2:
        year_counts = (
            pd.Series(analogue_start_times.year)
            .value_counts()
            .sort_index()
        )
        fig_year = px.bar(
            x=year_counts.index, y=year_counts.values,
            labels={"x": "Year", "y": "Count"},
            title="Yearly distribution of analogues",
            color_discrete_sequence=["coral"],
        )
        fig_year.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_year, width="stretch")

    # ── Time series overview ─────────────────────────────────────────────────
    if scores is not None:
        score_times = times_pd[: len(scores)]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=score_times, y=scores,
            mode="lines", name="Rarity score",
            line=dict(color="#aec7e8", width=1),
        ))

        # Mark analogue start dates (guard: analogue indices may exceed scores length)
        an_mask = analogue_idx < len(scores)
        if an_mask.any():
            an_times_valid = analogue_start_times[an_mask]
            an_scores = scores[analogue_idx[an_mask]]
            fig_ts.add_trace(go.Scatter(
                x=an_times_valid, y=an_scores,
                mode="markers", name="Analogues",
                marker=dict(color="#2ca02c", size=7, symbol="circle"),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "Rarity score: %{y:.4f}<extra></extra>"
                ),
            ))

        # Mark query period
        q_start_time = times_pd[query_idx]
        q_end_time = times_pd[min(query_idx + traj_len_an - 1, len(times_pd) - 1)]
        fig_ts.add_vrect(
            x0=q_start_time, x1=q_end_time,
            fillcolor="#d62728", opacity=0.15,
            line_width=0, annotation_text="Query",
            annotation_position="top left",
        )

        fig_ts.update_layout(
            title="Rarity score time series — query window and analogue locations",
            xaxis_title="Date", yaxis_title="Score",
            height=320, margin=dict(t=40, b=20, l=40, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_ts, width="stretch")
