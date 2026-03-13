"""
Tab 3 — Cluster Analysis.

Provides a render(p) function that draws the full tab content.
`p` is the sidebar_params dict assembled by app.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from data_utils import load_data_cached
import torch

import src.preprocessing as pp

def center_inplace_blockwise(X, block_size=1024):
    """
    Center tensor X along dim=0 in-place using block processing.

    Args:
        X (torch.Tensor): tensor of shape (N, D)
        block_size (int): number of rows processed per block
    """
    # compute mean once
    mean = X.mean(dim=0)

    # subtract blockwise in-place
    N = X.shape[0]
    for i in range(0, N, block_size):
        X[i:i+block_size].sub_(mean)

    return X



MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render(p):
    st.header("Cluster Analysis")
    st.markdown(
        "Extract the top-N anomalous events, optionally reduce dimensionality with PCA, "
        "then run K-means clustering across a range of K. Composites and temporal "
        "distributions are shown interactively."
    )

    if st.session_state.scores is None:
        st.info("Run the scoring pipeline (or upload a .npz) first.")
        return

    scores = st.session_state.scores
    times = st.session_state.times
    lat = st.session_state.lat
    lon = st.session_state.lon
    load_params = st.session_state.load_params
    score_times = times[:len(scores)]

    # --- Cluster parameters ---
    with st.expander("Clustering parameters", expanded=True):
        cc1, cc2, cc3, cc4 = st.columns(4)
        n_top_cl = cc1.number_input("Top-N events", 10, 500, 100, step=10)
        k_min_cl = int(cc2.number_input("K min", 2, 15, 2, step=1))
        k_max_cl = int(cc3.number_input("K max", 2, 20, 10, step=1))
        n_init_cl = int(cc4.number_input("K-means n_init", 10, 200, 50, step=10))

        st.markdown("**Preprocessing** *(applied independently to the raw data)*")
        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        cl_remove_leap = pc1.checkbox(
            "Remove leap days", value=p["remove_leap"], key="cl_remove_leap",
            help="Remove Feb 29 entries (required when seasonal cycle is enabled).",
        )
        cl_remove_sc = pc2.checkbox(
            "Seasonal cycle", value=p["remove_sc"], key="cl_remove_sc",
            help="Remove 365-day climatological mean.",
        )
        cl_cos_lat = pc3.checkbox(
            "Cos-lat weighting", value=p["cos_lat"], key="cl_cos_lat",
            help="Apply cosine(latitude) spatial weighting.",
        )
        cl_pxstd = pc4.checkbox(
            "Pixel-wise std", value=p["pxstd"], key="cl_pxstd",
            help="Normalise each grid point to unit variance.",
        )
        cl_detrend = pc5.checkbox(
            "Detrend", value=False, key="cl_detrend",
            help="Remove linear trend at each grid point before clustering.",
        )
        cl_pca = pc6.checkbox(
            "PCA (95% var)", value=p["use_pca"], key="cl_pca",
            help="Reduce to PCA components retaining ≥95% variance before K-means.",
        )

    run_cl_btn = st.button("Run Cluster Analysis", type="primary")

    if run_cl_btn:
        if load_params is None:
            st.error(
                "Raw data not available. Re-run the scoring pipeline from a file "
                "(uploaded NPZ files do not include the raw fields)."
            )
        else:
            try:
                with st.spinner("Applying cluster preprocessing..."):
                    cl_data, _, lat, _ = load_data_cached(
                        **load_params,
                        remove_leap=cl_remove_leap,
                        remove_sc=cl_remove_sc, cos_lat=False,
                        pxstd=False, detrend=cl_detrend,
                    )

                with st.spinner("Extracting top events..."):
                    top_idx_cl = np.argsort(-scores)[:int(n_top_cl)]
                    print(scores[top_idx_cl])
                    fields_2d = cl_data[top_idx_cl]        # (n, H, W)
                    n_ev, nlat_c, nlon_c = fields_2d.shape
                    # base_fields: preprocessed (pre-PCA) fields used for composites
                    base_fields = fields_2d.detach().clone() if isinstance(fields_2d, torch.Tensor) else fields_2d.copy()
                    if cl_cos_lat:
                        fields_2d = pp.cos_lat_weighting(fields_2d, lat, nlon_c)

                    fields_2d = fields_2d.reshape(n_ev, nlat_c * nlon_c)
                    # valid_cols = ~np.isnan(fields_2d).any(axis=0)
                    # fields_2d = fields_2d[:, valid_cols]

           
                    if cl_pxstd:
                        fields_2d = pp.pixelwise_standardize(fields_2d)
                with st.spinner("PCA..." if cl_pca else "Preparing features..."):
                    if cl_pca:
                        fields_2d = torch.from_numpy(fields_2d).float() if not isinstance(fields_2d, torch.Tensor) else fields_2d.float()
                        q = min(100, fields_2d.shape[0] - 1, fields_2d.shape[1])
                        print(f"Performing low-rank SVD with q={q} components...")
                        U, S, _ = torch.svd_lowrank(center_inplace_blockwise(fields_2d, 1024), q=q)
                        del fields_2d
                        cumvar = torch.cumsum(S**2 / (S**2).sum(), dim=0)
                        npc = max(int(torch.argmax((cumvar >= 0.95).float())) + 1, 3)
                        fields_embed = (U[:, :npc] * S[:npc]).cpu().numpy()
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
    if st.session_state.cluster_results is None:
        return

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
    kmin_stored = cr["kmin"]
    kmax_stored = cr["kmax"]

    event_times = score_times[top_idx_cl]
    ks = list(range(kmin_stored, kmax_stored + 1))

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
        st.plotly_chart(fig_sil, width="stretch")

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
        st.plotly_chart(fig_inertia, width="stretch")

    st.divider()

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
        composites.append(
            base_fields[members].mean(axis=0) if len(members) > 0
            else np.zeros((nlat_c, nlon_c))
        )

    if lat is not None and lon is not None:
        n_cols = min(selected_k, 3)
        n_rows = int(np.ceil(selected_k / n_cols))
        proj = ccrs.PlateCarree()

        all_vals = np.concatenate([c.ravel() for c in composites])
        vmax_glob = float(np.nanpercentile(np.abs(all_vals), 95))
        vmax_glob = max(vmax_glob, 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax_glob, vcenter=0, vmax=vmax_glob)

        # Extra row at the bottom for the shared horizontal colorbar
        fig_comp = plt.figure(figsize=(5.5 * n_cols, 3.8 * n_rows + 0.6))
        gs_comp = fig_comp.add_gridspec(
            n_rows + 1, n_cols,
            height_ratios=[3.8] * n_rows + [0.25],
            hspace=0.35, wspace=0.05,
        )

        mesh = None
        for k_i in range(selected_k):
            row, col = divmod(k_i, n_cols)
            ax = fig_comp.add_subplot(gs_comp[row, col], projection=proj)
            mesh = ax.pcolormesh(
                lon, lat, composites[k_i],
                cmap="RdBu_r", norm=norm,
                transform=proj, shading="auto",
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey")
            ax.set_extent(
                [float(lon.min()), float(lon.max()),
                 float(lat.min()), float(lat.max())],
                crs=proj,
            )
            gl = ax.gridlines(
                draw_labels=(col == 0 or row == n_rows - 1),
                linewidth=0.3, color="grey", alpha=0.5, linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            if col > 0:
                gl.left_labels = False
            ax.set_title(f"Cluster {k_i + 1}  (n={counts[k_i]})", fontsize=10)

        for k_i in range(selected_k, n_rows * n_cols):
            row, col = divmod(k_i, n_cols)
            fig_comp.add_subplot(gs_comp[row, col]).set_visible(False)

        # Shared colorbar spanning entire bottom row
        cax_comp = fig_comp.add_subplot(gs_comp[n_rows, :])
        cbar = fig_comp.colorbar(mesh, cax=cax_comp, orientation="horizontal")
        cbar.set_label(p["parameter"], fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        fig_comp.suptitle(f"Composites — K={selected_k}", fontsize=12, y=1.01)
        st.pyplot(fig_comp, width="stretch")
        plt.close(fig_comp)
    else:
        # Fallback without geographic coordinates (NPZ upload path)
        n_cols = min(selected_k, 3)
        n_rows = int(np.ceil(selected_k / n_cols))
        fig_comp, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 3.5 * n_rows),
            squeeze=False,
            sharex=True, sharey=True,
        )
        all_vals = np.concatenate([c.ravel() for c in composites])
        vmax_glob = max(float(np.nanpercentile(np.abs(all_vals), 95)), 1e-6)
        for k_i in range(selected_k):
            row, col = divmod(k_i, n_cols)
            axes[row, col].imshow(
                composites[k_i], cmap="RdBu_r",
                vmin=-vmax_glob, vmax=vmax_glob, aspect="auto",
            )
            axes[row, col].set_title(f"Cluster {k_i + 1}  (n={counts[k_i]})")
        for k_i in range(selected_k, n_rows * n_cols):
            row, col = divmod(k_i, n_cols)
            axes[row, col].set_visible(False)
        fig_comp.tight_layout()
        st.pyplot(fig_comp, width="stretch")
        plt.close(fig_comp)

    # Cluster sizes bar chart
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
    st.plotly_chart(fig_sizes, width="stretch")

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
                st.plotly_chart(fig_month, width="stretch")

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
                st.plotly_chart(fig_year, width="stretch")

            # Per-member silhouette values
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
                st.plotly_chart(fig_sil_m, width="stretch")

            # Date table
            df_cl = pd.DataFrame({
                "Date": [t.strftime("%Y-%m-%d") for t in member_times],
                "Rarity score": np.round(member_scores, 4),
            }).sort_values("Rarity score", ascending=False)
            st.dataframe(df_cl, width="stretch", height=250)

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
