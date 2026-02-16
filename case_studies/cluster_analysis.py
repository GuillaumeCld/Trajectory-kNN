"""
Example usage:

# Run with default settings
python cluster_analysis.py

# Disable PCA and seasonal cycle removal
python cluster_analysis.py --no_pca --no_remove_seasonal_cycle

# Use different dataset and results file
python cluster_analysis.py \
    --file_path Data/other_dataset.nc \
    --date_path case_studies/results/t2m/other_top100.csv

# Disable cosine latitude weighting
python cluster_analysis.py --no_cos_lat_weighting
"""

import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib.colors import TwoSlopeNorm

import src.preprocessing as pp


# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster analysis and composite visualization of extreme fields."
    )

    parser.add_argument("--file_path", type=str,
                        default="Data/t2m_daily_avg_1950_2023.nc")

    parser.add_argument("--date_path", type=str,
                        default="case_studies/results/t2m/t2m_trajlen5_k10_top100.csv")

    parser.add_argument("--parameter", type=str, default="t2m")
    parser.add_argument("--out_path", type=str,
                        default="case_studies/results/t2m/")

    # Preprocessing flags
    parser.add_argument("--no_remove_leap", action="store_true")
    parser.add_argument("--no_remove_seasonal_cycle", action="store_true")
    parser.add_argument("--no_cos_lat_weighting", action="store_true")
    parser.add_argument("--no_pixelwise_standardization", action="store_true")
    parser.add_argument("--no_pca", action="store_true")

    parser.add_argument("--scaling_factor", type=float, default=1.0)
    parser.add_argument("--scaling_constant", type=float, default=0.0)

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    print(args)

    remove_leap = not args.no_remove_leap
    remove_seasonal_cycle = not args.no_remove_seasonal_cycle
    cos_lat_weighting = not args.no_cos_lat_weighting
    pixelwise_standardization = not args.no_pixelwise_standardization
    use_pca = not args.no_pca

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, "all_dates"), exist_ok=True)

    # =========================
    # Load data
    # =========================
    ds = xr.open_dataset(args.file_path)

    lon = ds["lon"].values.astype(np.float32) if "lon" in ds else ds["longitude"].values.astype(np.float32)
    lat = ds["lat"].values.astype(np.float32) if "lat" in ds else ds["latitude"].values.astype(np.float32)
    time = pd.to_datetime(ds["time"].values).normalize()

    spatial_dims = [d for d in ds.dims if d != "time"]
    data = ds.transpose("time", *spatial_dims)[args.parameter].values.astype(np.float32)

    data /= args.scaling_factor
    data -= args.scaling_constant
    ds.close()

    top100_dates = pd.to_datetime(
        pd.read_csv(args.date_path)["time"]
    ).dt.normalize()

    # =========================
    # Preprocessing
    # =========================
    if remove_leap:
        data, time = pp.remove_bisex_dailydata(data, time)

    if remove_seasonal_cycle:
        data = pp.remove_seasonal_cycle365(data, time)

    month_vector = pd.to_datetime(time).month
    year_vector = pd.to_datetime(time).year

    # =========================
    # Extract abnormal fields
    # =========================
    abnormal_index = np.isin(time, top100_dates)
    idx_an = np.where(abnormal_index)[0]
    n_abnormal = len(idx_an)

    abnormal_fields = data[idx_an, :, :]
    base_fields = abnormal_fields.copy()

    nlat = len(lat)
    nlon = len(lon)

    abnormal_fields = abnormal_fields.reshape(n_abnormal, nlat * nlon)
    abnormal_fields = abnormal_fields[:, ~np.isnan(abnormal_fields).any(axis=0)]

    if cos_lat_weighting:
        data = pp.cos_lat_weighting(data, lat, nlon)

    if pixelwise_standardization:
        abnormal_fields = pp.pixelwise_standardize(abnormal_fields)

    # =========================
    # PCA
    # =========================
    if use_pca:
        pca_model = PCA()
        score = pca_model.fit_transform(abnormal_fields)
        explained = pca_model.explained_variance_ratio_ * 100
        cumexp = np.cumsum(explained)

        npc = np.argmax(cumexp >= 95) + 1
        npc = max(npc, 3)

        abnormal_fields = score[:, :npc]


    print(f"Shape of abnormal fields after preprocessing: {abnormal_fields.shape}")
    print(f"Max value in abnormal fields: {np.nanmax(abnormal_fields)}")
    # =========================
    # K selection
    # =========================
    Kmin = 2
    Kmax = min(10, n_abnormal - 1)

    avg_sil = np.zeros(Kmax + 1)
    all_idx = {}

    for K in range(Kmin, Kmax + 1):
        kmeans = KMeans(n_clusters=K, n_init=50, max_iter=1000,
                        random_state=0, init="k-means++")
        idxK = kmeans.fit_predict(abnormal_fields)

        all_idx[K] = idxK
        sil = silhouette_samples(abnormal_fields, idxK)
        avg_sil[K] = sil.mean()

    bestK = np.argmax(avg_sil[Kmin:Kmax + 1]) + Kmin
    cl = all_idx[bestK]

    # =========================
    # Compute composites
    # =========================
    composites = np.zeros((bestK, nlat, nlon))
    counts = np.zeros(bestK)

    for k in range(bestK):
        members = np.where(cl == k)[0]
        counts[k] = len(members)
        composites[k] = base_fields[members].mean(axis=0)

    # =========================
    # Composite Plot
    # =========================
    min_val = np.nanmin(composites)
    max_val = np.nanmax(composites)
    raw_max = np.ceil(max(abs(min_val), abs(max_val)))
    levels = np.linspace(-raw_max, raw_max, 9)
    norm = TwoSlopeNorm(vmin=-raw_max, vcenter=0.0, vmax=raw_max)

    fig, axes = plt.subplots(
        1, bestK,
        figsize=(4 * bestK, 5),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    if bestK == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
        ax.coastlines()
        cf = ax.contourf(
            lon, lat, composites[k],
            levels=levels,
            cmap="RdBu_r",
            norm=norm,
            transform=ccrs.PlateCarree(),
            extend='both'
        )
        ax.set_title(f"Cluster {k+1} (n={int(counts[k])})")

    cbar = fig.colorbar(cf, ax=axes, orientation='horizontal', pad=0.08)
    cbar.set_label(f"{args.parameter} anomalies")

    plt.savefig(os.path.join(args.out_path, "Figure_composites.png"), dpi=300)
    plt.close()

    # =========================
    # Figure 1: K selection
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(range(Kmin, Kmax + 1), avg_sil[Kmin:Kmax + 1], "-o", 
            linewidth=2.5, markersize=8, color='#2c3e50', label='Silhouette Score')

    # Highlight the selected Best K
    plt.axvline(bestK, color='#e74c3c', linestyle='--', alpha=0.8, label=f'Best K = {bestK}')
    plt.scatter(bestK, avg_sil[bestK], color='#e74c3c', s=120, zorder=5)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Mean Silhouette Score", fontsize=12)
    plt.title("Optimal Cluster Selection (Silhouette Method)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path + "K_selection.png", dpi=300)
    plt.close()

    # =========================
    # Figure: Cluster sizes
    # =========================
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, bestK + 1), counts, color='steelblue', edgecolor='black', alpha=0.8)

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')

    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Frequency (n)", fontsize=12)
    plt.title("Cluster Membership Distribution", fontsize=14, fontweight='bold')
    plt.xticks(range(1, bestK + 1))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out_path + "Figure_sizes.png", dpi=300)
    plt.close()

    # =========================
    # Month / Year distributions
    # =========================
    plt.figure(figsize=(10, 3 * bestK))
    letters = "abcdefghijklmnopqrstuvwxyz"
    panel = 1
    for k in range(bestK):
        members = idx_an[cl == k]

        plt.subplot(bestK, 2, panel)
        plt.hist(month_vector[members])
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.title(f"{letters[panel-1]}) Month cluster {k+1}")
        panel += 1

        plt.subplot(bestK, 2, panel)
        plt.hist(year_vector[members])
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.title(f"{letters[panel-1]}) Year cluster {k+1}")
        panel += 1

    plt.tight_layout()
    plt.savefig(args.out_path+"Figure_time.png")
    plt.close()

    # =========================
    # Figure: distribution of silhouette values
    # =========================
    plt.figure(figsize=(5 * bestK, 4))
    for k in range(bestK):
        members = np.where(cl == k)[0]
        sil = silhouette_samples(abnormal_fields, cl, metric='euclidean')
        sil_members = sil[members]
        sorted_idx = np.argsort(sil_members)
        
        plt.subplot(1, bestK, k + 1)
        plt.barh(range(len(members)), sil_members[sorted_idx], alpha=0.6)
        plt.xlabel("Silhouette value")
        plt.ylabel("Cluster element")
        plt.title(f"{letters[k]}) Cluster {k+1} silhouette distribution")
    plt.tight_layout()
    plt.savefig(args.out_path+"Figure_silhouette.png")
    plt.close()


    fig, axes = plt.subplots(bestK, 2, figsize=(10, 3 * bestK), sharex='col')

    for k in range(bestK):
        members = idx_an[cl == k]
        
        # Month Distribution (Column 0)
        ax_m = axes[k, 0] if bestK > 1 else axes[0]
        ax_m.hist(month_vector[members], bins=np.arange(0.5, 13.5, 1), 
                rwidth=0.8, color='teal', alpha=0.7)
        ax_m.set_xticks(range(1, 13))
        ax_m.set_title(f"({letters[k*2]}) Cluster {k+1}: Monthly Dist.", fontsize=11)
        ax_m.set_ylabel("Count")

        # Year Distribution (Column 1)
        ax_y = axes[k, 1] if bestK > 1 else axes[1]
        ax_y.hist(year_vector[members], bins=15, color='coral', alpha=0.7, edgecolor='white')
        ax_y.set_title(f"({letters[k*2+1]}) Cluster {k+1}: Yearly Dist.", fontsize=11)

    # Add X-labels to the bottom-most plots only
    if bestK > 1:
        axes[-1, 0].set_xlabel("Month")
        axes[-1, 1].set_xlabel("Year")
    else:
        axes[0].set_xlabel("Month")
        axes[1].set_xlabel("Year")

    plt.tight_layout()
    plt.savefig(args.out_path + "Figure_time.png", dpi=300)
    plt.close()




if __name__ == "__main__":
    main()
