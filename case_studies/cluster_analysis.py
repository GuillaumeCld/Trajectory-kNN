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
from datetime import datetime, timedelta

# =========================
# User parameters
# =========================
file_path = "Data/z500_anom_daily_eu.nc"
date_path = "case_studies/results/z500/z500_trajlen5_k10_top100_relative.csv"
parameter = "z500"
out_path = "case_studies/results/z500/"

# Prepocessing options
remove_leap = True # should be always True
remove_seasonal_cycle = True
cos_lat_weighting = True
pixelwise_standardization = True
pca = True
scaling_factor = 100.0

# =========================
# Load data 
# =========================
ds = xr.open_dataset(file_path)

lon = ds["lon"].values.astype(np.float32) if "lon" in ds else ds["longitude"].values.astype(np.float32)
lat = ds["lat"].values.astype(np.float32) if "lat" in ds else ds["latitude"].values.astype(np.float32)
time = ds["time"].values
time = pd.to_datetime(time).normalize()
spatial_dims = [d for d in ds.dims if d != "time"]
# (T, H, W) !!! load all data into memory !!!
data = ds.transpose("time", *spatial_dims)[parameter].values.astype(np.float32) / scaling_factor
ds.close()


top100_dates = pd.read_csv(date_path)["time"].values.astype("datetime64[ns]")
top100_dates = pd.to_datetime(top100_dates).normalize()


# =========================
# Preprocessing
# =========================
def remove_bisex_dailydata(data, time):

    time = pd.to_datetime(time)
    mask = ~((time.month == 2) & (time.day == 29))
    return data[mask], time[mask]

def remove_seasonal_cycle365(data, time):
    time = pd.to_datetime(time).normalize()

    # Check for leap days
    if np.any((time.month == 2) & (time.day == 29)):
        raise ValueError("Leap days present. Remove them first.")

    # Encode (month, day) as unique integers: 1..365
    # E.g., Jan 1 -> 1, Jan 2 -> 2, ..., Dec 31 -> 365
    month_cumsum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    # This works because Feb 29 is removed, so Feb always has 28 days
    md_index = month_cumsum[time.month.values - 1] + (time.day.values - 1)  # 0-based index

    # Preallocate seasonal cycle array
    seasonal_cycle = np.zeros((365, *data.shape[1:]), dtype=data.dtype)
    counts = np.zeros(365, dtype=int)

    # Accumulate sums
    np.add.at(seasonal_cycle, md_index, data)
    np.add.at(counts, md_index, 1)

    # Divide by counts (only where count > 0)
    seasonal_cycle[counts > 0] /= counts[counts > 0][:, None, None]

    # Deseasonalize
    deseasonalized = data - seasonal_cycle[md_index]

    return deseasonalized



if remove_leap:
    data, time  = remove_bisex_dailydata(data, time)

if remove_seasonal_cycle:
    data = remove_seasonal_cycle365(data, time)

month_vector = pd.to_datetime(time).month
year_vector = pd.to_datetime(time).year

# =========================
# Extract fields
# =========================
abnormal_index = np.isin(time, top100_dates)
idx_an = np.where(abnormal_index)[0]
n_abnormal = len(idx_an)

abnormal_fields = data[idx_an, :, :]
base_fields = abnormal_fields.copy()
nlon = len(lon)
nlat = len(lat)

abnormal_fields = abnormal_fields.reshape(n_abnormal, nlon * nlat)


# =========================
# Area weighting
# =========================
if cos_lat_weighting:
    wlat = np.cos(np.deg2rad(lat))
    W = np.tile(wlat, (nlon, 1)).T.flatten()
    Ws = np.sqrt(W)

    abnormal_fields = abnormal_fields * Ws

# remove spatial dimensions with any NaN values across time
abnormal_fields = abnormal_fields[:, ~np.isnan(abnormal_fields).any(axis=0)]

# =========================
# Standardize
# =========================
if pixelwise_standardization:
    mu = abnormal_fields.mean(axis=0)
    sd = abnormal_fields.std(axis=0)
    sd[sd == 0] = 1.0

    abnormal_fields = (abnormal_fields - mu) / sd

# =========================
# PCA
# =========================
if pca:
    pca = PCA()
    score = pca.fit_transform(abnormal_fields)
    explained = pca.explained_variance_ratio_ * 100
    cumexp = np.cumsum(explained)

    npc = np.argmax(cumexp >= 95) + 1
    npc = max(npc, 3)

    abnormal_fields = score[:, :npc]

# =========================
# K selection
# =========================
Kmin = 2
Kmax = min(10, n_abnormal - 1)

avg_sil = np.zeros(Kmax + 1, dtype=float)
all_idx = {}

for K in range(Kmin, Kmax + 1):
    kmeans = KMeans(n_clusters=K, n_init=50, max_iter=1000, random_state=0, init="k-means++")
    idxK = kmeans.fit_predict(abnormal_fields)

    all_idx[K] = idxK
    sil = silhouette_samples(abnormal_fields, idxK)
    avg_sil[K] = sil.mean()

bestK = np.argmax(avg_sil[Kmin:Kmax + 1]) + Kmin
cl = all_idx[bestK]

letters = list("abcdefghijklmnopqrstuvwxyz")

# =========================
# Compute composites
# =========================
print(f"nlon: {nlon}, nlat: {nlat}, bestK: {bestK}, field shape: {abnormal_fields.shape}")
composites = np.zeros((bestK, nlat, nlon))
counts = np.zeros(bestK, dtype=int)

for k in range(bestK):
    members = np.where(cl == k)[0]
    counts[k] = len(members)
    composites[k, :, :] = base_fields[members].mean(axis=0)

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
plt.savefig(out_path + "Figure_K_selection.png", dpi=300)
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
plt.savefig(out_path + "Figure_sizes.png", dpi=300)
plt.close()

# =========================
# Month / Year distributions
# =========================
plt.figure(figsize=(10, 3 * bestK))

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
plt.savefig(out_path+"Figure_time.png")
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
plt.savefig(out_path+"Figure_silhouette.png")
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
plt.savefig(out_path + "Figure_time.png", dpi=300)
plt.close()

# =========================
# Figure: Cluster composites
# =========================
from matplotlib.colors import TwoSlopeNorm


min_val = np.nanmin(composites)
max_val = np.nanmax(composites)
raw_max = np.ceil(max(abs(min_val), abs(max_val)))
# if raw_max % 2 != 0:
#     raw_max -= 1  # Ensure we have an even max

print(raw_max)

levels = np.linspace(-raw_max, raw_max, 9)#.astype(int)
print(levels)

norm = TwoSlopeNorm(vmin=-raw_max, vcenter=0.0, vmax=raw_max)


fig, axes = plt.subplots(
    1, bestK, 
    figsize=(4 * bestK, 5), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    constrained_layout=True
)

if bestK == 1: axes = [axes]

for k, ax in enumerate(axes):
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.coastlines(linewidth=0.8)
    
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = gl.right_labels = False
    if k > 0: gl.left_labels = False

    cf = ax.contourf(
        lon, lat, composites[k, :, :], 
        levels=levels, 
        cmap="RdBu_r", 
        norm=norm,
        transform=ccrs.PlateCarree(),
        extend='both'
    )
    
    ax.set_title(f"({letters[k]}) Cluster {k+1}\n$n={counts[k]}$")

cbar = fig.colorbar(cf, ax=axes, orientation='horizontal', pad=0.08, fraction=0.05)
cbar.set_ticks(levels) 
cbar.set_label(f"{parameter} anomalies")

plt.savefig(out_path+"Figure_composites.png", bbox_inches='tight', dpi=300)
plt.close()