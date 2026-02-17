import os
import time
import numpy as np, torch
import pandas as pd
import xarray as xr
from src.rarity_scoring_base import compute_distances_and_scores

# =========================
# Parameters
# =========================
k = 10
parameter = "msl"
file_path = "Data/era5_msl_daily_eu.nc"
ds = xr.open_dataset(file_path)
ds = ds.sel(time=slice("1995-01-01", "2015-12-31"))
ds = ds.sel(lon=slice(-22, 45))
ds = ds.sel(lat=slice(72, 27))

# ds = ds.sel(time=slice("1979-01-01", "2013-12-31"))
# ds = ds.sel(lon=slice(-15, 25))
# ds = ds.sel(lat=slice(70, 35))


times = ds["time"].values.astype("datetime64[D]")
df_storm = pd.read_csv("Data/Extremes/CLIMK–WINDS.csv")
dates = pd.to_datetime(df_storm["Dates"], format='%Y%m%d', dayfirst=True).values.astype("datetime64[D]")
print(dates)
# dates = pd.to_datetime(df_storm["Date"], dayfirst=True).values.astype("datetime64[D]")
# df_storm2 = pd.read_csv("Data/Extremes/XWS_insurance_storms.csv")
# dates2 = pd.to_datetime(df_storm2["Date"], dayfirst=True).values.astype("datetime64[D]")
# dates = np.concatenate([dates, dates2])

da = ds[parameter]
spatial_dims = [d for d in da.dims if d != "time"]
data = da.transpose("time", *spatial_dims).values.astype(np.float32)

df_storm = pd.DataFrame({
    "Date": pd.to_datetime(dates).strftime('%Y-%m-%d')
})
# =========================
# Main computation
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
storms_idx = np.where(np.isin(times, dates))

for traj_length in [1, 3, 5, 7]:
    scores = compute_distances_and_scores(
        data,
        traj_length,
        k,
        q_batch=1024,
        r_chunk=1024,
        device=device,
        dtype=torch.float32,
        exclusion_zone=traj_length
    )

    score_storms = torch.tensor([
        scores[idx:idx+traj_length].max()
        for idx in storms_idx[0]
    ])
    empirical_percentile = (
        (scores[None, :] < score_storms[:, None])
        .float()               # <-- convert Bool → Float
        .mean(dim=1)           # use dim instead of axis in PyTorch
        * 100
    )

    df_storm[traj_length] = empirical_percentile
    full_rank = (
        (scores[None, :] > score_storms[:, None])
            .sum(dim=1)
            + 1
    )
    df_storm[f"{traj_length}_rank"] = full_rank

df_storm.to_csv(f"case_studies/results/storm_ranking_{parameter}.csv", index=False)


import matplotlib.pyplot as plt
# plot boxplots of the scores for each trajectory length
plt.figure(figsize=(10, 6))
for traj_length in [1, 3, 5, 7]:
    plt.boxplot(df_storm[traj_length], positions=[traj_length], widths=0.6)
plt.xticks([1, 3, 5, 7], ['1-day', '3-day', '5-day', '7-day'])
plt.xlabel('Trajectory Length')
plt.ylabel('Empirical Percentile Score')
plt.title('Empirical Percentile Scores of CLIMK-WINDS Storms')
plt.grid(True)
plt.savefig(f"case_studies/results/storm_ranking_{parameter}.png")
plt.show()

plt.figure(figsize=(10, 6))
for traj_length in [1, 3, 5, 7]:
    plt.boxplot(df_storm[f"{traj_length}_rank"], positions=[traj_length], widths=0.6)
plt.xticks([1, 3, 5, 7], ['1-day', '3-day', '5-day', '7-day'])
plt.xlabel('Trajectory Length')
plt.ylabel('Rank')
plt.title('Ranks of CLIMK-WINDS Storms')
plt.grid(True)
plt.savefig(f"case_studies/results/storm_ranking_{parameter}_ranks.png")
plt.yscale('log')  
plt.show()

print(df_storm.median(numeric_only=True))
print(df_storm.mean(numeric_only=True))