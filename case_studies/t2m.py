import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import torch
from src.rarity_scoring_interval import knn_scores
from src.distance_matrix import build_matrix

# =========================
# Parameters
# =========================
traj_length = 5
k = 10
parameter = "t2m"
file_path = "Data/t2m_daily_avg_1950_2023.nc"

# =========================
# Main computation
#  =========================
print(
    f"Computing knn scores for {parameter} with trajectory length {traj_length} and k={k}...")
start_time = time.time()
scores = knn_scores(
    file_path, parameter, traj_length, k, q_batch=1024*3, r_chunk=1024*3, device="cpu", exclusion_zone=traj_length)

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds.")




print(scores)
#  =========================
# Save results and print summary
#  =========================

ds = xr.open_dataset(file_path)
times = ds["time"].values
ds.close()




out_path = f"case_studies/results/{parameter}/{parameter}_trajlen{traj_length}_k{k}"
np.savez(out_path, scores=scores, times=np.array(times))




# get indices of top 100 scores
top_100_dates = np.argsort(- scores.numpy())[:100]
top_100_times = times[top_100_dates]

df = pd.DataFrame(
    {"time": top_100_times, "score": scores.numpy()[top_100_dates]})
df["time"] = pd.to_datetime(df["time"])
df.to_csv(f"{out_path}_top100_interval.csv", index=False)

# build_matrix(file_path, parameter, traj_length, k, q_batch=1024*3, r_chunk=1024*3, device="cuda",
#              exclusion_zone=traj_length, h5_path=f"{out_path}_distances.h5", dtype=torch.float32)
# print(f"Distance matrix saved to {out_path}_distances.h5")

times = pd.to_datetime(times)

# Build DataFrame
df = pd.DataFrame({
    "time": times[:len(scores)], # align times with scores
    "score": scores
})

# Extract calendar month (1–12)
df["month"] = df["time"].dt.month

# Compute monthly climatology (mean & std across all years)
monthly_stats = df.groupby("month")["score"].agg(["mean", "std"])

# Merge stats back to original dataframe
df = df.merge(monthly_stats, on="month", how="left")

# Compute relative anomaly (monthly z-score)
df["relative_score"] = np.abs((df["score"] - df["mean"]) / df["std"])

relative_scores = df["relative_score"].values

print("Relative anomaly scores:")
print(relative_scores)

# save top 100 relative anomaly scores
top_100_relative = df.sort_values("relative_score", ascending=False).head(100)
top_100_relative.to_csv(f"{out_path}_top100_relative_interval.csv", index=False)