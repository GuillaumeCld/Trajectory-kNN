import os, time
import numpy as np
import pandas as pd
import xarray as xr
from src.rarity_scoring_base import knn_scores


# =========================
# Parameters
# =========================
traj_length = 15
k = 10
parameter = "tp"
file_path = "Data/tp_daily_sum_1950_2023.nc"

# =========================
# Main computation  
# =========================
print(f"Computing knn scores for {parameter} with trajectory length {traj_length} and k={k}...")
start_time = time.time()
scores = knn_scores(
    file_path, parameter, traj_length, k, q_batch=1024*3, r_chunk=1024*3 , device="cuda", exclusion_zone=traj_length)

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds.")

print(scores)
# =========================
# Save results and print summary
# =========================

ds = xr.open_dataset(file_path)
times = ds["time"].values
ds.close()

out_path = f"case_studies/results/{parameter}/{parameter}_trajlen{traj_length}_k{k}"
np.savez(out_path, scores=scores, times=np.array(times))



top_100_dates = np.argsort(- scores.numpy())[:100] # get indices of top 100 scores
top_100_times = times[top_100_dates]

df = pd.DataFrame({"time": top_100_times, "score": scores.numpy()[top_100_dates]})
df["time"] = pd.to_datetime(df["time"])
df.to_csv(f"{out_path}_top100.csv", index=False)