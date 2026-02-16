"""
Usage:
python score.py \
    --traj_length 5 \
    --k 10 \
    --parameter msl \
    --file_path Data/era5_msl_daily_eu.nc \
    --device cuda 
    --remove_leap False\
    --remove_seasonal_cycle False \
    --cos_lat_weighting False \
    --pixelwise_standardization False \
    --algorithm exclusion
    -- lon_min -15 \
    -- lon_max 25 \
    -- lat_min 35 \
    -- lat_max 70 \
    --start_year 1995 \
    --end_year 2015
"""


import os
import argparse
import torch
import numpy as np
import pandas as pd
import xarray as xr
import src.rarity_scoring_base
import src.rarity_scoring_exclusion
import src.rarity_scoring_interval
import src.preprocessing as pp

# =========================
# Argument Parser
# =========================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute rarity scores from ERA5 dataset."
    )

    parser.add_argument("--traj_length", type=int, default=5,
                        help="Trajectory length (default: 5)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors (default: 10)")
    parser.add_argument("--parameter", type=str, default="msl",
                        help="Variable name in dataset (default: msl)")
    parser.add_argument("--file_path", type=str,
                        default="Data/era5_msl_daily_eu.nc",
                        help="Path to NetCDF file")
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu or cuda). Default: auto-detect")
    parser.add_argument("--remove_leap", action="store_true",
                        help="Whether to remove leap days (default: False)")
    parser.add_argument("--remove_seasonal_cycle", action="store_true",
                        help="Whether to remove seasonal cycle (default: False)")
    parser.add_argument("--cos_lat_weighting", action="store_true",
                        help="Whether to apply cosine latitude weighting (default: False)")
    parser.add_argument("--pixelwise_standardization", action="store_true",
                        help="Whether to apply pixel-wise standardization (default: False)")
    parser.add_argument("--algorithm", type=str, default="base", choices=["base", "exclusion", "interval"],
                        help="Algorithm to use for scoring (default: base)")

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    traj_length = args.traj_length
    k = args.k
    parameter = args.parameter
    file_path = args.file_path
    lon_min, lon_max = args.lon_min, args.lon_max
    lat_min, lat_max = args.lat_min, args.lat_max
    start_year, end_year = args.start_year, args.end_year

    if args.algorithm == "base":
        compute_distances_and_scores = src.rarity_scoring_base.compute_distances_and_scores
    elif args.algorithm == "exclusion":
        compute_distances_and_scores = src.rarity_scoring_exclusion.compute_distances_and_scores
    elif args.algorithm == "interval":
        compute_distances_and_scores = src.rarity_scoring_interval.compute_distances_and_scores

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Preprocessing
    # =========================
    ds = xr.open_dataset(file_path)
    if start_year and end_year:
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    if lon_min and lon_max :
        ds = ds.sel(lon=slice(lon_min, lon_max))
    if lat_min and lat_max:
        ds = ds.sel(lat=slice(lat_max, lat_min))

    da = ds[parameter]
    spatial_dims = [d for d in da.dims if d != "time"]
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    times = da["time"].values
    ds.close()

    if args.remove_leap:
        data, time = pp.remove_bisex_dailydata(data, da["time"].values)
    else:
        time = da["time"].values
    if args.remove_seasonal_cycle:
        data = pp.remove_seasonal_cycle365(data, time.values)
    if args.cos_lat_weighting:
        data = pp.cos_lat_weighting(data, da["lat"].values, len(da["lon"]))
    if args.pixelwise_standardization:
        data = pp.pixelwise_standardize(data)

    print(f"Data shape after preprocessing: {data.shape}")
    # =========================
    # Compute scores
    # =========================
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

    # =========================
    # Save results and print summary
    # =========================
    out_path = (
        f"case_studies/results/{parameter}/"
        f"{parameter}_trajlen{traj_length}_k{k}"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez(out_path, scores=scores.cpu().numpy(), times=np.array(times))

    # get indices of top 100 scores
    scores_np = scores.cpu().numpy()
    top_100_dates = np.argsort(-scores_np)[:100]
    top_100_times = times[top_100_dates]

    df = pd.DataFrame({
        "time": pd.to_datetime(top_100_times),
        "score": scores_np[top_100_dates]
    })
    df.to_csv(f"{out_path}_top100.csv", index=False)


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
    top_100_relative.to_csv(f"{out_path}_top100_relative.csv", index=False)
if __name__ == "__main__":
    main()
