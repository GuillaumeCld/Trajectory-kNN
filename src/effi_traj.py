import torch
import xarray as xr
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import time
import os
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def blocked_norm_compute(X, block_size, dev):
    T, D = X.shape
    norms = torch.empty(T, dtype=torch.float32, device=dev)
    for block_start in range(0, T, block_size):
        block_end = min(block_start + block_size, T)
        block = X[block_start:block_end]
        norms[block_start:block_end] = (block * block).sum(dim=1)
    return norms


@torch.no_grad()
def knn_scores(nc_path, var, traj_length, k=10, q_batch=128, r_chunk=4096, device=None):
    # Load dataset
    ds = xr.open_dataset(nc_path)
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    # (T, H, W) !!! load all data into memory !!!
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    time = ds["time"].values
    ds.close()

    T, H, W = data.shape
    D = H * W  # vectorize spatial dimensions

    # Choose device for compute if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Move full dataset to device once so all computation stays on device
    X = torch.from_numpy(data).to(torch.float32).reshape(
        T, D).to(dev)  # (T, D) on device


    # Precompute norms on device
    norms = blocked_norm_compute(X, r_chunk, dev)

    N = T - traj_length + 1
    scores = torch.empty(N, dtype=torch.float32, device=dev)

    # 

    distances_space = torch.empty((T, T), dtype=torch.float32, device="cpu")
    # blocked outer loop on rows, loop over time
    for row_start in range(0, T, q_batch):
        row_end = min(row_start + q_batch, N)

        # blocked inner loop on columns, loop over time
        for column_start in range(0, T, r_chunk):
            column_end = min(column_start + r_chunk, N)


            # compute distance of the block, distance on space fields
            rows = X[row_start:row_end] 
            columns = X[column_start:column_end]

            # distances_ij -> distance between space fields of time i and j
            distances = norms[row_start:row_end].unsqueeze(1) + \
                norms[column_start:column_end].unsqueeze(0) - 2.0 * rows @ columns.T
            
            distances_space[row_start:row_end, column_start:column_end] = distances.to("cpu")

    # Now compute trajectory distances on device
    distances_traj = torch.empty((N, N), dtype=torch.float32, device="cpu")
    # initialize the first row
    for t_offset in range(traj_length):
        distances_traj[0, :] += distances_space[t_offset, t_offset:t_offset + N]

    # fiil the distances with the recurrence
    for i in range(1, N):
        distances_traj[i, 1:] = distances_traj[i - 1, :-1] - distances_space[i - 1, :N-1] + distances_space[i - 1 + traj_length, :N-1]
        distances_traj[i, 0] = distances_traj[0,i]

    # Take the mean distance to k nearest neighbors excluding self
    sorted_distances, _ = torch.topk(distances_traj, k=k+1, largest=False)
    knn_distances = sorted_distances[:, 1:k+1]
    scores = knn_distances.mean(dim=1)

    return scores.to("cpu"), time[:N]


if __name__ == "__main__":
    traj_length = 1

    # for traj_length in range(1, 20, 1):
    traj_length = 30
    k = 30
    parameter = "msl"
    start_time = time.time()
    scores, times = knn_scores(
        "Data/era5_msl_daily_eu.nc", parameter, traj_length, k, q_batch=256, r_chunk=4096, device="cuda")
    end_time = time.time()
