import torch
import xarray as xr
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import time
import os


torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')

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
def knn_scores(nc_path, var, traj_length, k=10, q_batch=128, r_chunk=4096, device=None, dtype=torch.float32):
    # Load dataset
    start_time = time.time()
    ds = xr.open_dataset(nc_path)
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    # (T, H, W) !!! load all data into memory !!!
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    end_time = time.time()
    print(f"Data loading took {end_time - start_time:.2f} seconds.")
    # time = ds["time"].values
    ds.close()

    T, H, W = data.shape
    D = H * W  # vectorize spatial dimensions

    # Choose device for compute if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Move full dataset to device once so all computation stays on device
    start_time = time.time()
    X_cpu = torch.from_numpy(data).to(dtype).reshape(
        T, D)  # (T, D) on device
    if dev.type == "cuda":
        X_cpu = X_cpu.pin_memory()
    end_time = time.time()
    print(f"Data transfer to {dev} took {end_time - start_time:.2f} seconds.")

    # Precompute norms on device
    start_time = time.time()
    norms = blocked_norm_compute(X_cpu, r_chunk, dev)
    end_time = time.time()
    print(f"Norms computation on {dev} took {end_time - start_time:.2f} seconds.")

    N = T - traj_length + 1
    scores = torch.empty(N, dtype=dtype, device="cpu")



    distances_space = torch.empty((T, T), dtype=dtype, device="cpu")
    # blocked outer loop on rows, loop over time
    start_time = time.time()
    for row_start in range(0, T, q_batch):
        row_end = min(row_start + q_batch, T)

        rows = X_cpu[row_start:row_end].to(dev, non_blocking=True)
        row_norms = norms[row_start:row_end].to(dev, non_blocking=True)
        # blocked inner loop on columns, loop over time
        for column_start in range(0, T, r_chunk):
            column_end = min(column_start + r_chunk, T)


            # compute distance of the block, distance on space fields
            
            cols = X_cpu[column_start:column_end].to(dev, non_blocking=True)
            col_norms = norms[column_start:column_end].to(dev, non_blocking=True)
            # distances_ij -> distance between space fields of time i and j
            distances = row_norms[:, None] + col_norms[None, :] - 2.0 * (rows @ cols.T)
            distances_space[row_start:row_end, column_start:column_end] = distances.to("cpu")

    end_time = time.time()
    print(f"Space distances computation took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    # compute trajectory distances 
    L = traj_length
    N = T - L + 1



    # D_0(j) = sum_{t=0..L-1} d_space(t, j+t)
    distances_traj0 = torch.zeros(N, dtype=dtype, device="cpu")
    for t_offset in range(L):
        distances_traj0 += distances_space[t_offset, t_offset:t_offset + N]

    distances_traj = distances_traj0.clone()

    sorted_distances, _ = torch.topk(distances_traj, k=k+1, largest=False)
    scores[0] = sorted_distances[1:k+1].mean()

    # Recurrence for i = 1..N-1:
    # D_i(j) = D_{i-1}(j-1) - d_space(i-1, j-1) + d_space(i+L-1, j+L-1), for j>=1
    for i in range(1, N):
        distances_traj[1:] = (
            distances_traj[:-1]
            - distances_space[i - 1, 0:N-1]              # cols 0..N-2  (length N-1)
            + distances_space[i - 1 + L, L:T]            # cols L..T-1  (length N-1)
        )
        distances_traj[0] = distances_traj0[i]  # uses symmetry: D_i(0) = D_0(i)

        sorted_distances, _ = torch.topk(distances_traj, k=k+1, largest=False)
        scores[i] = sorted_distances[1:k+1].mean()
    end_time = time.time()
    print(f"Trajectory distances and k-NN scoring took {end_time - start_time:.2f} seconds.")

    scores = scores.clamp(min=0.0)
    scores = torch.sqrt(scores)
    return scores, _


if __name__ == "__main__":

    # for traj_length in range(1, 20, 1):
    traj_length = 30
    k = 30
    parameter = "msl"
    start_time = time.time()
    scores, times = knn_scores(
        "Data/era5_msl_daily_eu.nc", parameter, traj_length, k, q_batch=512, r_chunk=2048, device="cuda")
    end_time = time.time()
