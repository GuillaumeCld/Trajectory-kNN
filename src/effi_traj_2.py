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
    c = data.mean()
    data -= c  # remove mean
    end_time = time.time()
    # print(f"Data loading took {end_time - start_time:.2f} seconds.")
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
    X = torch.from_numpy(data).to(dtype).reshape(
        T, D).to(dev)  # (T, D) on device
    end_time = time.time()
    # print(f"Data transfer to {dev} took {end_time - start_time:.2f} seconds.")

    # Precompute norms on device
    start_time = time.time()
    norms = blocked_norm_compute(X, r_chunk, dev)
    end_time = time.time()
    # print(f"Norms computation on {dev} took {end_time - start_time:.2f} seconds.")

    N = T - traj_length + 1
    scores = torch.empty(N, dtype=dtype, device="cpu")



    distances_space = torch.empty((T, T), dtype=dtype, device="cpu")
    # blocked outer loop on rows, loop over time
    start_time = time.time()
    for row_start in range(0, T, q_batch):
        row_end = min(row_start + q_batch, T)

        rows = X[row_start:row_end] 
        row_norms = norms[row_start:row_end]
        # blocked inner loop on columns, loop over time
        for column_start in range(0, T, r_chunk):
            column_end = min(column_start + r_chunk, T)


            # compute distance of the block, distance on space fields
            
            cols = X[column_start:column_end]
            col_norms = norms[column_start:column_end]
            # distances_ij -> distance between space fields of time i and j
            distances = row_norms[:, None] + col_norms[None, :] - 2.0 * (rows @ cols.T)
            distances_space[row_start:row_end, column_start:column_end] = distances.to("cpu")
    distances_space = distances_space.clamp(min=0.0)
    end_time = time.time()
    # print(f"Space distances computation took {end_time - start_time:.2f} seconds.")

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
        distances_traj = distances_traj.clamp(min=0.0)
        # remove self-match at j=i +- L/2
        distances_traj[i] = float("inf")

        sorted_distances, _ = torch.topk(distances_traj, k=k+1, largest=False)
        scores[i] = sorted_distances[1:k+1].mean()
    end_time = time.time()
    # print(f"Trajectory distances and k-NN scoring took {end_time - start_time:.2f} seconds.")

    # scores = scores.clamp(min=0.0)
    # scores = torch.sqrt(scores)
    return scores.to("cpu"), _


def similarity_compute(nc_path_hist, nc_path_query, var, r_chunk=4096, device=None, dtype=torch.float32):
    # Load historical dataset
    ds_hist = xr.open_dataset(nc_path_hist)
    da_hist = ds_hist[var]

    spatial_dims = [d for d in da_hist.dims if d != "time"]
    data_hist = da_hist.transpose("time", *spatial_dims).values.astype(np.float32)
    ds_hist.close()

    T_hist, H, W = data_hist.shape
    D = H * W  # vectorize spatial dimensions

    # Load query dataset
    ds_query = xr.open_dataset(nc_path_query)
    da_query = ds_query[var]

    data_query = da_query.transpose("time", *spatial_dims).values.astype(np.float32)
    ds_query.close()

    T_query = data_query.shape[0]
    assert T_query >= 1, "Query dataset must have at least one time step."

    # Choose device for compute if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Move full datasets to device once so all computation stays on device
    X_hist = torch.from_numpy(data_hist).to(dtype).reshape(
        T_hist, D).to(dev)  # (T_hist, D) on device
    X_query = torch.from_numpy(data_query).to(dtype).reshape(
        T_query, D).to(dev)  # (T_query, D) on device

    # Precompute norms on device
    norms_hist = blocked_norm_compute(X_hist, r_chunk, dev)
    norms_query = blocked_norm_compute(X_query, r_chunk, dev)

    distances = torch.empty((T_query, T_hist), dtype=dtype, device=dev)
    rows = X_query
    row_norms = norms_query
    # compute distances between query and historical data    
    for column_start in range(0, T_hist, r_chunk):
        column_end = min(column_start + r_chunk, T_hist)

        cols = X_hist[column_start:column_end]
        col_norms = norms_hist[column_start:column_end]
        # distances_ij -> distance between space fields of time i and j
        dists = row_norms[:, None] + col_norms[None, :] - 2.0 * (rows @ cols.T)
        distances[:, column_start:column_end] = dists

    N = T_hist - traj_length + 1
    distances_traj = torch.zeros(N, dtype=dtype, device=dev)
    for t_offset in range(T_query):
        distances_traj += distances   [t_offset, t_offset:t_offset + N]


if __name__ == "__main__":

    etimes = []
    # for traj_length in range(1, 30, 1):
    traj_length = 5
    k = 30
    parameter = "msl"
    start_time = time.time()
    scores, times = knn_scores(
        "Data/era5_msl_daily_eu.nc", parameter, traj_length, k, q_batch=1024*3, r_chunk=1024*3 , device="cuda")
    end_time = time.time()
    elapsed = end_time - start_time
    etimes.append(elapsed)

    out_dir = "result/traj/"
    out_path = os.path.join(
        out_dir, f"{parameter}_trajlen{traj_length}_k{k}.npz")
    # np.savez(out_path, scores=scores, times=np.array(times))

    print(f"traj_length={traj_length}, k={k}, time={elapsed:.2f} seconds.")
    print(f"Minimum score: {scores.min().item():.4f}")

    # np.savetxt("effi_traj_2_times_cpu.txt", np.array(etimes))
    
