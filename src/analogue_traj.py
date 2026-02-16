import torch
import xarray as xr
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import time
import os
from rarity_scoring_exclusion import knn_scores 

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
def similarity_compute(nc_path_hist, nc_path_query, var, r_chunk=4096, device=None, dtype=torch.float32):
    """
    Compute distances between trajectories from historical and query datasets i.e. analogue search.

    :param nc_path_hist: Description
    :param nc_path_query: Description
    :param var: Description
    :param r_chunk: Description
    :param device: Description
    :param dtype: Description
    """

    # Load historical dataset
    ds_hist = xr.open_dataset(nc_path_hist)
    da_hist = ds_hist[var]

    spatial_dims = [d for d in da_hist.dims if d != "time"]
    data_hist = da_hist.transpose(
        "time", *spatial_dims).values.astype(np.float32) 
    hist_time = ds_hist["time"].values
    c = data_hist.mean()
    data_hist -= c  # remove mean
    ds_hist.close()
    T_hist, H, W = data_hist.shape
    print(f"Historical data shape: {data_hist.shape}")
    D = H * W  # vectorize spatial dimensions

    # Load query dataset
    ds_query = xr.open_dataset(nc_path_query)
    da_query = ds_query[var]

    data_query = da_query.transpose(
        "time", *spatial_dims).values.astype(np.float32) 
    print(f"Query data shape: {data_query.shape}")
    
    query_time = ds_query["time"].values
    data_query -= c  # remove mean
    ds_query.close()

    T_query = data_query.shape[0]
    assert T_query >= 1, "Query dataset must have at least one time step."
    assert T_hist >= T_query, "Historical dataset must be at least as long as the query dataset."
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
    distances = distances.clamp(min=0.0)
    for i in range(T_query):
        for j in range(T_hist):
            computed_dist = distances[i, j].item()
            true_dist = np.sum((data_query[i].flatten() - data_hist[j].flatten()) ** 2)
            if np.isclose(true_dist, 0.0):
                if not np.isclose(computed_dist, 0.0, atol=1e-5):
                    print(f"Discrepancy at ({i}, {j}): computed {computed_dist}, true {true_dist}")
            elif not np.isclose(np.abs(computed_dist-true_dist)/true_dist, 0.0, atol=5e-6):
                print(f"Discrepancy at ({i}, {j}): computed {computed_dist}, true {true_dist}, diff {np.abs(computed_dist-true_dist)/true_dist:.1e}")

    N = T_hist - T_query + 1
    distances_traj = torch.zeros(N, dtype=dtype, device=dev)
    for t_offset in range(T_query):
        distances_traj += distances[t_offset, t_offset:t_offset + N]

    print("min per-step:", distances.min().item())
    print("min traj:", distances_traj.min().item())

    distances_traj = distances_traj.clamp(min=0.0)
    # distances_traj = torch.sqrt(distances_traj)

    # handle overlap on the same time indices if any
    overlap_idx = np.where((hist_time >= query_time[0]) & (hist_time <= query_time[-1]))[0]

    return distances_traj.cpu(), overlap_idx


if __name__ == "__main__":

    # for traj_length in range(1, 20, 1):
    traj_length = 5
    k = 30
    parameter = "msl"
    start_time = time.time()
    nc_path_hist = "Data/era5_msl_daily_eu.nc"
    nc_path_query = f"Data/era5_msl_daily_eu_traj_19890225_len{traj_length}.nc"
    start_time = time.time()
    sim_traj, overlap_idx = similarity_compute(
        nc_path_hist, nc_path_query, parameter, r_chunk=4096, device="cuda", dtype=torch.float32)
    


    for idx in overlap_idx:
        sim_traj[idx] = np.inf  # invalidate overlapping indices
        break
    traj_score_idx = np.argpartition(sim_traj.numpy(), k+1)[1:k]
    traj_score = sim_traj[traj_score_idx].mean().item()

    end_time = time.time()

    elapsed_time = end_time - start_time
    
    print(
        f"Trajectory length: {traj_length}, Time taken: {elapsed_time:.2f} seconds")

    # print(
    #     f"Min distance: {sim_traj.min().item()}, Max distance: {sim_traj.max().item()}")
    # print(f"Mean distance: {sim_traj.mean().item()}")
    
    


    all_scores, _ = knn_scores(
        nc_path_hist, parameter, traj_length, k, q_batch=4096, r_chunk=4096, device="cuda")
    print(f"Traj score: {traj_score:.4f}")
    print(f"Min knn score: {all_scores.min().item():.4f}, Max knn score: {all_scores.max().item():.4f}")
    print(f"Expedted trajectory score: {all_scores[overlap_idx[0]]}")

    def interpret_score(all_scores, traj_score, tau_common=0.90, tau_extreme=0.99):
        all_scores = np.asarray(all_scores, dtype=float)

        # empirical CDF: fraction of scores <= traj_score
        F_hat = np.mean(all_scores <= traj_score)

        # classification
        if traj_score > all_scores.max():
            label = "never_seen"
        elif F_hat <= tau_common:
            label = "common"
        elif F_hat >= tau_extreme:
            label = "extreme"
        else:
            label = "rare"

        return F_hat, label
    
    F_hat, label = interpret_score(all_scores.numpy(), traj_score, tau_common=0.90)
    print(
        f"Trajectory score: {traj_score:.4f}, Empirical CDF: {F_hat:.4f}, Classification: {label}")