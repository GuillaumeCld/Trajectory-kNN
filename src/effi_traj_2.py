import torch
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import os
# from memory_profiler import profile

import matplotlib.pyplot as plt
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
def knn_scores(nc_path, var, traj_length, k=10, q_batch=128, r_chunk=4096, device=None, dtype=torch.float32, exclusion_zone=-1):
    if exclusion_zone == -1:
        exclusion_zone = traj_length

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
    times = ds["time"].values
    ds.close()

    return compute_distances_and_scores(data, traj_length, k, q_batch, r_chunk, device, dtype, exclusion_zone), times

# @profile
def compute_distances_and_scores(data, traj_length, k, q_batch, r_chunk, device, dtype, exclusion_zone):

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
        # remove self-match at j=i +- L
        self_matchs = range(max(0, i - exclusion_zone + 1), min(N, i + exclusion_zone))
        distances_traj[self_matchs] = float("inf")
        sorted_distances, sorted_indices = torch.topk(distances_traj, k=k*exclusion_zone, largest=False)

        current_mins = [sorted_indices[0].item()]
        for idx in sorted_indices[1:]:
            idx_item = idx.item()
            if all(abs(idx_item - cm) >= exclusion_zone for cm in current_mins):
                current_mins.append(idx_item)
            if len(current_mins) >= k:
                break


        scores[i] = distances_traj[current_mins].mean()
    end_time = time.time()
    # print(f"Trajectory distances and k-NN scoring took {end_time - start_time:.2f} seconds.")

    # scores = scores.clamp(min=0.0)
    # scores = torch.sqrt(scores)
    return scores.to("cpu")


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


def time_method(
    T,
    H,
    W,
    traj_length,
    q_batch,
    r_chunk,
    k,
    runs=3,
):
    times = []

    for _ in range(runs):
        data = np.random.rand(T, H, W).astype(np.float32)

        start = time.time()
        _ = compute_distances_and_scores(
            data,
            traj_length,
            k,
            q_batch,
            r_chunk,
            device="cuda",
            dtype=torch.float32,
            exclusion_zone=traj_length,
        )
        times.append(time.time() - start)

        del data
        torch.cuda.empty_cache()

    return np.mean(times), np.std(times)


def experiment_data_size():
    H, W = 180, 280
    traj_length = 4
    q_batch = 1024
    r_chunk = 1024
    k = 10

    T_values = [365*10, 365*25, 365*50, 365*75]
    means, stds = [], []

    for T in T_values:
        mean_t, std_t = time_method(
            T, H, W, traj_length, q_batch, r_chunk, k
        )
        means.append(mean_t)
        stds.append(std_t)

    return T_values, means, stds

def plot_results(x, y, yerr, xlabel):
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, y, yerr=yerr,
        marker="o",
        linewidth=2,
        capsize=4
    )
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Computation Time (seconds)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def experiment_traj_length():
    T = 365 * 75
    H, W = 180, 280
    q_batch = 1024
    r_chunk = 1024
    k = 10

    traj_lengths = [1, 2, 4, 8, 16]
    means, stds = [], []

    for L in traj_lengths:
        mean_t, std_t = time_method(
            T, H, W, L, q_batch, r_chunk, k
        )
        means.append(mean_t)
        stds.append(std_t)

    return traj_lengths, means, stds

def computation_time():
    # create synthetic data
    # -----------------------------
    # create synthetic data
    # -----------------------------
    T = 365 * 75    
    H = 180
    W = 280

    traj_length = 1
    q_batch = 1024
    r_chunk = 1024
    k = 10

    data = np.random.rand(T, H, W).astype(np.float32)

    # -----------------------------
    # Your method timing
    # -----------------------------
    start_time = time.time()

    _ = compute_distances_and_scores(
        data,
        traj_length,
        k,
        q_batch,
        r_chunk,
        device="cuda",
        dtype=torch.float32,
        exclusion_zone=traj_length
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Your method took {elapsed:.2f} seconds.")

    import faiss
    mat_trajectories = []
    print("Building matrix")
    for i in range(T - traj_length + 1):
        traj = data[i:i+traj_length].reshape(-1)
        mat_trajectories.append(traj)
    mat_trajectories = np.array(mat_trajectories).astype(np.float32)
    del data
    print("Building FAISS index")

    index = faiss.IndexFlatL2(H * W * traj_length)
    index.add(mat_trajectories)
    print(f"FAISS index has {index.ntotal} vectors.")

    start_time = time.time()
    D, I = index.search(mat_trajectories, k=k*traj_length)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"FAISS k-NN search took {elapsed:.2f} seconds.")



if __name__ == "__main__":  
    # computation_time()
    # Trajectory length experiment
    x, y, yerr = experiment_traj_length()
    plot_results(x, y, yerr, xlabel="Trajectory length")

    # Data size experiment
    x, y, yerr = experiment_data_size()
    plot_results(x, y, yerr, xlabel="Number of time steps (T)")


#     etimes = []
# # for traj_length in range(1, 30, 1):
#     traj_length = 5
#     ks = [i if i != 0 else 1 for i in range(0, 101, 5)]

#     # for k in ks:
#     k = 10
#     parameter = "msl"
#     start_time = time.time()
#     scores, times = knn_scores(
#         "Data/era5_msl_daily_eu.nc", parameter, traj_length, k, q_batch=1024*3, r_chunk=1024*3 , device="cuda")
#     end_time = time.time()
#     elapsed = end_time - start_time
#     etimes.append(elapsed)

#     out_dir = "result/traj/"
#     out_path = os.path.join(
#         out_dir, f"{parameter}_trajlen{traj_length}_k{k}_exclusion.npz")
#     np.savez(out_path, scores=scores, times=np.array(times))

#     print(f"traj_length={traj_length}, k={k}, time={elapsed:.2f} seconds.")
#     print(f"Minimum score: {scores.min().item():.4f}")

#     np.savetxt("effi_traj_2_times_gpu.txt", np.array(etimes))

#     top_100_dates = np.argsort(scores.numpy())[:100]
#     top_100_times = times[top_100_dates]
#     df = pd.DataFrame({"time": top_100_times, "score": scores.numpy()[top_100_dates]})
#     df.to_csv(f"effi_traj_2_top100_trajlen{traj_length}_k{k}_exclusion.csv", index=False)   
#     print("Finished all computations.")


    


