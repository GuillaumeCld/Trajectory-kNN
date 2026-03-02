import torch
import numpy as np
import xarray as xr
import time
from src.distance_matrix import distance_matrix
import matplotlib.pyplot as plt


def mle_intrinsic_dimension(D, ks=[20]):
    """
    D: pairwise distance matrix (N x N)
    k: number of neighbors (e.g., 20)
    """
    N = D.shape[0]

    # Avoid zero distances
    D = torch.sqrt(D + 1e-12)

    # Sort distances
    sorted_D, _ = torch.sort(D, dim=1)

    # First element is zero (self-distance), skip it

    ids_mean = []
    ids_std = []
    for k in ks:

        T = sorted_D[:, 1:k+1]  # shape: [N, k]

        # put inf on diagonal band to exclude self-distances +- 15
        n = D.size(0)
        idx = torch.arange(n, device=D.device)

        mask = torch.ones_like(D, dtype=torch.bool)

        mask = torch.ones_like(D, dtype=torch.bool)

        for i in range(-15, 16):
            if i != 0:
                mask.diagonal(offset=i).fill_(False)

        D.masked_fill_(~mask, float('inf'))   # in-place to avoid copy

        # get k+1 smallest (since you skip index 0)
        T, _ = torch.topk(D, k=k+1, dim=1, largest=False)

        # T = T[:, 1:]  # drop the first (usually self-distance)
        # D_masked = D.masked_fill(~mask, float('inf'))
        # sorted_D_masked, _ = torch.sort(D_masked, dim=1)
        # T = sorted_D_masked[:, :k]
# 

        Tk = T[:, -1].unsqueeze(1)  # k-th neighbor
        logs = torch.log(Tk / T[:, :-1])  # shape: [N, k-1]

        local_id = (logs.mean(dim=1) + 1e-12).reciprocal()
        ids_mean.append(local_id.mean().item())
        ids_std.append(local_id.std().item())

    return ids_mean, ids_std


def two_nn_intrinsic_dimension(D, eps=1e-12):
    """
    Stable Two-Nearest-Neighbors intrinsic dimension estimator.

    D : pairwise distance matrix (N x N)
        Can be squared or unsquared.
    """

    # Ensure positive distances
    D = torch.sqrt(D + eps)

    # Sort distances
    sorted_D, _ = torch.sort(D, dim=1)

    # First and second neighbors
    r1 = sorted_D[:, 1]
    r2 = sorted_D[:, 2]

    # Avoid degenerate cases
    valid = r2 > r1 + eps
    r1 = r1[valid]
    r2 = r2[valid]

    mu = r2 / (r1 + eps)
    log_mu = torch.log(mu + eps)

    # Global estimator (stable)
    d_hat = 1.0 / log_mu.mean()

    return d_hat.item(), 0


traj_length = 3
k = 30
parameter = "msl"
file_path = "Data/era5_msl_daily_eu.nc"

ds = xr.open_dataset(file_path)
da = ds[parameter]

spatial_dims = [d for d in da.dims if d != "time"]
# (T, H, W) !!! load all data into memory !!!
data = da.transpose("time", *spatial_dims).values.astype(np.float32)
ds.close()

ks = [20, 30, 40]  # range(10, np.sqrt(data.shape[0]).astype(int), 5)
for traj_length in range(1, 16, 2):
    mean_ids = []
    std_ids = []
    start_time = time.time()

    matrix = distance_matrix(data, traj_length, k, q_batch=1024*3, r_chunk=1024*3, device="cuda",
                             exclusion_zone=traj_length, dtype=torch.float32)

    # print("Computing intrinsic dimension...")
    mean_ids, std_ids = mle_intrinsic_dimension(matrix, ks)
    end_time = time.time()

    plt.figure(figsize=(10, 6))
    plt.errorbar(ks, mean_ids, yerr=std_ids, fmt='-o', ecolor='r', capsize=5)
    plt.title(
        f'Estimated Intrinsic Dimension vs Number of Neighbors (Trajectory Length={traj_length})')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Estimated Intrinsic Dimension')
    plt.grid()
    plt.savefig(f"intrinsic_dimension_trajlen{traj_length}.png")

    print(f"Average Intrinsic Dimension for traj_length={traj_length}: {np.mean(mean_ids)} (std: {np.mean(std_ids)})")

# plt.figure(figsize=(10, 6))
# plt.errorbar(range(1, 15, 2), mean_ids, yerr=std_ids, fmt='-o', ecolor='r', capsize=5)
# plt.title('Estimated Intrinsic Dimension vs Trajectory Length')
# plt.xlabel('Trajectory Length')
# plt.ylabel('Estimated Intrinsic Dimension')
# plt.grid()
# plt.show()
