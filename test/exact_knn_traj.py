import torch
import xarray as xr
import numpy as np

from src.knn_traj import knn_scores

@torch.compile
def double_precision_scoring(nc_path, var, k, traj_length):
    ds = xr.open_dataset(nc_path)
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]

    data = da.transpose("time", *spatial_dims).values.astype(np.float64)
    time = ds["time"].values


    ds.close()

    T, H, W = data.shape
    D = H * W  

    device = "cpu" 
    dev = torch.device(device)

    X = torch.from_numpy(data).to(torch.float64).reshape(
        T, D).to(dev)
    
    del data

    X_traj = torch.stack([X[i:T - traj_length + i + 1] for i in range(traj_length)], dim=1).reshape(T - traj_length + 1, traj_length * D)
    del X

    distances = torch.cdist(X_traj, X_traj, p=2)  # (T, T) distances matrix on device

    # Take the mean distance to k nearest neighbors excluding self
    sorted_distances, _ = torch.topk(distances, k=k+1, largest=False)
    knn_distances = sorted_distances[:, 1:k+1]
    scores = knn_distances.mean(dim=1)  

    return scores, time

nc_path = "Data/era5_msl_daily_eu_small.nc"
var = "msl"
k = 1
traj_length = 7 
 
scores_double, _ = double_precision_scoring(nc_path, var, k, traj_length)

scores_single, _ = knn_scores(nc_path, var, traj_length, k=k, device="cuda")

# Numerical accuracy check
diff = torch.abs(scores_double - scores_single.to(torch.float64))
# diff stats
print(f"Max difference: {diff.max().item():.2e}")
print(f"Mean difference: {diff.mean().item():.2e}")
print(f"Std difference: {diff.std().item():.2e}")
# Relative error
relative_error = diff / torch.clamp_min(torch.abs(scores_double), 1e-6)
print(f"Max relative error: {relative_error.max().item():.2e}") 
print(f"Mean relative error: {relative_error.mean().item():.2e}")
print(f"Std relative error: {relative_error.std().item():.2e}")