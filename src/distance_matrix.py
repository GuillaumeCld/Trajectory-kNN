import torch
import h5py
import numpy as np
import xarray as xr

def blocked_norm_compute(X, block_size, dev):
    T, D = X.shape
    norms = torch.empty(T, dtype=torch.float32, device=dev)
    for block_start in range(0, T, block_size):
        block_end = min(block_start + block_size, T)
        block = X[block_start:block_end]
        norms[block_start:block_end] = (block * block).sum(dim=1)
    return norms

def build_matrix(
    nc_path,
    var,
    traj_length,
    k,
    q_batch,
    r_chunk,
    device,
    dtype,
    exclusion_zone,
    h5_path="distances_traj.h5",
):

    ds = xr.open_dataset(nc_path)
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    # (T, H, W) !!! load all data into memory !!!
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    ds.close()

    T, H, W = data.shape
    D = H * W
    print(f"Data shape: (T={T}, H={H}, W={W}), vectorized to D={D}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    X = torch.from_numpy(data).to(dtype).reshape(T, D).to(dev)

    # Remove invalid columns
    # valid_mask = ~torch.isnan(X).any(dim=0)
    # valid_mask = valid_mask & (X != 0).any(dim=0)
    # X = X[:, valid_mask]
    # D = valid_mask.sum().item()
    # print(f"Number of valid spatial dimensions: {D}")

    norms = blocked_norm_compute(X, r_chunk, dev)

    N = T - traj_length + 1

    # ---------------------------------------------------------
    #  Compute full spatial distance matrix (CPU)
    # ---------------------------------------------------------
    distances_space = torch.empty((T, T), dtype=dtype, device="cpu")

    for row_start in range(0, T, q_batch):
        row_end = min(row_start + q_batch, T)

        rows = X[row_start:row_end]
        row_norms = norms[row_start:row_end]

        for column_start in range(0, T, r_chunk):
            column_end = min(column_start + r_chunk, T)

            cols = X[column_start:column_end]
            col_norms = norms[column_start:column_end]

            distances = (
                row_norms[:, None]
                + col_norms[None, :]
                - 2.0 * (rows @ cols.T)
            )

            distances_space[row_start:row_end, column_start:column_end] = distances.to("cpu")

    distances_space = distances_space.clamp(min=0.0)

    # ---------------------------------------------------------
    #  Create HDF5 dataset for trajectory distances
    # ---------------------------------------------------------
    L = traj_length
    N = T - L + 1


    distances_traj_matrix = torch.empty((N, N), dtype=dtype, device="cpu")

    # Compute first row D_0
    distances_traj0 = torch.zeros(N, dtype=dtype, device="cpu")
    for t_offset in range(L):
        distances_traj0 += distances_space[t_offset, t_offset:t_offset + N]

    distances_traj = distances_traj0.clone()
    distances_traj_matrix[0, :] = distances_traj

    # Compute remaining rows using recurrence
    for i in range(1, N):
        distances_traj[1:] = (
            distances_traj[:-1]
            - distances_space[i - 1, 0:N-1]
            + distances_space[i - 1 + L, L:T]
        )

        distances_traj[0] = distances_traj0[i]
        distances_traj = distances_traj.clamp(min=0.0)
        distances_traj_matrix[i, :] = distances_traj

    del distances_space # free memory
    distances_traj_matrix = torch.sqrt(distances_traj_matrix)
    with h5py.File(h5_path, "w") as f:

        dset = f.create_dataset(
            "distances_traj",
            shape=(N, N),
            dtype=np.float32,
            chunks=(1, N),        # row-wise chunking
            compression="gzip"
        )

        # Write entire matrix at the end
        dset[:, :] = distances_traj_matrix.numpy()
