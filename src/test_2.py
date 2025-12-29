import torch
import xarray as xr
import numpy as np
import time

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def knn_scores_streaming(
    nc_path: str,
    var: str,
    traj_length: int,
    k: int = 10,
    q_batch: int = 64,
    r_chunk: int = 1024,
    device: str | None = None,
):
    """
    Streaming kNN trajectory score computation that DOES NOT require X to fit on GPU
    and DOES NOT store NxN distance matrices.

    - Keeps full X on CPU (optionally pinned for faster H2D copies).
    - Copies only the needed blocks to GPU.
    - Computes trajectory distances in blocks and maintains per-row top-k smallest distances.
    - Uses float32 only (no half precision).
    """
    # Load dataset into CPU memory
    with xr.open_dataset(nc_path) as ds:
        da = ds[var]
        spatial_dims = [d for d in da.dims if d != "time"]
        data = da.transpose("time", *spatial_dims).values.astype(np.float32)
        times = ds["time"].values

    T, H, W = data.shape
    D = H * W
    if traj_length < 1 or traj_length > T:
        raise ValueError(f"traj_length must be in [1, {T}], got {traj_length}")
    N = T - traj_length + 1
    if k < 1 or k >= N:
        raise ValueError(f"k must be in [1, {N-1}], got {k}")

    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # CPU tensor (optionally pinned)
    X_cpu = torch.from_numpy(data).reshape(T, D).contiguous()  # float32 CPU
    if dev.type == "cuda":
        X_cpu = X_cpu.pin_memory()

    # CPU norms (float32)
    norms_cpu = (X_cpu * X_cpu).sum(dim=1)  # (T,)

    scores_cpu = torch.empty(N, dtype=torch.float32, device="cpu")

    # Main streaming loops
    for row_start in range(0, N, q_batch):
        row_end = min(row_start + q_batch, N)
        B = row_end - row_start

        # Keep best k per query-row on GPU if available, else CPU
        best = torch.full((B, k), float("inf"), dtype=torch.float32, device=dev)

        for col_start in range(0, N, r_chunk):
            col_end = min(col_start + r_chunk, N)
            C = col_end - col_start

            # Accumulate trajectory distances for this (B, C) block
            acc = torch.zeros((B, C), dtype=torch.float32, device=dev)

            for t in range(traj_length):
                # CPU slices
                Q_cpu = X_cpu[row_start + t : row_end + t]      # (B, D) on CPU
                R_cpu = X_cpu[col_start + t : col_end + t]      # (C, D) on CPU

                # Move to device (float32 only)
                Q = Q_cpu.to(dev, non_blocking=True)
                R = R_cpu.to(dev, non_blocking=True)

                qn = norms_cpu[row_start + t : row_end + t].to(dev, non_blocking=True).unsqueeze(1)  # (B,1)
                rn = norms_cpu[col_start + t : col_end + t].to(dev, non_blocking=True).unsqueeze(0)  # (1,C)

                # squared euclidean: ||q||^2 + ||r||^2 - 2 q·r
                acc += qn + rn - 2.0 * (Q @ R.T)

            # Exclude self matches (only where query index == ref index)
            overlap_start = max(row_start, col_start)
            overlap_end = min(row_end, col_end)
            if overlap_start < overlap_end:
                i0 = overlap_start - row_start
                j0 = overlap_start - col_start
                n_ov = overlap_end - overlap_start
                acc[i0 : i0 + n_ov, j0 : j0 + n_ov].fill_(float("inf"))

            # Update running top-k smallest distances per row
            merged = torch.cat([best, acc], dim=1)  # (B, k+C)
            best, _ = torch.topk(merged, k=k, largest=False, dim=1)

        # Row scores = mean of k nearest distances
        scores_cpu[row_start:row_end] = best.mean(dim=1).to("cpu")

    return scores_cpu, times[:N]


if __name__ == "__main__":
    traj_length = 30
    k = 30
    parameter = "msl"

    start_time = time.time()
    scores, times = knn_scores_streaming(
        "Data/era5_msl_daily_eu.nc",
        parameter,
        traj_length,
        k=k,
        q_batch=256,     # tune based on GPU memory
        r_chunk=2048,    # tune based on GPU memory
        device="cuda",
    )
    end_time = time.time()

    print("scores shape:", scores.shape)
    print("times shape:", times.shape)
    print("Elapsed:", end_time - start_time, "s")
