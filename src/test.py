import os, time
import numpy as np
import xarray as xr
import torch
import plotly.express as px
import pandas as pd
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def knn_scores_recurrence_optimized(
    nc_path: str,
    var: str,
    traj_length: int,
    k: int = 10,
    device: str | None = None,
    # streaming controls
    diag_chunk: int = 8192,      # how many "instant" distances to compute per chunk along a diagonal
    update_chunk: int = 16384,   # how many window-pairs to update top-k per chunk
    exclude_radius: int = 0,     # skip |i-j| <= exclude_radius (recommended to avoid overlap-neighbors)
    use_amp: bool = True,        # speed up dot products on GPU
):
    """
    Exact kNN anomaly score for each window start i (0..N-1), where:
        D_traj(i,j) = sum_{t=0..L-1} ||X_{i+t} - X_{j+t}||^2
    computed via diagonal streaming recurrence:
        d_t(delta) = ||X_t - X_{t+delta}||^2
        traj_d(i, i+delta) = rolling_sum_L(d_t(delta))

    Returns:
        scores_cpu: torch.FloatTensor (N,) on CPU
        times:      times[:N]
    """

    # -------------------- Load data --------------------
    ds = xr.open_dataset(nc_path)
    da = ds[var]
    spatial_dims = [d for d in da.dims if d != "time"]
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    times = ds["time"].values
    ds.close()

    T, H, W = data.shape
    D = H * W
    L = int(traj_length)
    if not (1 <= L <= T):
        raise ValueError(f"traj_length must be in [1, T]. Got {L}, T={T}.")

    N = T - L + 1
    if N <= 0:
        raise ValueError("No windows available (T - L + 1 <= 0).")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # (T, D) on device
    X = torch.from_numpy(data).reshape(T, D).to(dev, dtype=torch.float32)
    del data

    # -------------------- Precompute norms --------------------
    # norms[t] = ||X_t||^2
    norms = (X * X).sum(dim=1)  # (T,)

    # -------------------- Top-k buffers (squared distances) --------------------
    k_eff = min(k, max(0, N - 1))
    if k_eff == 0:
        scores = torch.full((N,), float("nan"), device=dev, dtype=torch.float32)
        return scores.cpu(), times[:N]

    knn_vals = torch.full((N, k_eff), float("inf"), device=dev, dtype=torch.float32)

    # AMP config (only meaningful on CUDA)
    amp_enabled = bool(use_amp and dev.type == "cuda")
    amp_dtype = torch.float16  # good default on most consumer GPUs

    # -------------------- Stream diagonals delta = j - i --------------------
    # Valid window-start pairs are (i, j=i+delta) with i in [0, N-delta-1]
    for delta in tqdm(range(1, N), desc="Diagonal offsets"):
        if exclude_radius and delta <= exclude_radius:
            continue

        # For this delta, instantaneous distances exist for t=0..T-delta-1
        M = T - delta
        if M < L:
            continue

        pair_count = N - delta  # number of trajectory-pairs for this delta

        # Rolling sum state for d_t over length L:
        # keep a ring buffer of last L d-values and a running sum.
        ring = torch.empty((L,), device=dev, dtype=torch.float32)
        ring_pos = 0
        filled = 0
        run_sum = torch.tensor(0.0, device=dev, dtype=torch.float32)

        # We'll produce trajectory distances traj_d[i] for i=0..pair_count-1.
        # These correspond to windows starting at i and i+delta.
        out_i = 0  # how many traj distances have been produced so far

        # We update knn_vals in chunks for efficiency
        # Buffer distances and indices until update_chunk is reached.
        buf_i = []
        buf_dist = []

        def flush_buffers():
            nonlocal buf_i, buf_dist
            if not buf_i:
                return
            idx = torch.cat(buf_i, dim=0)          # i
            dist = torch.cat(buf_dist, dim=0)      # (b,)
            jdx = idx + delta

            # dist is squared traj distance. Update top-k for both endpoints.
            dist_col = dist.unsqueeze(1)  # (b,1)

            merged_i = torch.cat([knn_vals[idx], dist_col], dim=1)
            knn_vals[idx], _ = torch.topk(merged_i, k_eff, dim=1, largest=False)

            merged_j = torch.cat([knn_vals[jdx], dist_col], dim=1)
            knn_vals[jdx], _ = torch.topk(merged_j, k_eff, dim=1, largest=False)

            buf_i.clear()
            buf_dist.clear()

        # Process instantaneous distances along this diagonal in chunks
        # We only need d_t for t up to (pair_count + L - 2) to produce pair_count rolling sums.
        # Because traj_d[i] uses d_{i..i+L-1}, last i = pair_count-1 needs up to t = pair_count+L-2.
        t_max_needed = pair_count + L - 1  # exclusive upper bound on t for d_t
        # Clamp to available M (= T-delta) instantaneous distances
        t_max_needed = min(t_max_needed, M)

        for t0 in range(0, t_max_needed, diag_chunk):
            t1 = min(t0 + diag_chunk, t_max_needed)
            a = X[t0:t1]                 # (b, D)
            b = X[t0 + delta:t1 + delta] # (b, D)

            # d = ||a||^2 + ||b||^2 - 2*(a·b)
            # compute dot in AMP if enabled, then cast back to fp32
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    dot = (a * b).sum(dim=1)
                dot = dot.float()
            else:
                dot = (a * b).sum(dim=1)

            d = norms[t0:t1] + norms[t0 + delta:t1 + delta] - 2.0 * dot
            d = torch.clamp(d, min=0.0)  # avoid tiny negative from roundoff

            # feed each d into rolling sum, emit traj distances when window is full
            for val in d:
                if filled < L:
                    ring[ring_pos] = val
                    run_sum += val
                    ring_pos = (ring_pos + 1) % L
                    filled += 1
                    if filled < L:
                        continue
                else:
                    # subtract the overwritten element, add new
                    run_sum += val - ring[ring_pos]
                    ring[ring_pos] = val
                    ring_pos = (ring_pos + 1) % L

                # now we have a full window => traj distance for current out_i
                if out_i < pair_count:
                    # buffer update
                    buf_i.append(torch.tensor([out_i], device=dev, dtype=torch.long))
                    buf_dist.append(run_sum.view(1))
                    out_i += 1

                    if out_i % update_chunk == 0:
                        flush_buffers()

                if out_i >= pair_count:
                    break

            if out_i >= pair_count:
                break

        flush_buffers()

    # Final score: mean of sqrt of squared distances
    scores = torch.sqrt(knn_vals).mean(dim=1)
    return scores.cpu(), times[:N]


if __name__ == "__main__":
    traj_length = 11
    k = 30
    parameter = "msl"

    start_time = time.time()
    scores, times = knn_scores_recurrence_optimized(
        "Data/era5_msl_daily_eu.nc",
        parameter,
        traj_length=traj_length,
        k=k,
        device="cuda",
        diag_chunk=8192,
        update_chunk=16384,
        exclude_radius=0,   # try e.g. 15 or 30 to avoid trivial overlapping neighbors
        use_amp=True,
    )
    end_time = time.time()
    print(f"Trajectory of length {traj_length} k-NN scoring completed in {end_time - start_time:.2f} seconds.")

    out_dir = "result/traj/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{parameter}_knn_scores_traj{traj_length}_{k}.npz")
    np.savez(out_path, scores=scores.numpy(), time=times)

    fig = px.line(x=times, y=scores.numpy(), labels={"x": "Time", "y": "Anomaly Score"})
    fig.show()

    top100_idx = np.argsort(-scores.numpy())[:100]
    top100_dates = [pd.to_datetime(times[i]).date() for i in top100_idx]
    df = pd.DataFrame({"date": top100_dates, "anomaly_score": scores.numpy()[top100_idx]})
    df.to_csv(os.path.join(out_dir, f"{parameter}_top100_anomalous_dates_{traj_length}_{k}.csv"), index=False)
