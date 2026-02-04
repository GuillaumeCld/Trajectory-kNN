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
def d_space_block(rows, row_norms, cols, col_norms):
    return row_norms[:, None] + col_norms[None, :] - 2.0 * (rows @ cols.T)


@torch.no_grad()
def compute_d_space(rows, row_norms, cols, col_norms):
    # rows: (R, D), cols: (C, D)
    return row_norms[:, None] + col_norms[None, :] - 2.0 * (rows @ cols.T)


def compute_distances_and_scores(
    data, traj_length, k, q_batch, r_chunk, device, dtype, exclusion_zone
):
    T, H, W = data.shape
    D = H * W
    L = traj_length
    N = T - L + 1

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    X = torch.from_numpy(data).to(dtype).reshape(T, D).to(dev)
    norms = blocked_norm_compute(X, r_chunk, dev)

    scores = torch.empty(N, dtype=dtype, device="cpu")

    # -------------------------
    # D_0 computation (batched)
    # -------------------------
    distances_traj0 = torch.zeros(N, dtype=dtype, device=dev)

    for t in range(L):
        rows = X[t:t+1]
        row_norms = norms[t:t+1]
        cols = X[t:t+N]
        col_norms = norms[t:t+N]

        distances_traj0 += d_space_block(rows, row_norms, cols, col_norms).squeeze(0)

    distances_traj = distances_traj0.clone()

    sorted_distances, _ = torch.topk(distances_traj, k=k+1, largest=False)
    scores[0] = sorted_distances[1:k+1].mean().cpu()

    # --------------------------------
    # Recurrence computed in batches
    # --------------------------------
    for i0 in range(1, N, q_batch):
        i1 = min(i0 + q_batch, N)
        B = i1 - i0

        # ---- outgoing block ----
        rows_out = X[i0-1:i1-1]              # (B, D)
        row_norms_out = norms[i0-1:i1-1]
        cols_out = X[0:N-1]                  # (N-1, D)
        col_norms_out = norms[0:N-1]

        d_out = d_space_block(
            rows_out, row_norms_out, cols_out, col_norms_out
        )                                    # (B, N-1)

        # ---- incoming block ----
        rows_in = X[i0+L-1:i1+L-1]            # (B, D)
        row_norms_in = norms[i0+L-1:i1+L-1]
        cols_in = X[L:T]                     # (N-1, D)
        col_norms_in = norms[L:T]

        d_in = d_space_block(
            rows_in, row_norms_in, cols_in, col_norms_in
        )                                    # (B, N-1)

        # ---- update trajectories ----
        for b in range(B):
            i = i0 + b
            distances_traj[1:] = (
                distances_traj[:-1]
                - d_out[b]
                + d_in[b]
            )
            distances_traj[0] = distances_traj0[i]
            distances_traj = distances_traj.clamp(min=0.0)

            # exclusion zone
            lo = max(0, i - exclusion_zone + 1)
            hi = min(N, i + exclusion_zone)
            distances_traj[lo:hi] = float("inf")

            sorted_distances, sorted_indices = torch.topk(
                distances_traj, k=k * exclusion_zone, largest=False
            )

            current_mins = [sorted_indices[0].item()]
            for idx in sorted_indices[1:]:
                idx_item = idx.item()
                if all(abs(idx_item - cm) >= exclusion_zone for cm in current_mins):
                    current_mins.append(idx_item)
                if len(current_mins) >= k:
                    break

            scores[i] = distances_traj[current_mins].mean().cpu()

    return scores








