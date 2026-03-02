"""
"""

import time
import numpy as np
import pandas as pd
import faiss
import torch
from src.rarity_scoring_base import compute_distances_and_scores
from memory_profiler import profile

# ---------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------
T = 365 * 75
TRAJ_LENGTH = 2

# 50 100 150 200 250
H, W = 180, 280
K = 10  # number of nearest neighbors


# ---------------------------------------------------------
# Single run on GPU
# ---------------------------------------------------------
@profile
def run_algo():
    device="cpu"
    data = np.random.rand(T, H, W).astype(np.float32)
    _ = compute_distances_and_scores(
        data,
        TRAJ_LENGTH,
        K,
        1024,
        1024,
        device=device,
        dtype=torch.float32,
        exclusion_zone=TRAJ_LENGTH,
    )


# def build_matrix(data, traj_length):
#     T = data.shape[0]
#     mat = np.empty(
#         (T - traj_length + 1, H * W * traj_length),
#         dtype=np.float32,
#     )
#     for i in range(len(mat)):
#         mat[i] = data[i: i + traj_length].reshape(-1)
#     return mat

# @profile
# def run_faiss():
#     data = np.random.rand(T, H, W).astype(np.float32)
#     use_gpu = False  

#     mat = build_matrix(data, TRAJ_LENGTH)
#     del data
#     dim = mat.shape[1]

#     # Build FAISS index
#     index = faiss.IndexFlatIP(dim)  # Inner product similarity
#     index_cpu = faiss.IndexFlatIP(dim)  # Inner product similarity

#     if use_gpu:
#         # Move to GPU
#         res = faiss.StandardGpuResources()  # Allocate GPU resources
#         index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
#     else:
#         index = index_cpu

#     index.add(mat)

#     # Search all vectors against themselves
#     D, I = index.search(mat, K)
    

if __name__ == "__main__":

    run_algo()
    # run_faiss()
