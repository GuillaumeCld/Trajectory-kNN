"""
"""

import time
import numpy as np
import pandas as pd
# import faiss
import torch
from src.rarity_scoring_base import compute_distances_and_scores

# ---------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------
T = 365 * 75
TRAJ_LENGTH = 1

# 50 100 150 200 250
H, W = 180, 280
K = [1, 10, 20, 30, 40, 50]  # number of nearest neighbors


# ---------------------------------------------------------
# TRAKNN
# ---------------------------------------------------------
def run_algo():

    for device in ["cpu", "cuda"]:
        runtime = []
        for k in K:
            print(
                f"Running algo with traj_length={TRAJ_LENGTH}, k={k}, device={device}...")
            data = np.random.rand(T, H, W).astype(np.float32)
            start = time.time()
            _ = compute_distances_and_scores(
                data,
                TRAJ_LENGTH,
                k,
                1024,
                1024,
                device=device,
                dtype=torch.float32,
                exclusion_zone=TRAJ_LENGTH,
            )

            elapsed = time.time() - start
            runtime.append(elapsed)

        df = pd.DataFrame({
            "k": K,
            "runtime": runtime
        })
        df.to_csv(
            f"experiments/results/algo_{device}_k_trajlen{TRAJ_LENGTH}.csv", index=False)


def build_matrix(data, traj_length):
    T = data.shape[0]
    mat = np.empty(
        (T - traj_length + 1, H * W * traj_length),
        dtype=np.float32,
    )
    for i in range(len(mat)):
        mat[i] = data[i: i + traj_length].reshape(-1)
    return mat


# def run_faiss():
#     data = np.random.rand(T, H, W).astype(np.float32)
#     use_gpu = True  
#     runtime = []
#     for k in K:
#         start = time.time()
#         mat = build_matrix(data, TRAJ_LENGTH)
#         dim = mat.shape[1]

#         # Build FAISS index
#         index = faiss.IndexFlatIP(dim)  # Inner product similarity
#         index_cpu = faiss.IndexFlatIP(dim)  # Inner product similarity

#         if use_gpu:
#             # Move to GPU
#             res = faiss.StandardGpuResources()  # Allocate GPU resources
#             index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
#         else:
#             index = index_cpu

#         index.add(mat)

#         # Search all vectors against themselves
#         D, I = index.search(mat, k)
#         elapsed = time.time() - start
#         runtime.append(elapsed)
#         print( f"FAISS search with k={k} took {elapsed:.4f} seconds.")
#     df = pd.DataFrame({
#         "k": K,
#         "runtime": runtime
#     })  
#     df.to_csv(
#         f"experiments/results/faiss_gpu_k_trajlen{TRAJ_LENGTH}.csv", index=False)


if __name__ == "__main__":

    run_algo()
    # run_faiss()
