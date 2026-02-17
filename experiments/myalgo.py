"""
My algo running time experiment
ECML PKDD – Applied Data Science
"""

import os
import time
import csv
import numpy as np
from src.rarity_scoring_base import compute_distances_and_scores
import torch 

# ---------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------
T_VALUES = [365 * 75]
TRAJ_LENGTHS = [1]

# 50 100 150 200 250
H, W = 250, 250
K = 10  # number of nearest neighbors
DEVICE = "cpu"

RESULTS_FILE = f"experiments/results/algo_{DEVICE}_{H}.csv"

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def load_completed():
    if not os.path.exists(RESULTS_FILE):
        return set()
    with open(RESULTS_FILE, "r") as f:
        return {
            (int(r["T"]), int(r["traj_length"]))
            for r in csv.DictReader(f)
        }


def save_result(row):
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "traj_length",
                "num_vectors",
                "dim",
                "faiss_time",
                "status",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def build_matrix(data, traj_length):
    T = data.shape[0]
    mat = np.empty(
        (T - traj_length + 1, H * W * traj_length),
        dtype=np.float32,
    )
    for i in range(len(mat)):
        mat[i] = data[i : i + traj_length].reshape(-1)
    return mat


# ---------------------------------------------------------
# Single run on GPU
# ---------------------------------------------------------
def run_algo(T, traj_length ):
    data = np.random.rand(T, H, W).astype(np.float32)
    start = time.time()
    _ = compute_distances_and_scores(
        data,
        traj_length,
        K,
        1024,
        1024,
        device=DEVICE,
        dtype=torch.float32,
        exclusion_zone=traj_length,
    )
    
    elapsed = time.time() - start

    return elapsed, 0 , 0


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
if __name__ == "__main__":
    done = load_completed()

    for T in T_VALUES:
        for L in TRAJ_LENGTHS:
            if (T, L) in done:
                print(f"[SKIP] T={T}, L={L}")
                continue

            print(f"[RUN]  T={T}, L={L}")

            try:
                t, n, d = run_algo(T, L)
                save_result(
                    {
                        "T": T,
                        "traj_length": L,
                        "num_vectors": n,
                        "dim": d,
                        "faiss_time": f"{t:.4f}",
                        "status": "OK",
                    }
                )
                print(f"[OK]   {t:.2f}s")

            except Exception as e:
                save_result(
                    {
                        "T": T,
                        "traj_length": L,
                        "num_vectors": "",
                        "dim": "",
                        "faiss_time": "",
                        "status": "FAILED",
                    }
                )
                print(f"[FAIL] {e}")
