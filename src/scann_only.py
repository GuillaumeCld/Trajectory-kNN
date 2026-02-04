"""
ScaNN feasibility experiment
ECML PKDD – Applied Data Science
"""

import os
import time
import csv
import numpy as np
import scann
import tensorflow as tf
# ---------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------
T_VALUES = [365 * 10, 365 * 25, 365 * 50, 365 * 75]
TRAJ_LENGTHS = [1, 2, 4, 8, 16]

H, W = 180, 280
K = 10

RESULTS_FILE = "scann_results_gpu.csv"

# Fixed ScaNN configuration
NUM_LEAVES = 64
LEAVES_TO_SEARCH = 16

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
                "scann_time",
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
# Single run (may get killed)
# ---------------------------------------------------------
def run_scann(T, traj_length):
    data = np.random.rand(T, H, W).astype(np.float32)
    mat = build_matrix(data, traj_length)
    
    start = time.time()
    searcher = scann.scann_ops_pybind.builder(
        mat, K, "dot_product"
    ).tree(
        num_leaves=NUM_LEAVES,
        num_leaves_to_search=LEAVES_TO_SEARCH,
    ).score_ah(
        2, anisotropic_quantization_threshold=0.2
    ).build()

   
    mat = tf.convert_to_tensor(mat)
    _ = searcher.search_batched(mat)          
    elapsed = time.time() - start

    return elapsed, mat.shape[0], mat.shape[1]


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
if __name__ == "__main__":
    done = load_completed()

    for T in T_VALUES:
        for L in TRAJ_LENGTHS:
            if (T, L) in done:
                print(f"[SKIP] T={T}, L={L}")
                break

            print(f"[RUN]  T={T}, L={L}")

            try:
                t, n, d = run_scann(T, L)
                save_result(
                    {
                        "T": T,
                        "traj_length": L,
                        "num_vectors": n,
                        "dim": d,
                        "scann_time": f"{t:.4f}",
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
                        "scann_time": "",
                        "status": "FAILED",
                    }
                )
                print(f"[FAIL] {e}")
