"""
FAISS-GPU feasibility experiment
ECML PKDD – Applied Data Science
"""

import os
import time
import csv
import numpy as np
import faiss  # Make sure you have faiss-gpu installed

# ---------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------
T_VALUES = [365 * 75]
TRAJ_LENGTHS = [1]

H, W = 250, 250
K = 10  # number of nearest neighbors

RESULTS_FILE = f"experiments/results/faiss_results_gpu_{H}.csv"

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
def run_faiss_gpu(T, traj_length, use_gpu=True):
    data = np.random.rand(T, H, W).astype(np.float32)
    start = time.time()

    mat = build_matrix(data, traj_length)
    dim = mat.shape[1]

    # CPU index first
    index_cpu = faiss.IndexFlatIP(dim)  # Inner product similarity

    if use_gpu:
        # Move to GPU
        res = faiss.StandardGpuResources()  # Allocate GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
    else:
        index = index_cpu

    index.add(mat)

    # Search all vectors against themselves
    D, I = index.search(mat, K)
    elapsed = time.time() - start

    return elapsed, mat.shape[0], dim


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
                t, n, d = run_faiss_gpu(T, L)
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
