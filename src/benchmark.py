"""
TRAKNN Benchmark — execution time and peak memory for all three versions.

Data loading is included inside the timed region for every version
so that the comparison reflects the true end-to-end cost a user pays.

Usage
-----
    python benchmark.py

Results are printed to stdout and saved to benchmark_results.csv.
"""

import time
import gc
import csv
import argparse
import tracemalloc

import torch

from traknn import rarity_scoring_base

from traknn_streaming import (
    rarity_scoring_buffer,
    rarity_scoring_ooc,
    _load_netcdf,
    _open_reader,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH    = "Data/era5_gfs_msl_daily.nc"
PARAMETER    = "msl"
TRAJ_LENGTH  = 7
K            = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float32

# GeMM tile sizes — identical across versions for a fair comparison
Q_BATCH      = 1024    # row tile size        (versions 1 and 2)
R_CHUNK      = 512    # column tile size     (all versions)
BATCH_SIZE   = 1024    # buffer batch size B  (versions 2 and 3)

# =============================================================================
# Measurement helpers
# =============================================================================

def _reset():
    """Free cached memory and reset VRAM peak counter before each run."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _run(label: str, fn) -> dict:
    """
    Execute fn() — a zero-argument callable that performs data loading
    AND the algorithm — and measure wall-clock time, peak host RAM
    (tracemalloc) and peak VRAM (torch.cuda).

    Data loading is inside fn() so all three versions are measured
    end-to-end on equal terms.

    Returns dict with keys: version, time_s, ram_mb, vram_mb.
    """
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    _reset()
    tracemalloc.start()
    t0 = time.perf_counter()

    scores = fn()

    elapsed = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram = _peak_vram_mb()

    print(f"\n  [{label}] done")
    print(f"    Time        : {elapsed:.1f} s")
    print(f"    Peak RAM    : {peak_ram / 1e6:.0f} MB")
    print(f"    Peak VRAM   : {peak_vram:.0f} MB")
    print(f"    scores[0:5] : {scores[:5].tolist()}")

    return {
        "version" : label,
        "time_s"  : round(elapsed, 2),
        "ram_mb"  : round(peak_ram / 1e6, 0),
        "vram_mb" : round(peak_vram, 0),
    }


# =============================================================================
# Per-version callables (data loading inside each one)
# =============================================================================

def run_v1():
    data = _load_netcdf(DATA_PATH, PARAMETER)
    return rarity_scoring_base(
        data,
        traj_length    = TRAJ_LENGTH,
        k              = K,
        r_chunk        = R_CHUNK,
        q_batch        = Q_BATCH,
        device         = DEVICE,
        dtype          = DTYPE,
        exclusion_zone = TRAJ_LENGTH,
    )


def run_v2():
    data = _load_netcdf(DATA_PATH, PARAMETER)
    return rarity_scoring_buffer(
        data,
        traj_length    = TRAJ_LENGTH,
        k              = K,
        r_chunk        = R_CHUNK,
        batch_size     = BATCH_SIZE,
        device         = DEVICE,
        dtype          = DTYPE,
        exclusion_zone = TRAJ_LENGTH,
    )


def run_v3():
    reader, T, _, _ = _open_reader(DATA_PATH, PARAMETER)
    return rarity_scoring_ooc(
        reader,
        T              = T,
        traj_length    = TRAJ_LENGTH,
        k              = K,
        r_chunk        = R_CHUNK,
        batch_size     = BATCH_SIZE,
        device         = DEVICE,
        dtype          = DTYPE,
        exclusion_zone = TRAJ_LENGTH,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    global DATA_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Path to the NetCDF input file.",
    )
    args = parser.parse_args()
    DATA_PATH = args.data_path

    print(f"Device      : {DEVICE}")
    print(f"Data        : {DATA_PATH}")
    print(f"traj_length : {TRAJ_LENGTH}")
    print(f"k           : {K}")
    print(f"q_batch     : {Q_BATCH}  r_chunk : {R_CHUNK}  batch_size : {BATCH_SIZE}")
    print("\nNote: data loading is included in measured time for all versions.")

    results = [
        _run("v1_base",   run_v1),
        _run("v2_buffer", run_v2),
        _run("v3_ooc",    run_v3),
    ]

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    w = 60
    print(f"\n{'='*w}")
    print("  SUMMARY  (loading + algorithm)")
    print(f"{'='*w}")
    print(f"  {'Version':<14} {'Time (s)':>10} {'RAM (MB)':>10} {'VRAM (MB)':>10}")
    print(f"  {'-'*(w-2)}")
    for r in results:
        print(
            f"  {r['version']:<14}"
            f"{r['time_s']:>10.1f}"
            f"{r['ram_mb']:>10.0f}"
            f"{r['vram_mb']:>10.0f}"
        )
    print(f"{'='*w}")

    # ------------------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------------------
    out = "benchmark_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["version", "time_s", "ram_mb", "vram_mb"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
