# Spatiotemporal Anomaly Detection

A Python package for detecting rare and anomalous spatiotemporal patterns in climate data using trajectory-based k-nearest neighbor (kNN) scoring. Implemented in PyTorch for efficient CPU and GPU computation.

[![License: CeCILL](https://img.shields.io/badge/license-CeCILL-blue.svg)](LICENSE)

![Spatiotemporal Anomaly Detection](Figures/chaos.jpg)

---

## Overview

Climate and weather data consists of large sequences of spatial fields, gridded snapshots of variables like temperature, pressure, or precipitation over a geographic domain. Identifying which of these patterns are genuinely rare, rather than simply seasonal or weather-regime-driven, is a core challenge in climate science.

This project addresses that challenge by embedding consecutive spatial fields into **trajectories**, compact representations of how a field evolves over a short time window, and scoring each trajectory by how far it sits from its nearest historical counterparts. Trajectories with large distances to their k nearest neighbors are flagged as anomalous. The approach is non-parametric, requires no labels, and makes no distributional assumptions about the data.

The implementation supports GPU acceleration, memory-efficient blocked computation, preprocessing pipelines, and a suite of case study and benchmarking scripts.

---

## How It Works

The algorithm proceeds in 5 stages:

1. **Optional preprocessing.** Raw spatial fields are cleaned and normalized before scoring:
   - Leap days (Feb 29) are removed to ensure a fixed 365-day annual cycle.
   - The seasonal cycle is subtracted by computing a 365-day climatological mean and removing it from each day, leaving time series of anomalies.
   - Spatial fields are weighted by the cosine of latitude to correct for the convergence of meridians at high latitudes.
   - Each grid point is standardized to unit variance across time.
   - Spatial dimensionality can be reduced via PCA (Randomized SVD).

2. **Spatial distance computation.** The pairwise Euclidean distance between every pair of daily spatial fields is computed, producing a T × T distance matrix. This is done in blocked matrix operations that keep memory use bounded, and symmetry is exploited to halve the computation: only the upper triangle is computed and then mirrored.

3. **Trajectory distance via recurrence.** Rather than summing L spatial distances from scratch for every trajectory pair, the algorithm uses a sliding-window recurrence: the distance between two trajectories starting one day later equals the previous distance, minus the spatial distance of the day that left the window, plus the spatial distance of the day that entered it. This reduces the per-step cost from O(L) to O(1).

4. **kNN scoring with temporal exclusion.** For each trajectory, the k smallest distances are found among all other trajectories, excluding those within a temporal window of ±L days (the exclusion zone). This prevents trivially close time-adjacent patterns from dominating the neighbor list. The anomaly score is the mean of these k distances.

5. **Output.** Each trajectory receives a scalar score. High scores indicate rare patterns with no close historical analogues; low scores indicate common, frequently recurring patterns.

---

## Features

- PyTorch-based implementation with seamless CPU/CUDA support
- netCDF4 / xarray input (ERA5-compatible, standard climate data format)
- Optional: Cosine latitude weighting for geophysical correctness
- Optional: Seasonal cycle removal (365-day climatology, leap-day aware)
- Optional: pixel-wise standardization
- Optional: PCA dimensionality reduction via Randomized SVD before scoring
- Configurable trajectory length, number of neighbors, and exclusion zone
- Trajectory analogue search: find the closest historical matches to a query period
- Intrinsic dimension estimation (MLE and Two-Nearest-Neighbors estimators)
- Full trajectory distance matrix export to HDF5 for downstream analysis
- FAISS-based CPU and GPU baselines for benchmarking comparison
- Case study scripts for climate extreme event detection and clustering

---

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate trajectory_knn
pip install -e .
```

### Optional: FAISS baselines

FAISS is only needed to run the baseline benchmarks, not the main algorithm.

```bash
# CPU variant
conda env create -f env_faiss_cpu.yml

# GPU variant
conda env create -f env_faiss_gpu.yml
```

---

## Quick Start

### Python API

```python
from src.rarity_scoring_base import knn_scores

scores = knn_scores(
    nc_path="Data/era5_msl_daily_eu.nc",
    var="msl",
    traj_length=5,       # number of consecutive days per trajectory
    k=10,                # number of nearest neighbors
    q_batch=1024,        # query batch size (tune to GPU memory)
    r_chunk=1024,        # reference chunk size (tune to GPU memory)
    device="cuda",       # "cpu" or "cuda"
    exclusion_zone=10    # temporal exclusion window in days
)
# scores[i] is the anomaly score for the trajectory starting at day i
```

### Preprocessing pipeline

```python
import xarray as xr
import numpy as np
from src.preprocessing import (
    remove_bisex_dailydata,
    remove_seasonal_cycle365,
    cos_lat_weighting,
    pixelwise_standardize,
)

ds = xr.open_dataset("Data/era5_msl_daily_eu.nc")
data = ds["msl"].values          # shape (T, H, W)
time = ds["time"].values

data, time = remove_bisex_dailydata(data, time)
data = remove_seasonal_cycle365(data, time)
data = cos_lat_weighting(data, ds["lat"].values, len(ds["lon"]))
data = pixelwise_standardize(data)
```

---

## Case Studies

The `case_studies/` directory contains end-to-end scripts for real climate applications.

### Anomaly scoring (CLI)

```bash
python case_studies/score.py \
    --traj_length 5 \
    --k 10 \
    --parameter msl \
    --file_path Data/era5_msl_daily_eu.nc \
    --device cuda \
    --remove_seasonal_cycle \
    --cos_lat_weighting
```

Output: an NPZ file with scores and timestamps, and a CSV of the top 100 most anomalous dates.

### Cluster analysis of extreme events

```bash
python case_studies/cluster_analysis.py \
    --file_path Data/t2m_daily_avg_1950_2023.nc \
    --date_path case_studies/results/t2m/t2m_trajlen5_k10_top100.csv
```

Applies K-means to the top anomalies and produces composite spatial maps per cluster.

### Storm matching

```bash
python case_studies/emdat_storm_matching.py   # match detected anomalies to EM-DAT events
python case_studies/storm_matching_random.py  # random baseline for statistical validation
```

---

## Benchmarks and Experiments

Performance scaling experiments are in `experiments/`:

```bash
# Main algorithm runtime
python experiments/myalgo.py

# FAISS baselines
python experiments/faiss_cpu_only.py
python experiments/faiss_gpu_only.py

# Scaling with number of neighbors k
python experiments/k_scaling.py

# Memory profiling
python experiments/memory.py

# Visualize results
python experiments/visu.py
```

Results (CSV) and figures (PDF/PNG) are written to `experiments/results/` and `experiments/figures/`.

---

## Project Structure

```
.
├── src/
│   ├── rarity_scoring_base.py       # Main kNN anomaly detection algorithm
│   ├── rarity_scoring_exclusion.py  # Variant with stricter exclusion zone handling
│   ├── rarity_scoring_interval.py   # Interval-based scoring variant
│   ├── distance_matrix.py           # Full trajectory distance matrix (HDF5 export)
│   ├── analogue_traj.py             # Historical analogue search
│   ├── intrinsic_dimension.py       # Intrinsic dimensionality estimators (MLE, 2NN)
│   ├── preprocessing.py             # Leap-day removal, deseasonalization, weighting
│   └── utils.py                     # Shared utilities
├── experiments/
│   ├── myalgo.py                    # Runtime benchmark: main algorithm
│   ├── myalgo_lowmem.py             # Runtime benchmark: low-memory variant
│   ├── faiss_cpu_only.py            # FAISS CPU baseline
│   ├── faiss_gpu_only.py            # FAISS GPU baseline
│   ├── k_scaling.py                 # Scaling with k
│   ├── memory.py                    # Memory profiling
│   ├── visu.py                      # Result visualization
│   ├── results/                     # Benchmark output (CSV)
│   └── figures/                     # Generated plots
├── case_studies/
│   ├── score.py                     # CLI scoring pipeline
│   ├── cluster_analysis.py          # K-means clustering of top anomalies
│   ├── emdat_storm_matching.py      # Match anomalies to EM-DAT storm events
│   ├── storm_matching_random.py     # Random baseline for storm matching
│   ├── compare_dates.py             # Date comparison utilities
│   └── results/                     # Output data and figures
├── environment.yml                  # Main conda environment
├── env_faiss_cpu.yml                # FAISS CPU conda environment
├── env_faiss_gpu.yml                # FAISS GPU conda environment
└── pyproject.toml                   # Package metadata
```



---
<!-- 
## Citation

If you use this software in your research, please cite:

```bibtex

```

--- -->

## License

This project is licensed under the [CeCILL Free Software License Agreement](LICENSE).

<!-- Author: Guillaume Coulaud — [guillaume.coulaud@inria.fr](mailto:guillaume.coulaud@inria.fr) -->
