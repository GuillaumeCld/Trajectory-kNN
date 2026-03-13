import time
import tracemalloc
import numpy as np
import torch
import pandas as pd

# ----------------------------
# Original NumPy-style versions
# ----------------------------
def compute_md_index_from_md(month, day):
    month_cumsum = torch.tensor(
        [0,31,59,90,120,151,181,212,243,273,304,334],
        dtype=torch.long,
        device=month.device
    )

    return month_cumsum[month - 1] + (day - 1)

def remove_seasonal_cycle365_numpy(data, time):
    time = pd.to_datetime(time).normalize()

    # Check for leap days
    if np.any((time.month == 2) & (time.day == 29)):
        raise ValueError("Leap days present. Remove them first.")

    # Encode (month, day) as unique integers: 1..365
    month_cumsum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    md_index = month_cumsum[time.month.values - 1] + (time.day.values - 1)

    # Preallocate seasonal cycle array
    seasonal_cycle = np.zeros((365, *data.shape[1:]), dtype=data.dtype)
    counts = np.zeros(365, dtype=int)

    # Accumulate sums
    np.add.at(seasonal_cycle, md_index, data)
    np.add.at(counts, md_index, 1)

    # Divide by counts (only where count > 0)
    seasonal_cycle[counts > 0] /= counts[counts > 0][:, None, None]

    # Deseasonalize in place
    data -= seasonal_cycle[md_index]
    return data

def cos_lat_weighting_numpy(data, latitudes):
    latitudes = np.asarray(latitudes)
    weights = np.sqrt(np.cos(np.deg2rad(latitudes)))[:, np.newaxis]
    return data * weights


def pixelwise_standardize_numpy(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


# ----------------------------
# Efficient PyTorch versions
# ----------------------------

def cos_lat_weighting_torch_(data, latitudes):
    weights = torch.sqrt(torch.cos(torch.deg2rad(latitudes))).view(1, -1, 1)
    data.mul_(weights)
    return data


def pixelwise_standardize_torch_(data):
    mean = data.mean(dim=0)
    std = data.std(dim=0).clamp_min_(1e-12)
    data.sub_(mean)
    data.div_(std)
    return data


def remove_seasonal_cycle365_torch(data: torch.Tensor, md_index: torch.Tensor):
    """
    Remove the 365-day seasonal cycle from data in-place.

    Parameters
    ----------
    data : torch.Tensor
        Shape (T, ...), where T is time.
    md_index : torch.Tensor
        Long tensor of shape (T,) containing day-of-year indices in [0, 364].
        Leap days must already be removed.

    Returns
    -------
    data : torch.Tensor
        Deseasonalized tensor (modified in-place).
    """

    T = data.shape[0]
    device = data.device
    dtype = data.dtype

    flat = data.view(T, -1)
    seasonal = torch.zeros(365, flat.shape[1], dtype=dtype, device=device)
    counts = torch.zeros(365, dtype=dtype, device=device)

    # accumulate sums
    seasonal.index_add_(0, md_index, flat)

    # accumulate counts
    ones = torch.ones(T, dtype=dtype, device=device)
    counts.index_add_(0, md_index, ones)

    # compute means
    seasonal /= counts.clamp_min(1).unsqueeze(1)

    # subtract seasonal cycle in-place
    flat.sub_(seasonal.index_select(0, md_index))

    return data
# ----------------------------
# Benchmark helper
# ----------------------------

def benchmark(func, *args):
    tracemalloc.start()

    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return t1 - t0, peak / 1e6


# ----------------------------
# Synthetic dataset
# ----------------------------

T = 365 * 30
nlat = 180
nlon = 280
times = pd.date_range("2000-01-01", periods=T, freq="D")
mask = ~((times.month == 2) & (times.day == 29))  # no leap days
times = times[mask]

np_data = np.random.randn(T, nlat, nlon).astype(np.float32)
np_data = np_data[:len(times)]  # ensure no leap days
torch_data = torch.tensor(np_data.copy(), device="cpu")

latitudes_np = np.linspace(-90, 90, nlat)
latitudes_torch = torch.tensor(latitudes_np, dtype=torch.float32)

# ----------------------------
# Run benchmarks
# ----------------------------

print("\n--- Cos Lat Weighting ---")

t, m = benchmark(cos_lat_weighting_numpy, np_data.copy(), latitudes_np)
print(f"NumPy version:  time={t:.3f}s  peak_mem={m:.1f} MB")

t, m = benchmark(cos_lat_weighting_torch_, torch_data.clone(), latitudes_torch)
print(f"PyTorch inplace: time={t:.3f}s  peak_mem={m:.1f} MB")


print("\n--- Pixelwise Standardization ---")

t, m = benchmark(pixelwise_standardize_numpy, np_data.copy())
print(f"NumPy version:  time={t:.3f}s  peak_mem={m:.1f} MB")

t, m = benchmark(pixelwise_standardize_torch_, torch_data.clone())
print(f"PyTorch inplace: time={t:.3f}s  peak_mem={m:.1f} MB")


print("\n--- Remove Seasonal Cycle ---")

t, m = benchmark(remove_seasonal_cycle365_numpy, np_data.copy(), times)
print(f"NumPy version:  time={t:.3f}s  peak_mem={m:.1f} MB")
md_index = compute_md_index_from_md(
    torch.tensor(times.month.values, dtype=torch.long),
    torch.tensor(times.day.values, dtype=torch.long)
)
t, m = benchmark(remove_seasonal_cycle365_torch, torch_data.clone(), md_index)
print(f"PyTorch inplace: time={t:.3f}s  peak_mem={m:.1f} MB")