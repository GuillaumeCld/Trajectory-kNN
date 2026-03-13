import pandas as pd
import numpy as np
import torch
def remove_bisex_dailydata(data, time):

    time = pd.to_datetime(time)
    mask = ~((time.month == 2) & (time.day == 29))
    return data[mask], time[mask]


def cos_lat_weighting(data, latitudes, nlon):
    data = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
    latitudes = torch.from_numpy(latitudes) if not isinstance(latitudes, torch.Tensor) else latitudes
    weights = torch.sqrt(torch.cos(torch.deg2rad(latitudes))).view(1, -1, 1)
    data.mul_(weights)
    return data


def pixelwise_standardize(data):
    data = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
    mean = data.mean(dim=0)
    std = data.std(dim=0).clamp_min_(1e-12)
    data.sub_(mean)
    data.div_(std)
    return data

def compute_md_index_from_md(month, day):
    month_cumsum = torch.tensor(
        [0,31,59,90,120,151,181,212,243,273,304,334],
        dtype=torch.long,
        device=month.device
    )
    return month_cumsum[month - 1] + (day - 1)


def remove_seasonal_cycle365(data: torch.Tensor, times):
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
    md_index = compute_md_index_from_md(
    torch.tensor(times.month.values, dtype=torch.long),
    torch.tensor(times.day.values, dtype=torch.long))

    data = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
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

# import torch

# def cos_lat_weighting_(data: torch.Tensor, latitudes: torch.Tensor):
#     """
#     Apply sqrt(cos(lat)) weighting in-place.

#     data: (T, nlat, nlon)
#     latitudes: (nlat,) in degrees
#     """

#     weights = torch.sqrt(torch.cos(torch.deg2rad(latitudes)))
#     weights = weights.view(1, -1, 1)  # broadcast over time and longitude

#     data.mul_(weights)
#     return data


# def remove_seasonal_cycle365(data: torch.Tensor, md_index: torch.Tensor):
#     """
#     Remove the 365-day seasonal cycle from data in-place.

#     Parameters
#     ----------
#     data : torch.Tensor
#         Shape (T, ...), where T is time.
#     md_index : torch.Tensor
#         Long tensor of shape (T,) containing day-of-year indices in [0, 364].
#         Leap days must already be removed.

#     Returns
#     -------
#     data : torch.Tensor
#         Deseasonalized tensor (modified in-place).
#     """

#     T = data.shape[0]
#     device = data.device
#     dtype = data.dtype

#     flat = data.view(T, -1)
#     seasonal = torch.zeros(365, flat.shape[1], dtype=dtype, device=device)
#     counts = torch.zeros(365, dtype=dtype, device=device)

#     # accumulate sums
#     seasonal.index_add_(0, md_index, flat)

#     # accumulate counts
#     ones = torch.ones(T, dtype=dtype, device=device)
#     counts.index_add_(0, md_index, ones)

#     # compute means
#     seasonal /= counts.clamp_min(1).unsqueeze(1)

#     # subtract seasonal cycle in-place
#     flat.sub_(seasonal.index_select(0, md_index))

#     return data

# def pixelwise_standardize_blockwise_(data, block=1024):
#     mean = data.mean(dim=0)
#     std = data.std(dim=0).clamp_min_(1e-12)

#     T = data.shape[0]

#     for i in range(0, T, block):
#         x = data[i:i+block]
#         x.sub_(mean)
#         x.div_(std)

#     return data