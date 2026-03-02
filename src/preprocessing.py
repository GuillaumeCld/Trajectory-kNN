import pandas as pd
import numpy as np

def remove_bisex_dailydata(data, time):

    time = pd.to_datetime(time)
    mask = ~((time.month == 2) & (time.day == 29))
    return data[mask], time[mask]

def remove_seasonal_cycle365(data, time):
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


def cos_lat_weighting(data, latitudes, nlon):
    latitudes = np.asarray(latitudes)
    weights = np.sqrt(np.cos(np.deg2rad(latitudes)))[:, np.newaxis]
    return data * weights


def pixelwise_standardize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return (data - mean) / std


