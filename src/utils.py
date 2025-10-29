import numpy as np
import pandas as pd
import xarray as xr

def remove_daily_climatology(data, time):
    """
    Remove the average per day of year from the data, keep the spatial dimension.
    Parameters:
    - data: np.ndarray of shape (T, H, W)
    - time: np.ndarray of shape (T,) with datetime values
    Returns:
    - data: np.ndarray of shape (T, H, W)
    """
    
    day_of_year = pd.to_datetime(time).dayofyear
    daily_climatology = np.array([data[day_of_year == doy].mean(axis=0) for doy in range(1, 367)])
    data = data - daily_climatology[day_of_year - 1]

    return data


def save_subset(nc_path):
    ds = xr.open_dataset(nc_path)
    n_time = 365 * 5
    ds = ds.isel(time=slice(0, n_time))
    ds.to_netcdf(f"Data/era5_msl_daily_eu_small.nc")
