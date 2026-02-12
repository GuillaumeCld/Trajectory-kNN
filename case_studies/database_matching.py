import numpy as np
import pandas as pd
from scipy.io import loadmat
 
# =========================
# Parameters
# =========================
file = "case_studies/results/msl/msl_trajlen5_k30_top100.csv"

# =========================
# Dates (remove Feb 29 automatically)
# =========================
dates = pd.date_range("1950-01-01", pd.Timestamp.today(), freq="D")
dates = dates[~((dates.month == 2) & (dates.day == 29))]

dates_format = dates.year * 10000 + dates.month * 100 + dates.day
dates_format = dates_format.to_numpy()

month_vector = dates.month.to_numpy()
year_vector = dates.year.to_numpy()

# =========================
# Load storm database
# =========================
mat = loadmat("Articolo_Analogues_traj/EU-WindstormWikipedia.mat")
StormDatabase1 = mat["StormDatabase1"].ravel()



# =========================
# Process anomalous date files
# =========================

N_Storm_detected = []

dates_extr = pd.read_csv(file)["time"].values.astype("datetime64[ns]")
dates_extr = pd.to_datetime(dates_extr).normalize()
dates_extr_format = dates_extr.year * 10000 + dates_extr.month * 100 + dates_extr.day



detected_dates = np.intersect1d(StormDatabase1, dates_extr_format)
N_Storm_detected.append(len(detected_dates))


mask = np.isin(StormDatabase1, detected_dates)

print("N_Storm_detected:", N_Storm_detected)

print("Detected storm dates:")
detected_dates_dt = pd.to_datetime(StormDatabase1[mask].astype(str), format='%Y%m%d')
print("\nDetected storm dates (datetime):")
print(detected_dates_dt)


