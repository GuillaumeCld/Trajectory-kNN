import pandas as pd
import numpy as np
traj_length = 5

comparasion_file = "case_studies/results/msl/msl_trajlen5_k10_top100.csv"
dates = pd.read_csv(comparasion_file)["time"].values.astype("datetime64[ns]")[:100]
dates = pd.to_datetime(dates).normalize()
# add to dates the next 4 days to account for trajectory length
# for i in range(1, traj_length):
#     dates = np.concatenate((dates, pd.to_datetime(dates) + pd.Timedelta(days=i)))
# dates = np.unique(dates)
# print(len(dates))
# # =========================
# Storm databases
# =========================
climk_winds = pd.read_csv("Data/Extremes/CLIMK–WINDS.csv") 
# format : 
# Dates,Names,ranks
# 20070118,KYRILL,1.0

xws_large_storms = pd.read_csv("Data/Extremes/XWS_large_storms.csv")
# format :
# Date
# 2/11/1981

xws_insurance_storms = pd.read_csv("Data/Extremes/XWS_insurance_storms.csv")
# format :
#Name,Date
#Anatol,3/12/1999

# standardize date formats
climk_winds["Dates"] = pd.to_datetime(climk_winds["Dates"],format='%Y%m%d').dt.normalize()
xws_large_storms["Dates"] = pd.to_datetime(xws_large_storms["Date"], dayfirst=True).dt.normalize()
xws_insurance_storms["Dates"] = pd.to_datetime(xws_insurance_storms["Date"], dayfirst=True).dt.normalize()
# normalize to midnight

# =========================
# Matching
# =========================
def count_matches(storm_dates, comparasion_dates):
    matches = set(storm_dates).intersection(set(comparasion_dates))
    return len(matches), matches

climk_matches_count, climk_matches = count_matches(climk_winds["Dates"], dates)
xws_large_matches_count, xws_large_matches = count_matches(xws_large_storms["Dates"], dates)
xws_insurance_matches_count, xws_insurance_matches = count_matches(xws_insurance_storms["Dates"], dates)

# =========================
# Report results
# =========================
print(f"CLIMK-WINDS matches: {climk_matches_count} / {len(climk_winds)}")
print(f"XWS Large Storms matches: {xws_large_matches_count} / {len(xws_large_storms)}")
print(f"XWS Insurance Storms matches: {xws_insurance_matches_count} / {len(xws_insurance_storms)}")

print("\nMatched CLIMK-WINDS storm dates:")
for date in climk_matches:
    print(date)

print("\nMatched XWS Large Storms dates:")
for date in xws_large_matches:
    print(date)

print("\nMatched XWS Insurance Storms dates:")
for date in xws_insurance_matches:
    print(date)