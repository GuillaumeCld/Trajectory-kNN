"""
Example usage:

python storm_matching.py \
    --comparison_file case_studies/results/msl/msl_trajlen5_k10_top100.csv \
    --top_n 50 \
    --use_climk
"""

import argparse
import pandas as pd
import numpy as np
import random




# =========================
# Matching function
# =========================
def count_matches(storm_dates, comparison_dates):
    matches = set(storm_dates).intersection(set(comparison_dates))
    return len(matches), matches


# =========================
# Main
# =========================
def main():


    # Default file paths
    CLIMK_FILE = "Data/Extremes/CLIMK–WINDS.csv"
    XWS_LARGE_FILE = "Data/Extremes/XWS_large_storms.csv"
    XWS_INSURANCE_FILE = "Data/Extremes/XWS_insurance_storms.csv"
    # =========================
    # EM-DAT
    # =========================
    EMDAT_FILE = "Data/Extremes/emdat_Storm.csv"

    df_emdat = pd.read_csv(EMDAT_FILE)

    df_emdat["Start Date"] = pd.to_datetime(df_emdat["Start Date"], errors="coerce")
    df_emdat["End Date"] = pd.to_datetime(df_emdat["End Date"], errors="coerce")
    df_emdat = df_emdat.dropna(subset=["Start Date", "End Date"])

    # Group overlapping events per subtype
    grouped_events = []

    for subtype in df_emdat["Disaster Subtype"].unique():
        subtype_events = (
            df_emdat[df_emdat["Disaster Subtype"] == subtype]
            .sort_values("Start Date")
        )

        current_start = None
        current_end = None

        for _, row in subtype_events.iterrows():
            start = row["Start Date"]
            end = row["End Date"]

            if current_start is None:
                current_start = start
                current_end = end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                grouped_events.append((current_start, current_end, subtype))
                current_start = start
                current_end = end

        if current_start is not None:
            grouped_events.append((current_start, current_end, subtype))

    grouped_events_df = pd.DataFrame(
        grouped_events,
        columns=["Start Date", "End Date", "Disaster Subtype"]
    )
    # Unique subtypes
    sub_types = grouped_events_df["Disaster Subtype"].unique()

    # Dictionary to store total matches per subtype
    count_emdat_per_subtype = {subtype: 0 for subtype in sub_types}

    count_climk = 0
    count_large = 0
    count_inasurance = 0

    for seed in range(100):
        np.random.seed(seed)
        random.seed(seed)

        # =========================
        # CLIMK-WINDS
        # =========================
        times = pd.date_range(start="1995-01-01", end="2015-12-31", freq="D")
        dates = np.random.choice(times, size=100, replace=False)

        climk_winds = pd.read_csv(CLIMK_FILE)
        climk_winds["Dates"] = pd.to_datetime(
            climk_winds["Dates"], format='%Y%m%d'
        ).dt.normalize()

        count, matches = count_matches(climk_winds["Dates"], dates)
        count_climk += count


        # =========================
        # XWS Large Storms
        # =========================
        times = pd.date_range(start="1974-01-01", end="2013-12-31", freq="D")
        dates = np.random.choice(times, size=100, replace=False)
        
        xws_large = pd.read_csv(XWS_LARGE_FILE)
        xws_large["Dates"] = pd.to_datetime(
            xws_large["Date"], dayfirst=True
        ).dt.normalize()

        count, matches = count_matches(xws_large["Dates"], dates)
        count_large += count

        xws_insurance = pd.read_csv(XWS_INSURANCE_FILE)
        xws_insurance["Dates"] = pd.to_datetime(
            xws_insurance["Date"], dayfirst=True
        ).dt.normalize()

        count, matches = count_matches(xws_insurance["Dates"], dates)
        count_inasurance += count


        # =========================
        # EM-DAT
        # =========================
        times = pd.date_range(start="1950-01-01", end="2025-12-31", freq="D")
        dates = np.random.choice(times, size=100, replace=False)

        for date in dates:
            mask = (
                (grouped_events_df["Start Date"] <= date) &
                (grouped_events_df["End Date"] >= date)
            )

            matched_rows = grouped_events_df.loc[mask]

            if not matched_rows.empty:
                # In case overlapping subtypes exist (rare but possible)
                for subtype in matched_rows["Disaster Subtype"].unique():
                    count_emdat_per_subtype[subtype] += 1
        
    print(f"\nAverage CLIMK-WINDS matches over 100 runs: {count_climk / 100:.1f}")
    print(f"Average XWS Large Storms matches over 100 runs: {count_large / 100:.1f}")
    print(f"Average XWS Insurance Storms matches over 100 runs: {count_inasurance / 100:.1f}")
    print("\nAverage EM-DAT matches per subtype over 100 runs:")

    for subtype, total in count_emdat_per_subtype.items():
        print(f"{subtype}: {total / 100:.2f}")

if __name__ == "__main__":
    main()
