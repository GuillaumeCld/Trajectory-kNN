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


# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Match top rarity score dates with storm databases."
    )

    parser.add_argument("--traj_length", type=int, default=5,
                        help="Trajectory length (default: 5)")

    parser.add_argument("--comparison_file", type=str,
                        default="case_studies/results/msl/msl_trajlen5_k10_top100.csv",
                        help="CSV file containing top ranked dates")

    parser.add_argument("--top_n", type=int, default=100,
                        help="Number of top entries to use (default: 100)")

    # Boolean flags
    parser.add_argument("--use_climk", action="store_true",
                        help="Use CLIMK-WINDS database")

    parser.add_argument("--use_xws_large", action="store_true",
                        help="Use XWS Large Storms database")

    parser.add_argument("--use_xws_insurance", action="store_true",
                        help="Use XWS Insurance Storms database")

    return parser.parse_args()


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
    args = parse_args()

    traj_length = args.traj_length
    comparison_file = args.comparison_file
    top_n = args.top_n

    # Default file paths
    CLIMK_FILE = "Data/Extremes/CLIMK–WINDS.csv"
    XWS_LARGE_FILE = "Data/Extremes/XWS_large_storms.csv"
    XWS_INSURANCE_FILE = "Data/Extremes/XWS_insurance_storms.csv"

    # =========================
    # Load comparison dates
    # =========================
    dates = pd.read_csv(comparison_file)[
        "time"].values.astype("datetime64[ns]")
    dates = pd.to_datetime(dates).normalize()[:top_n]

    # Expand dates to account for trajectory length
    expanded_dates = dates.copy()
    for i in range(1, traj_length):
        expanded_dates = np.concatenate(
            (expanded_dates,
             (dates + pd.Timedelta(days=i)).values)
        )

    dates = np.unique(pd.to_datetime(expanded_dates).normalize())

    # =========================
    # Matching
    # =========================
    if args.use_climk:
        climk_winds = pd.read_csv(CLIMK_FILE)
        climk_winds["Dates"] = pd.to_datetime(
            climk_winds["Dates"], format='%Y%m%d'
        ).dt.normalize()

        count, matches = count_matches(climk_winds["Dates"], dates)

        print(f"\nCLIMK-WINDS matches: {count} / {len(climk_winds)}")
        for date in sorted(matches):
            print(date)

    if args.use_xws_large:
        xws_large = pd.read_csv(XWS_LARGE_FILE)
        xws_large["Dates"] = pd.to_datetime(
            xws_large["Date"], dayfirst=True
        ).dt.normalize()

        count, matches = count_matches(xws_large["Dates"], dates)

        print(f"\nXWS Large Storms matches: {count} / {len(xws_large)}")
        for date in sorted(matches):
            print(date)

    if args.use_xws_insurance:
        xws_insurance = pd.read_csv(XWS_INSURANCE_FILE)
        xws_insurance["Dates"] = pd.to_datetime(
            xws_insurance["Date"], dayfirst=True
        ).dt.normalize()

        count, matches = count_matches(xws_insurance["Dates"], dates)

        print(
            f"\nXWS Insurance Storms matches: {count} / {len(xws_insurance)}")
        for date in sorted(matches):
            print(date)

    # If no database selected
    if not (args.use_climk or args.use_xws_large or args.use_xws_insurance):
        print("No storm database selected. Use flags:")
        print("--use_climk --use_xws_large --use_xws_insurance")


if __name__ == "__main__":
    main()
