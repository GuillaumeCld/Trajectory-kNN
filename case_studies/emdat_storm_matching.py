"""
Example usage:
python emdat_temperature_matching.py \
    --comparison_file case_studies/results/t2m/t2m_trajlen7_k20_top100.csv \
    --top_n 50 
"""

import argparse
import pandas as pd
import numpy as np


# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Match top rarity score dates with EMDAT extreme temperature events."
    )

    parser.add_argument("--comparison_file", type=str,
                        default="case_studies/results/msl/msl_trajlen10_k10_top100.csv",
                        help="CSV file containing top ranked dates")

    parser.add_argument("--emdat_file", type=str,
                        default="Data/Extremes/emdat_Storm.csv",
                        help="EMDAT extreme temperature events CSV file")

    parser.add_argument("--top_n", type=int, default=100,
                        help="Number of top entries to use from the top 100 file (default: 100)")


    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()


    # -------------------------
    # Load top N scores
    # -------------------------
    df_top = pd.read_csv(args.comparison_file)
    df_top["time"] = pd.to_datetime(df_top["time"]).dt.normalize()
    df_top = df_top.sort_values("time").iloc[:args.top_n]

    # -------------------------
    # Load EMDAT data
    # -------------------------
    df_emdat = pd.read_csv(args.emdat_file)

    # Convert date columns and drop invalid rows
    df_emdat["Start Date"] = pd.to_datetime(df_emdat["Start Date"], errors="coerce")
    df_emdat["End Date"] = pd.to_datetime(df_emdat["End Date"], errors="coerce")
    df_emdat = df_emdat.dropna(subset=["Start Date", "End Date"])

    # Get unique subtypes
    sub_types = df_emdat["Disaster Subtype"].unique()

    # -------------------------
    # Group overlapping events per subtype
    # -------------------------
    grouped_events = []

    for subtype in sub_types:
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
            elif start <= current_end:  # overlap
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

    # -------------------------
    # Matching top dates with grouped events
    # -------------------------
    matches = []

    for date in df_top["time"]:
        mask = (
            (grouped_events_df["Start Date"] <= date) &
            (grouped_events_df["End Date"] >= date)
        )
        matched_rows = grouped_events_df.loc[mask]

        if not matched_rows.empty:
            subtype = matched_rows.iloc[0]["Disaster Subtype"]
            matches.append((date, subtype))

    matches_df = pd.DataFrame(matches, columns=["date", "subtype"])
    matches_df = matches_df.sort_values(["subtype", "date"])

    # -------------------------
    # Output results
    # -------------------------
    print(f"Found {len(matches_df)} matches between top scores and EMDAT extreme temperature events.")

    for subtype in sub_types:
        subtype_dates = matches_df.loc[matches_df["subtype"] == subtype, "date"].tolist()
        print(f"\n{subtype}: {len(subtype_dates)} matches")
        for date in subtype_dates:
            print(f"  - {date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
