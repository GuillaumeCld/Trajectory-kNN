#!/bin/bash

# Common parameters
TRAJ_LENGTH=1
K=10
PARAMETER="msl"
FILE_PATH="Data/era5_msl_daily_eu.nc"
# FILE_PATH="Data/t2m_daily_avg_1950_2023.nc"
DEVICE="cuda"
ALGORITHM="base"

# Script 1: Score
python case_studies/score.py \
    --traj_length "$TRAJ_LENGTH" \
    --k "$K" \
    --parameter "$PARAMETER" \
    --file_path "$FILE_PATH" \
    --device "$DEVICE" \
    --algorithm "$ALGORITHM" \



python case_studies/emdat_storm_matching.py \
    --comparison_file "case_studies/results/$PARAMETER/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100_relative.csv" \
    --top_n 100