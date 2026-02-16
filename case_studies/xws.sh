#!/bin/bash

# Common parameters
TRAJ_LENGTH=7
K=10
PARAMETER="msl"
FILE_PATH="Data/era5_msl_daily_eu.nc"
DEVICE="cuda"
ALGORITHM="base"
LON_MIN=-15
LON_MAX=25
LAT_MIN=35
LAT_MAX=70
START_YEAR=1979
END_YEAR=2013

# Script 1: Score
python case_studies/score.py \
    --traj_length "$TRAJ_LENGTH" \
    --k "$K" \
    --parameter "$PARAMETER" \
    --file_path "$FILE_PATH" \
    --device "$DEVICE" \
    --algorithm "$ALGORITHM" \
    --lon_min "$LON_MIN" \
    --lon_max "$LON_MAX" \
    --lat_min "$LAT_MIN" \
    --lat_max "$LAT_MAX" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR"

# Script 2: Storm matching
python case_studies/storm_matching.py \
    --comparison_file "case_studies/results/$PARAMETER/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100.csv" \
    --traj_length "$TRAJ_LENGTH" \
    --top_n 100 \
    --use_xws_large \
    --use_xws_insurance

# Script 3: EMDAT temperature matching
python case_studies/emdat_temperature_matching.py \
    --comparison_file "case_studies/results/$PARAMETER/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100.csv" \
    --top_n 100