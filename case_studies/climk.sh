#!/bin/bash

# ==========================================
# Common parameters
# ==========================================

TRAJ_LENGTH=7
K=10
PARAMETER="t2m"

FILE_PATH="Data/t2m_daily_avg_1950_2023.nc"
DATE_PATH="case_studies/results/${PARAMETER}/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100_relative_interval.csv"

OUT_PATH="case_studies/results/${PARAMETER}/"

SCALING_FACTOR=1.0
SCALING_CONSTANT=273.15

# Preprocessing toggles (true / false)
REMOVE_LEAP=true
REMOVE_SEASONAL_CYCLE=true
COS_LAT_WEIGHTING=true
PIXELWISE_STANDARDIZATION=true
USE_PCA=true


# ==========================================
# Build optional flags
# ==========================================

FLAGS=""

if [ "$REMOVE_LEAP" = false ]; then
    FLAGS="$FLAGS --no_remove_leap"
fi

if [ "$REMOVE_SEASONAL_CYCLE" = false ]; then
    FLAGS="$FLAGS --no_remove_seasonal_cycle"
fi

if [ "$COS_LAT_WEIGHTING" = false ]; then
    FLAGS="$FLAGS --no_cos_lat_weighting"
fi

if [ "$PIXELWISE_STANDARDIZATION" = false ]; then
    FLAGS="$FLAGS --no_pixelwise_standardization"
fi

if [ "$USE_PCA" = false ]; then
    FLAGS="$FLAGS --no_pca"
fi


python case_studies/cluster_analysis.py \
--file_path "$FILE_PATH" \
--date_path "$DATE_PATH" \
--parameter "$PARAMETER" \
--out_path "$OUT_PATH" \
--scaling_factor "$SCALING_FACTOR" \
--scaling_constant "$SCALING_CONSTANT" \
$FLAGS

