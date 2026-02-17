#!/bin/bash

# ==========================================
# Common parameters
# ==========================================

TRAJ_LENGTH=5
K=10
PARAMETER="msl"

FILE_PATH="Data/era5_msl_daily_eu.nc"
DEVICE="cuda"
ALGORITHM="base"
OUT_PATH="case_studies/results/${PARAMETER}/cluster_analysis_trajlen${TRAJ_LENGTH}_k${K}/"
mkdir -p "$OUT_PATH"

SCALING_FACTOR=100.0
SCALING_CONSTANT=0.0

# Preprocessing toggles (true / false)
REMOVE_LEAP=true
REMOVE_SEASONAL_CYCLE=true
COS_LAT_WEIGHTING=true
PIXELWISE_STANDARDIZATION=true
USE_PCA=ftrue


# ==========================================
# Derived paths
# ==========================================

RESULTS_FILE="case_studies/results/${PARAMETER}/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100.csv"
DATE_PATH="case_studies/results/${PARAMETER}/${PARAMETER}_trajlen${TRAJ_LENGTH}_k${K}_top100.csv"


# ==========================================
# Build clustering flags
# ==========================================

CLUSTER_FLAGS=""

if [ "$REMOVE_LEAP" = false ]; then
    CLUSTER_FLAGS="$CLUSTER_FLAGS --no_remove_leap"
fi

if [ "$REMOVE_SEASONAL_CYCLE" = false ]; then
    CLUSTER_FLAGS="$CLUSTER_FLAGS --no_remove_seasonal_cycle"
fi

if [ "$COS_LAT_WEIGHTING" = false ]; then
    CLUSTER_FLAGS="$CLUSTER_FLAGS --no_cos_lat_weighting"
fi

if [ "$PIXELWISE_STANDARDIZATION" = false ]; then
    CLUSTER_FLAGS="$CLUSTER_FLAGS --no_pixelwise_standardization"
fi

if [ "$USE_PCA" = false ]; then
    CLUSTER_FLAGS="$CLUSTER_FLAGS --no_pca"
fi


# ==========================================
# Run scoring
# ==========================================
python case_studies/score.py \
--traj_length "$TRAJ_LENGTH" \
--k "$K" \
--parameter "$PARAMETER" \
--file_path "$FILE_PATH" \
--device "$DEVICE" \
--algorithm "$ALGORITHM" \

# ==========================================
# Clustering analysis
# ==========================================
python case_studies/cluster_analysis.py \
--file_path "$FILE_PATH" \
--date_path "$DATE_PATH" \
--parameter "$PARAMETER" \
--out_path "$OUT_PATH" \
--scaling_factor "$SCALING_FACTOR" \
--scaling_constant "$SCALING_CONSTANT" \
$CLUSTER_FLAGS