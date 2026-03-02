import pandas as pd
import numpy as np
from scipy.stats import spearmanr

file_base = "case_studies/results/msl/msl_trajlen5_k10_top100.csv"
file_pca = "case_studies/results/msl/msl_trajlen5_k10_top100_pca.csv"

df_base = pd.read_csv(file_base)
df_pca = pd.read_csv(file_pca)
print(df_base.head())

dates_base = pd.to_datetime(df_base["time"]).dt.normalize()
dates_pca = pd.to_datetime(df_pca["time"]).dt.normalize()

df_base["time"] = dates_base
df_pca["time"] = dates_pca

# Compute Jaccard similarity
set_base = set(dates_base)
set_pca = set(dates_pca)
intersection = set_base.intersection(set_pca)
union = set_base.union(set_pca)
jaccard_similarity = len(intersection) / len(union)
print(f"Jaccard similarity between top 100 dates: {jaccard_similarity:.4f}")
print(f"Number of common dates: {len(intersection)} out of {len(union)} unique dates.")

# Set index to "time" column for both DataFrames (removed erroneous pd.to_datetime on integer index)
df_base = df_base.set_index("time")
df_pca = df_pca.set_index("time")

# Ensure intersection is a list of datetime objects
common_dates = pd.to_datetime(list(intersection))
print(f"Common dates: {common_dates}")
print(f"df_base: {df_base}")

# Align both DataFrames on common dates
df_base_aligned = df_base.loc[common_dates]
df_pca_aligned = df_pca.loc[common_dates]

# Sort both by index to ensure consistent ordering
df_base_aligned = df_base_aligned.sort_index()
df_pca_aligned = df_pca_aligned.sort_index()

print(df_base_aligned.head())

# Replace 'value_column' with the actual column name in your CSV
value_col = df_base_aligned.columns[0]  # adjust if needed

rank_base = df_base_aligned[value_col].rank()
rank_pca = df_pca_aligned[value_col].rank()

correlation, p_value = spearmanr(rank_base, rank_pca)
print(f"Spearman rank correlation: {correlation:.4f} (p-value: {p_value:.4e})")
