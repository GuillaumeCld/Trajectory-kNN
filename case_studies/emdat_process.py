import pandas as pd

file_path = "Data/Extremes/public_emdat_custom_request_2026-02-13_fdb19ded-51c4-4f3d-a3be-c315d8f13ad0.xlsx"

columns_needed = [
    "Disaster Type",
    "Disaster Subtype",
    "Country",
    "Subregion",
    "Location",
    "Start Year", "Start Month", "Start Day",
    "End Year", "End Month", "End Day",
]

# Read without forcing dtype
df = pd.read_excel(file_path, usecols=columns_needed)

date_cols = [
    "Start Year", "Start Month", "Start Day",
    "End Year", "End Month", "End Day"
]

# Convert safely to numeric (keeps NaN)
df[date_cols] = df[date_cols].apply(pd.to_numeric, errors="coerce")

# Drop rows missing required start date components
df = df.dropna(subset=["Start Year", "Start Month", "Start Day"])

# Convert ONLY the valid start columns to int
df[["Start Year", "Start Month", "Start Day"]] = (
    df[["Start Year", "Start Month", "Start Day"]]
    .astype(int)
)

# Build Start Date
df["Start Date"] = pd.to_datetime(
    dict(
        year=df["Start Year"],
        month=df["Start Month"],
        day=df["Start Day"],
    ),
    errors="coerce"
)

# Build End Date
df["End Date"] = pd.to_datetime(
    dict(
        year=df["End Year"],
        month=df["End Month"],
        day=df["End Day"],
    ),
    errors="coerce"
)

# If End Date invalid set equal to Start Date
df["End Date"] = df["End Date"].fillna(df["Start Date"])

df = df.dropna(subset=["Start Date"])
df.drop(columns=date_cols, inplace=True)

print(df.head())


disaster_types = df["Disaster Type"].unique()

for dtype in disaster_types:
    subset = df[df["Disaster Type"] == dtype]
    # reorder columns
    subset = subset[["Disaster Type", "Disaster Subtype", "Start Date", "End Date", "Country", "Subregion", "Location"]]
    subset.to_csv(f"case_studies/emdat_{dtype.replace(' ', '_')}.csv", index=False)