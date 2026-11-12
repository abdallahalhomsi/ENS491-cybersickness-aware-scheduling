import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/questset_movement_features.csv")
OUTPUT = Path("data/processed/questset_ml_dataset.csv")

df = pd.read_csv(INPUT)

# Features that are too sensitive to isolated spikes.
drop_cols = [
    "head_velocity_max",
    "head_acceleration_max",
    "head_jerk_max",
    "head_angular_velocity_max"
]

df = df.drop(columns=drop_cols)

# Sanity checks
print("Rows:", len(df))
print("Participants:", df["participant_id"].nunique())

print("\nRows per window:")
print(df["window"].value_counts())

print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nTargets:")
print(
    df[
        ["pre_ssq_total", "post_ssq_total", "delta_ssq_total"]
    ].describe()
)

df.to_csv(OUTPUT, index=False)

print(f"\nColumns: {len(df.columns)}")
print(f"Saved: {OUTPUT}")