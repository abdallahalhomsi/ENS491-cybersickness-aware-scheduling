import pandas as pd

df = pd.read_csv("data/processed/questset_movement_features.csv")

check_cols = [
    "head_total_distance",
    "head_net_displacement",
    "head_velocity_mean",
    "head_velocity_max",
    "head_acceleration_mean",
    "head_acceleration_max",
    "head_jerk_mean",
    "head_jerk_max",
    "head_total_rotation_rad",
    "head_angular_velocity_mean",
    "head_angular_velocity_max",
    "effective_hz"
]

print("Shape:", df.shape)

for col in check_cols:
    print("\n" + "=" * 50)
    print(col)
    print(df[col].describe())

print("\nTop 10 velocity maxima:")
print(
    df[
        [
            "participant_id",
            "game_name",
            "window",
            "head_velocity_max"
        ]
    ]
    .sort_values("head_velocity_max", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nTop 10 acceleration maxima:")
print(
    df[
        [
            "participant_id",
            "game_name",
            "window",
            "head_acceleration_max"
        ]
    ]
    .sort_values("head_acceleration_max", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nSampling frequency:")
print("Min:", df["effective_hz"].min())
print("Median:", df["effective_hz"].median())
print("Max:", df["effective_hz"].max())