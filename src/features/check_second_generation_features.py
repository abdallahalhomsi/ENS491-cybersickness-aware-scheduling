import pandas as pd
import numpy as np


PATH = "data/processed/questset_second_generation_features.csv"

df = pd.read_csv(PATH)

print("=" * 70)
print("SECOND-GENERATION FEATURE VALIDATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nParticipants:")
print(df["participant_id"].nunique())

print("\nRows per window:")
print(df["window"].value_counts().sort_index())


# ============================================================
# Duplicate checks
# ============================================================

duplicate_rows = df.duplicated(
    subset=[
        "participant_id",
        "session_number",
        "window"
    ]
).sum()

print("\nDuplicate session/window rows:")
print(duplicate_rows)


# ============================================================
# Target consistency across horizons
# ============================================================

target_consistency = (
    df.groupby(
        [
            "participant_id",
            "session_number"
        ]
    )[
        [
            "post_ssq_total",
            "delta_ssq_total",
            "pre_ssq_total"
        ]
    ]
    .nunique()
)

print("\nSessions with inconsistent targets:")
print(
    (
        target_consistency > 1
    ).any(axis=1).sum()
)


# ============================================================
# Feature availability by horizon
# ============================================================

for window in ["1min", "3min", "5min"]:

    subset = df[
        df["window"] == window
    ]

    available = (
        subset
        .notna()
        .any()
    )

    available_cols = available[
        available
    ].index.tolist()

    print(
        f"\n{window}: "
        f"{len(available_cols)} available columns"
    )


# ============================================================
# Leakage checks
# ============================================================

one_min = df[
    df["window"] == "1min"
]

three_min = df[
    df["window"] == "3min"
]

five_min = df[
    df["window"] == "5min"
]


one_min_future = [
    c for c in df.columns
    if (
        c.startswith("3min_")
        or c.startswith("5min_")
    )
]

three_min_future = [
    c for c in df.columns
    if c.startswith("5min_")
]


print("\n1-minute rows containing future features:")

if one_min_future:
    print(
        one_min[
            one_min_future
        ]
        .notna()
        .sum()
        .sum()
    )
else:
    print(0)


print("\n3-minute rows containing 5-minute features:")

if three_min_future:
    print(
        three_min[
            three_min_future
        ]
        .notna()
        .sum()
        .sum()
    )
else:
    print(0)


# ============================================================
# Required cumulative features
# ============================================================

checks = {
    "1min has 1min velocity":
        one_min[
            "1min_velocity_mean"
        ].notna().all(),

    "3min has 1min velocity":
        three_min[
            "1min_velocity_mean"
        ].notna().all(),

    "3min has 3min velocity":
        three_min[
            "3min_velocity_mean"
        ].notna().all(),

    "5min has 1min velocity":
        five_min[
            "1min_velocity_mean"
        ].notna().all(),

    "5min has 3min velocity":
        five_min[
            "3min_velocity_mean"
        ].notna().all(),

    "5min has 5min velocity":
        five_min[
            "5min_velocity_mean"
        ].notna().all(),

    "5min has 3min temporal slope":
        five_min[
            "3min_velocity_mean_slope"
        ].notna().all(),

    "5min has 5min temporal slope":
        five_min[
            "5min_velocity_mean_slope"
        ].notna().all()
}

print("\nCumulative feature checks:")

for name, result in checks.items():
    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )


# ============================================================
# Infinite values
# ============================================================

numeric = df.select_dtypes(
    include=[np.number]
)

inf_count = np.isinf(
    numeric.to_numpy()
).sum()

print("\nInfinite numeric values:")
print(inf_count)


# ============================================================
# Constant numeric columns within each horizon
# ============================================================

print("\nConstant numeric feature columns by horizon:")

ignore = {
    "session_number",
    "group",
    "order",
    "user",
    "window_seconds",
    "pre_ssq_total",
    "post_ssq_total",
    "delta_ssq_total"
}

for window in ["1min", "3min", "5min"]:

    subset = df[
        df["window"] == window
    ]

    numeric_cols = (
        subset
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    constants = []

    for col in numeric_cols:

        if col in ignore:
            continue

        values = (
            subset[col]
            .dropna()
        )

        if (
            len(values) > 0
            and values.nunique() <= 1
        ):
            constants.append(col)

    print(
        f"{window}: "
        f"{len(constants)}"
    )

    if constants:
        print(constants)


print("\nValidation complete.")