import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import linregress


# ============================================================
# Paths
# ============================================================

RAW_MOVEMENT_DIR = Path("data/raw/Questset/movement")
SESSIONS_PATH = Path("data/processed/questset_sessions_with_ssq.csv")

OUTPUT_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)


# ============================================================
# Configuration
# ============================================================

RESAMPLE_HZ = 50
TRACKING_JUMP_THRESHOLD = 0.10

WINDOWS = {
    "1min": 60,
    "3min": 180,
    "5min": 300
}

BLOCK_STARTS = [0, 60, 120, 180, 240]


# ============================================================
# Safe statistics
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.mean(x)


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.std(x)


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.percentile(x, q)


# ============================================================
# Resample movement data to uniform 50 Hz
# ============================================================

def resample_movement(df):

    df = (
        df
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    start = df["time"].iloc[0]
    end = df["time"].iloc[-1]

    step = 1.0 / RESAMPLE_HZ

    new_time = np.arange(
        start,
        end,
        step
    )

    columns = [
        "HeadPosX",
        "HeadPosY",
        "HeadPosZ",
        "HeadOrientationW",
        "HeadOrientationX",
        "HeadOrientationY",
        "HeadOrientationZ"
    ]

    data = {
        "time": new_time
    }

    old_time = df["time"].to_numpy()

    for col in columns:

        data[col] = np.interp(
            new_time,
            old_time,
            df[col].to_numpy()
        )

    out = pd.DataFrame(data)

    # --------------------------------------------------------
    # Renormalize quaternion values after interpolation
    # --------------------------------------------------------

    qcols = [
        "HeadOrientationW",
        "HeadOrientationX",
        "HeadOrientationY",
        "HeadOrientationZ"
    ]

    q = out[qcols].to_numpy(dtype=float)

    norms = np.linalg.norm(
        q,
        axis=1
    )

    valid = norms > 0

    q[valid] = (
        q[valid]
        / norms[valid, None]
    )

    out[qcols] = q

    return out


# ============================================================
# Calculate motion signals
# ============================================================

def calculate_motion_signals(df):

    time = df["time"].to_numpy(dtype=float)

    position = df[
        [
            "HeadPosX",
            "HeadPosY",
            "HeadPosZ"
        ]
    ].to_numpy(dtype=float)

    dt = np.diff(time)

    displacement = np.diff(
        position,
        axis=0
    )

    distance_step = np.linalg.norm(
        displacement,
        axis=1
    )

    # --------------------------------------------------------
    # Filter obvious position tracking jumps
    # --------------------------------------------------------

    valid_motion = (
        (dt > 0)
        & (
            distance_step
            <= TRACKING_JUMP_THRESHOLD
        )
    )

    velocity = np.full(
        len(distance_step),
        np.nan
    )

    velocity[valid_motion] = (
        distance_step[valid_motion]
        / dt[valid_motion]
    )

    # --------------------------------------------------------
    # Quaternion rotational motion
    # --------------------------------------------------------

    quaternion = df[
        [
            "HeadOrientationW",
            "HeadOrientationX",
            "HeadOrientationY",
            "HeadOrientationZ"
        ]
    ].to_numpy(dtype=float)

    q1 = quaternion[:-1]
    q2 = quaternion[1:]

    dots = np.abs(
        np.sum(
            q1 * q2,
            axis=1
        )
    )

    dots = np.clip(
        dots,
        -1.0,
        1.0
    )

    angular_change = (
        2.0 * np.arccos(dots)
    )

    angular_velocity = np.full(
        len(angular_change),
        np.nan
    )

    valid_dt = dt > 0

    angular_velocity[valid_dt] = (
        angular_change[valid_dt]
        / dt[valid_dt]
    )

    return {
        "position":
            position,

        "distance_step":
            distance_step,

        "velocity":
            velocity,

        "angular_velocity":
            angular_velocity,

        "tracking_jump_count":
            int(
                (
                    ~valid_motion
                ).sum()
            )
    }


# ============================================================
# Base movement features
# ============================================================

def extract_base_features(df):

    signals = calculate_motion_signals(
        df
    )

    velocity = (
        signals[
            "velocity"
        ]
    )

    angular_velocity = (
        signals[
            "angular_velocity"
        ]
    )

    position = (
        signals[
            "position"
        ]
    )

    distance_step = (
        signals[
            "distance_step"
        ]
    )

    features = {
        # ----------------------------------------------------
        # Translational motion
        # ----------------------------------------------------

        "velocity_mean":
            safe_mean(
                velocity
            ),

        "velocity_std":
            safe_std(
                velocity
            ),

        "velocity_p95":
            safe_percentile(
                velocity,
                95
            ),

        # ----------------------------------------------------
        # Rotational motion
        # ----------------------------------------------------

        "angular_velocity_mean":
            safe_mean(
                angular_velocity
            ),

        "angular_velocity_std":
            safe_std(
                angular_velocity
            ),

        "angular_velocity_p95":
            safe_percentile(
                angular_velocity,
                95
            ),

        # ----------------------------------------------------
        # Spatial displacement
        # ----------------------------------------------------

        "net_displacement":
            np.linalg.norm(
                position[-1]
                - position[0]
            ),

        # ----------------------------------------------------
        # Tracking quality
        # ----------------------------------------------------

        "tracking_jump_count":
            signals[
                "tracking_jump_count"
            ],

        "tracking_jump_fraction":
            (
                signals[
                    "tracking_jump_count"
                ]
                / max(
                    len(
                        distance_step
                    ),
                    1
                )
            )
    }

    return features


# ============================================================
# Extract consecutive 60-second blocks
# ============================================================

def extract_blocks(
    df,
    max_seconds
):

    rows = []

    for block_start in BLOCK_STARTS:

        block_end = (
            block_start
            + 60
        )

        # Do not use blocks beyond the current prediction horizon
        if block_end > max_seconds:
            continue

        block = df[
            (
                df["time"]
                >= block_start
            )
            &
            (
                df["time"]
                < block_end
            )
        ].copy()

        if len(block) < 2:
            continue

        signals = (
            calculate_motion_signals(
                block
            )
        )

        rows.append(
            {
                "block_start":
                    block_start,

                "velocity_mean":
                    safe_mean(
                        signals[
                            "velocity"
                        ]
                    ),

                "velocity_std":
                    safe_std(
                        signals[
                            "velocity"
                        ]
                    ),

                "angular_velocity_mean":
                    safe_mean(
                        signals[
                            "angular_velocity"
                        ]
                    ),

                "angular_velocity_std":
                    safe_std(
                        signals[
                            "angular_velocity"
                        ]
                    )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Temporal trajectory features
# ============================================================

def extract_temporal_features(
    blocks
):

    features = {}

    # Need at least two blocks to describe temporal change
    if len(blocks) < 2:
        return features

    temporal_columns = [
        "velocity_mean",
        "velocity_std",
        "angular_velocity_mean",
        "angular_velocity_std"
    ]

    x = blocks[
        "block_start"
    ].to_numpy(dtype=float)

    for col in temporal_columns:

        y = blocks[
            col
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x_valid = x[valid]
        y_valid = y[valid]

        if len(y_valid) < 2:
            continue

        # ----------------------------------------------------
        # Linear trend
        # ----------------------------------------------------

        slope = linregress(
            x_valid,
            y_valid
        ).slope

        features[
            f"{col}_slope"
        ] = slope

        # ----------------------------------------------------
        # First block vs last available block
        # ----------------------------------------------------

        features[
            f"{col}_early_late_change"
        ] = (
            y_valid[-1]
            - y_valid[0]
        )

        # ----------------------------------------------------
        # Temporal range
        # ----------------------------------------------------

        features[
            f"{col}_temporal_range"
        ] = (
            np.max(y_valid)
            - np.min(y_valid)
        )

        # ----------------------------------------------------
        # Temporal variation
        # ----------------------------------------------------

        features[
            f"{col}_temporal_std"
        ] = np.std(
            y_valid
        )

        # ----------------------------------------------------
        # Maximum 60-second block
        # ----------------------------------------------------

        features[
            f"{col}_max_block"
        ] = np.max(
            y_valid
        )

    return features


# ============================================================
# Main dataset construction
# ============================================================

def main():

    sessions = pd.read_csv(
        SESSIONS_PATH
    )

    rows = []

    total_sessions = len(
        sessions
    )

    # ========================================================
    # Process every Questset session
    # ========================================================

    for i, session in sessions.iterrows():

        print(
            f"[{i + 1}/{total_sessions}] "
            f"{session['participant_id']} "
            f"| "
            f"{session['game_name']}"
        )

        filename = Path(
            session[
                "movement_filepath"
            ]
        ).name

        movement_path = (
            RAW_MOVEMENT_DIR
            / filename
        )

        raw = pd.read_csv(
            movement_path
        )

        # ----------------------------------------------------
        # We only study the first five minutes
        # ----------------------------------------------------

        raw = raw[
            raw["time"] <= 300
        ].copy()

        # ----------------------------------------------------
        # Uniform 50 Hz data
        # ----------------------------------------------------

        movement_df = (
            resample_movement(
                raw
            )
        )

        # ====================================================
        # Precompute base features at 1, 3, and 5 minutes
        # ====================================================

        base_features_by_window = {}

        for (
            window_name,
            seconds
        ) in WINDOWS.items():

            window_df = movement_df[
                movement_df["time"]
                <= seconds
            ].copy()

            base_features_by_window[
                window_name
            ] = extract_base_features(
                window_df
            )

        # ====================================================
        # Build prediction rows
        # ====================================================

        for (
            current_window,
            current_seconds
        ) in WINDOWS.items():

            row = {
                # ------------------------------------------------
                # Identifiers
                # ------------------------------------------------

                "participant_id":
                    session[
                        "participant_id"
                    ],

                "session_number":
                    session[
                        "session_number"
                    ],

                "group":
                    session[
                        "group"
                    ],

                "order":
                    session[
                        "order"
                    ],

                "user":
                    session[
                        "user"
                    ],

                # ------------------------------------------------
                # Prediction horizon
                # ------------------------------------------------

                "window":
                    current_window,

                "window_seconds":
                    current_seconds,

                # ------------------------------------------------
                # Game context
                # ------------------------------------------------

                "game_name":
                    session[
                        "game_name"
                    ],

                "game_speed":
                    session[
                        "game_speed"
                    ],

                # ------------------------------------------------
                # User information
                # ------------------------------------------------

                "age":
                    session[
                        "age"
                    ],

                "gender":
                    session[
                        "gender"
                    ],

                "vr_experience":
                    session[
                        "vr_experience"
                    ],

                "gaming_experience":
                    session[
                        "gaming_experience"
                    ],

                # ------------------------------------------------
                # Baseline sickness + targets
                # ------------------------------------------------

                "pre_ssq_total":
                    session[
                        "pre_ssq_total"
                    ],

                "post_ssq_total":
                    session[
                        "post_ssq_total"
                    ],

                "delta_ssq_total":
                    session[
                        "delta_ssq_total"
                    ]
            }

            # ====================================================
            # Add every BASE feature observable by this horizon
            #
            # 1 min -> 1min
            # 3 min -> 1min + 3min
            # 5 min -> 1min + 3min + 5min
            # ====================================================

            for (
                source_window,
                source_seconds
            ) in WINDOWS.items():

                if (
                    source_seconds
                    > current_seconds
                ):
                    continue

                base_features = (
                    base_features_by_window[
                        source_window
                    ]
                )

                for (
                    feature_name,
                    value
                ) in base_features.items():

                    row[
                        f"{source_window}_"
                        f"{feature_name}"
                    ] = value

            # ====================================================
            # Add every TEMPORAL feature observable by horizon
            #
            # Temporal features begin at 3 min because at least
            # two 60-second blocks are required.
            #
            # 3 min -> 3min temporal features
            # 5 min -> 3min temporal + 5min temporal features
            # ====================================================

            for (
                temporal_window,
                temporal_seconds
            ) in WINDOWS.items():

                if (
                    temporal_seconds
                    > current_seconds
                ):
                    continue

                if (
                    temporal_seconds
                    < 180
                ):
                    continue

                blocks = extract_blocks(
                    movement_df,
                    temporal_seconds
                )

                temporal_features = (
                    extract_temporal_features(
                        blocks
                    )
                )

                for (
                    feature_name,
                    value
                ) in temporal_features.items():

                    row[
                        f"{temporal_window}_"
                        f"{feature_name}"
                    ] = value

            rows.append(
                row
            )

    # ========================================================
    # Create final dataframe
    # ========================================================

    output = pd.DataFrame(
        rows
    )

    # ========================================================
    # Validation summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SECOND-GENERATION FEATURE DATASET"
    )

    print(
        "=" * 70
    )

    print(
        "Rows:",
        len(output)
    )

    print(
        "Participants:",
        output[
            "participant_id"
        ].nunique()
    )

    print(
        "\nRows per window:"
    )

    print(
        output[
            "window"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nColumns:",
        len(output.columns)
    )

    # ========================================================
    # Check feature availability by prediction horizon
    # ========================================================

    print(
        "\nMissing values by prediction window:"
    )

    for window in [
        "1min",
        "3min",
        "5min"
    ]:

        subset = output[
            output[
                "window"
            ] == window
        ]

        completely_missing = (
            subset
            .isna()
            .all()
        )

        unavailable_columns = (
            completely_missing[
                completely_missing
            ]
            .index
            .tolist()
        )

        print(
            f"\n"
            f"{window}: "
            f"{len(unavailable_columns)} "
            f"completely unavailable columns"
        )

    # ========================================================
    # Integrity checks
    # ========================================================

    assert (
        len(output)
        == 360
    )

    assert (
        output[
            "participant_id"
        ].nunique()
        == 60
    )

    window_counts = (
        output[
            "window"
        ]
        .value_counts()
        .to_dict()
    )

    assert (
        window_counts[
            "1min"
        ]
        == 120
    )

    assert (
        window_counts[
            "3min"
        ]
        == 120
    )

    assert (
        window_counts[
            "5min"
        ]
        == 120
    )

    # --------------------------------------------------------
    # 3-minute predictions must contain first-minute data
    # --------------------------------------------------------

    assert (
        output.loc[
            output[
                "window"
            ] == "3min",
            "1min_velocity_mean"
        ]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # 5-minute predictions must contain 1- and 3-minute data
    # --------------------------------------------------------

    assert (
        output.loc[
            output[
                "window"
            ] == "5min",
            "1min_velocity_mean"
        ]
        .notna()
        .all()
    )

    assert (
        output.loc[
            output[
                "window"
            ] == "5min",
            "3min_velocity_mean"
        ]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # 5-minute predictions must contain earlier temporal data
    # --------------------------------------------------------

    assert (
        output.loc[
            output[
                "window"
            ] == "5min",
            "3min_velocity_mean_slope"
        ]
        .notna()
        .all()
    )

    assert (
        output.loc[
            output[
                "window"
            ] == "5min",
            "5min_velocity_mean_slope"
        ]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # No feature should be completely missing at 5 minutes
    # --------------------------------------------------------

    five_minute_rows = output[
        output[
            "window"
        ] == "5min"
    ]

    five_minute_all_missing = (
        five_minute_rows
        .isna()
        .all()
    )

    five_minute_all_missing = (
        five_minute_all_missing[
            five_minute_all_missing
        ]
        .index
        .tolist()
    )

    assert (
        len(
            five_minute_all_missing
        )
        == 0
    ), (
        "5-minute rows still contain completely "
        f"unavailable columns: "
        f"{five_minute_all_missing}"
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()