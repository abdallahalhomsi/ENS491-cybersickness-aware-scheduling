import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path("data/raw/Questset/movement")
SESSION_FILE = Path("data/processed/questset_sessions_with_ssq.csv")
OUT_DIR = Path("data/processed")

WINDOWS = {
    "1min": 60,
    "3min": 180,
    "5min": 300
}


def safe_stats(values, prefix):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) == 0:
        return {}

    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_median": values.median(),
        f"{prefix}_p25": values.quantile(0.25),
        f"{prefix}_p75": values.quantile(0.75),
        f"{prefix}_p95": values.quantile(0.95)
    }


def extract_features(df):
    features = {}
# --------------------------------------------------
# Resample to a uniform 50 Hz grid
# --------------------------------------------------

    RESAMPLE_HZ = 50
    step = 1.0 / RESAMPLE_HZ

    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    new_time = np.arange(
        df["time"].iloc[0],
        df["time"].iloc[-1],
        step
    )

    numeric_cols = [
        "HeadPosX",
        "HeadPosY",
        "HeadPosZ",
        "HeadOrientationW",
        "HeadOrientationX",
        "HeadOrientationY",
        "HeadOrientationZ"
    ]

    resampled = {"time": new_time}

    for col in numeric_cols:
        resampled[col] = np.interp(
            new_time,
            df["time"].to_numpy(),
            df[col].to_numpy()
        )

    df = pd.DataFrame(resampled)

    # Renormalize interpolated quaternions
    qcols = [
        "HeadOrientationW",
        "HeadOrientationX",
        "HeadOrientationY",
        "HeadOrientationZ"
    ]

    q = df[qcols].to_numpy()
    qnorm = np.linalg.norm(q, axis=1)

    valid_q = qnorm > 0
    q[valid_q] = q[valid_q] / qnorm[valid_q, None]

    df[qcols] = q

    features["resample_hz"] = RESAMPLE_HZ
    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    time = df["time"].to_numpy()

    dt = np.diff(time)

    valid_dt = dt > 0

    if valid_dt.sum() == 0:
        raise ValueError("No valid positive time differences found.")

    # --------------------------------------------------
    # Head position
    # --------------------------------------------------

    pos = df[
        ["HeadPosX", "HeadPosY", "HeadPosZ"]
    ].to_numpy(dtype=float)

    displacement = np.diff(pos, axis=0)

    distance_step = np.linalg.norm(displacement, axis=1)
    # Remove implausible tracking jumps
    valid_motion = distance_step <= 0.10

    distance_step_clean = distance_step.copy()
    distance_step_clean[~valid_motion] = np.nan

    # total physical head travel
    features["head_total_distance"] = np.nansum(distance_step_clean)

    # displacement from starting point
    net_disp = np.linalg.norm(pos[-1] - pos[0])
    features["head_net_displacement"] = net_disp

    # spatial range
    for axis, col in zip(["x", "y", "z"], range(3)):
        features[f"head_range_{axis}"] = (
            np.nanmax(pos[:, col]) - np.nanmin(pos[:, col])
        )

    # --------------------------------------------------
    # Linear velocity
    # --------------------------------------------------

    velocity = np.full(len(distance_step_clean), np.nan)

    valid_velocity = valid_dt & valid_motion

    velocity[valid_velocity] = (
        distance_step_clean[valid_velocity]
        / dt[valid_velocity]
    )
    features["tracking_jump_count"] = int((~valid_motion).sum())
    features["tracking_jump_fraction"] = float((~valid_motion).mean())

    features.update(
        safe_stats(velocity, "head_velocity")
    )

    # --------------------------------------------------
    # Linear acceleration
    # --------------------------------------------------

    if len(velocity) > 1:
        dt_acc = dt[1:]

        acceleration = np.full(len(velocity) - 1, np.nan)

        valid = (
            (dt_acc > 0)
            & np.isfinite(velocity[1:])
            & np.isfinite(velocity[:-1])
        )

        acceleration[valid] = (
            np.abs(np.diff(velocity)[valid]) / dt_acc[valid]
        )

        features.update(
            safe_stats(acceleration, "head_acceleration")
        )

        # --------------------------------------------------
        # Jerk
        # --------------------------------------------------

        if len(acceleration) > 1:
            dt_jerk = dt_acc[1:]

            jerk = np.full(len(acceleration) - 1, np.nan)

            valid = (
                (dt_jerk > 0)
                & np.isfinite(acceleration[1:])
                & np.isfinite(acceleration[:-1])
            )

            jerk[valid] = (
                np.abs(np.diff(acceleration)[valid])
                / dt_jerk[valid]
            )

            features.update(
                safe_stats(jerk, "head_jerk")
            )

    # --------------------------------------------------
    # Quaternion angular movement
    # --------------------------------------------------

    quat = df[
        [
            "HeadOrientationW",
            "HeadOrientationX",
            "HeadOrientationY",
            "HeadOrientationZ"
        ]
    ].to_numpy(dtype=float)

    # normalize quaternion
    norms = np.linalg.norm(quat, axis=1)
    valid_norm = norms > 0

    quat_norm = quat.copy()
    quat_norm[valid_norm] = (
        quat_norm[valid_norm] / norms[valid_norm, None]
    )

    q1 = quat_norm[:-1]
    q2 = quat_norm[1:]

    # absolute dot handles q and -q representing same rotation
    dots = np.abs(np.sum(q1 * q2, axis=1))
    dots = np.clip(dots, -1.0, 1.0)

    # angular difference in radians
    angular_change = 2 * np.arccos(dots)

    angular_velocity = np.full(len(angular_change), np.nan)

    angular_velocity[valid_dt] = (
        angular_change[valid_dt] / dt[valid_dt]
    )

    features["head_total_rotation_rad"] = np.nansum(angular_change)

    features.update(
        safe_stats(angular_velocity, "head_angular_velocity")
    )

    # --------------------------------------------------
    # Sampling information
    # --------------------------------------------------

    features["n_samples"] = len(df)
    features["window_duration_actual"] = (
        df["time"].iloc[-1] - df["time"].iloc[0]
    )

    features["median_dt"] = np.nanmedian(dt)
    features["effective_hz"] = (
        1 / np.nanmedian(dt) if np.nanmedian(dt) > 0 else np.nan
    )

    return features


def main():
    sessions = pd.read_csv(SESSION_FILE)

    all_rows = []

    for i, row in sessions.iterrows():

        filename = Path(row["movement_filepath"]).name
        filepath = RAW_DIR / filename

        print(
            f"[{i+1}/{len(sessions)}] "
            f"{row['participant_id']} | "
            f"{row['game_speed']}"
        )

        df = pd.read_csv(filepath)

        for window_name, seconds in WINDOWS.items():

            window_df = df[df["time"] <= seconds].copy()

            if len(window_df) < 2:
                print(
                    f"WARNING: insufficient samples for "
                    f"{filename} {window_name}"
                )
                continue

            features = extract_features(window_df)

            output_row = {
                "participant_id": row["participant_id"],
                "session_number": row["session_number"],
                "group": row["group"],
                "order": row["order"],
                "user": row["user"],
                "game_speed": row["game_speed"],
                "game_name": row["game_name"],
                "window": window_name,
                "window_seconds": seconds,
                "pre_ssq_total": row["pre_ssq_total"],
                "post_ssq_total": row["post_ssq_total"],
                "delta_ssq_total": row["delta_ssq_total"],
                "age": row["age"],
                "gender": row["gender"],
                "vr_experience": row["vr_experience"],
                "gaming_experience": row["gaming_experience"]
            }

            output_row.update(features)

            all_rows.append(output_row)

    output = pd.DataFrame(all_rows)

    out_path = OUT_DIR / "questset_movement_features.csv"
    output.to_csv(out_path, index=False)

    print("\n--------------------------------")
    print("Finished")
    print("--------------------------------")
    print("Rows:", len(output))
    print("Participants:", output["participant_id"].nunique())
    print("\nRows per window:")
    print(output["window"].value_counts())
    print("\nMissing values by column:")
    print(output.isna().sum()[output.isna().sum() > 0])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()