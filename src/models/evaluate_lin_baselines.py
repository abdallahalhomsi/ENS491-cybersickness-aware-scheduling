"""Evaluate literature baselines on the exported Questset participant folds.

Baseline labels follow the literature notes (Section 6.2), adapted to the
high-risk classification task (post SSQ > 30):
    B0  constant risk: training-fold prevalence (notes: training mean post-SSQ)
    B1  logistic readout of saved 1-minute cumulative rotation C
    B2  no-drain queue  S(t) = max(0, S(t-1) + R(t))
    B3  drain queue     S(t) = max(0, S(t-1) + R(t) - delta), delta chosen by inner CV
B2 and B3 share the normalized yaw/pitch input R(t), zero initialization and
the same logistic readout of S(60).

The metrics table is printed and saved to --out
(default results/tables/questset_lin_baselines.csv).

The repository's resampler mutates a NumPy view, so use pandas 2.x.
"""

import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from export_acceleration_risk_scores import FEATURES, build_model

# Import the existing extractor without requiring a new package or entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_second_generation_features import (
    RESAMPLE_HZ,
    TRACKING_JUMP_THRESHOLD,
    calculate_motion_signals,
    resample_movement,
)


ROOT = Path(__file__).resolve().parents[2]
KEY = ["participant_id", "session_number"]
ROTATION = "1min_cumulative_rotation"
WINDOW = 60.0
MAX_GAP = 0.05  # seconds, approximately three nominal Questset samples
EDGE_TOLERANCE = 1 / RESAMPLE_HZ
QUATERNION = [f"HeadOrientation{axis}" for axis in "WXYZ"]
POSITION = [f"HeadPos{axis}" for axis in "XYZ"]
# Select a quantile of observed training-slot inputs, then refit that quantile
# on the outer training set. These are search levels, not published drain rates.
DRAIN_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class TraceCoverageError(ValueError):
    """The observed trace cannot support the fixed-window queue."""


def load_sessions():
    data = ROOT / "data/processed"
    sessions = pd.read_csv(
        ROOT / "results/tables/questset_acceleration_risk_scores.csv"
    )
    features = pd.read_csv(data / "questset_second_generation_features.csv")
    manifest = pd.read_csv(data / "questset_sessions_with_ssq.csv")
    sessions = sessions.merge(
        features.loc[features["window"] == "1min", KEY + [ROTATION]],
        on=KEY, how="left", validate="one_to_one",
    ).merge(
        manifest[KEY + ["movement_filepath"]],
        on=KEY, how="left", validate="one_to_one",
    )
    if (
        len(sessions) != 120
        or sessions["participant_id"].nunique() != 60
        or set(sessions["cv_fold"]) != set(range(1, 6))
        or sessions.groupby("participant_id")["cv_fold"].nunique().ne(1).any()
        or not sessions["high_risk_true"].equals(
            sessions["post_ssq_total"].gt(30).astype(int)
        )
        or sessions[[ROTATION, "movement_filepath"]].isna().any().any()
    ):
        raise ValueError("The exported Questset cohort or its session joins are invalid.")
    return sessions


def collect_traces(sessions, archive_path):
    traces = {}
    rotation_errors = []
    full_grid_errors = []
    position_jumps = 0
    position_jump_sessions = 0
    sign_flip_sessions = 0
    with ZipFile(archive_path) as archive:
        members = defaultdict(list)
        for name in archive.namelist():
            path = PurePosixPath(name)
            if "Complete data" in path.parts and path.name.endswith("_movement.csv"):
                members[path.name].append(name)

        for index, row in sessions.iterrows():
            filename = PurePosixPath(row["movement_filepath"]).name
            try:
                matches = members[filename]
                if len(matches) != 1:
                    raise TraceCoverageError(f"expected one archive member, found {len(matches)}")
                with archive.open(matches[0]) as movement:
                    raw = pd.read_csv(movement, usecols=["time", *POSITION, *QUATERNION])
                trace, error, full_grid, jumps, sign_flips = prepare_trace(raw, row[ROTATION])
            except (TraceCoverageError, pd.errors.ParserError) as error:
                print(f"Excluded {row['participant_id']} session {row['session_number']} "
                      f"({filename}): {error}")
                continue
            traces[index] = trace
            rotation_errors.append(error)
            if full_grid:
                full_grid_errors.append(error)
            position_jumps += jumps
            position_jump_sessions += jumps > 0
            sign_flip_sessions += sign_flips > 0

    print(f"Eligible traces: {len(traces)}/{len(sessions)}")
    if rotation_errors:
        errors = np.asarray(rotation_errors)
        print(f"Saved B1 cumulative-rotation audit:{np.count_nonzero(abs(errors) <= 1e-6)} "
              f"agree within 1e-6 rad on the strict grid, median absolute "
              f"difference {np.median(abs(errors)):.6g} rad, "
              f"max {max(abs(errors)):.6g} rad.")
        print(f"On fully supported legacy grids, "
              f"{np.count_nonzero(np.abs(full_grid_errors) <= 1e-6)}/{len(full_grid_errors)} "
              f"match within 1e-6 rad. Strict reconstructions exceeding the saved total "
              f"by more than 1e-6 rad: {np.count_nonzero(errors < -1e-6)}.")
        print(f"Eligible traces with antipodal quaternion sign flips: {sign_flip_sessions}. "
              f"Excluded from the rotation-only queue input: translational position "
              f"jumps/invalid steps ({position_jumps} steps in {position_jump_sessions} sessions).")
    return traces


def prepare_trace(raw, saved_rotation):
    # Keep the archived time origin and audit raw coverage before interpolation.
    if not np.isfinite(raw["time"]).all():
        raise TraceCoverageError("non-finite timestamps")
    raw = raw.loc[raw["time"].between(0, WINDOW)]
    raw = raw.sort_values("time").drop_duplicates("time")
    time = raw["time"].to_numpy()
    if len(time) < 2:
        raise TraceCoverageError("fewer than two samples in the first 60 seconds")
    gaps = np.diff(time)
    issues = []
    if time[0] > EDGE_TOLERANCE or WINDOW - time[-1] > EDGE_TOLERANCE:
        issues.append(f"observed only {time[0]:.3f} to {time[-1]:.3f} s")
    if np.any(gaps > MAX_GAP):
        issues.append(f"{np.count_nonzero(gaps > MAX_GAP)} gaps > {MAX_GAP:g} s "
                      f"(largest {gaps.max():.3f} s)")
    quaternion = raw[QUATERNION].to_numpy()
    if not np.isfinite(quaternion).all() or np.any(np.linalg.norm(quaternion, axis=1) == 0):
        issues.append("invalid raw quaternions")
    if issues:
        raise TraceCoverageError(", ".join(issues))

    # Audit the archived feature using the original interpolation convention.
    # The saved builder resampled through 300 s before clipping to 60 s.
    grid = resample_movement(raw)
    signals = calculate_motion_signals(grid)
    rotation_error = saved_rotation - signals["cumulative_rotation"]
    full_grid = grid["time"].iloc[-1] + 1 / RESAMPLE_HZ > WINDOW + 1e-9
    position_steps = np.linalg.norm(np.diff(grid[POSITION].to_numpy(), axis=0), axis=1)
    jumps = np.count_nonzero(
        ~np.isfinite(position_steps) | (position_steps > TRACKING_JUMP_THRESHOLD)
    )
    # Preserve orientation under equivalent q/-q encodings before interpolation.
    quaternion = raw[QUATERNION].to_numpy()
    signs = np.where(np.sum(quaternion[1:] * quaternion[:-1], axis=1) < 0, -1, 1)
    sign_flips = np.count_nonzero(signs < 0)
    if sign_flips:
        raw = raw.copy()
        raw[QUATERNION] = quaternion * np.r_[1, np.cumprod(signs)][:, None]
        grid = resample_movement(raw)
    return slot_inputs(grid), rotation_error, full_grid, jumps, sign_flips


def slot_inputs(grid):
    """Time-weighted mean normalized yaw/pitch speed in each one-second bin."""
    q = grid[QUATERNION].to_numpy()
    if not np.allclose(np.linalg.norm(q, axis=1), 1.0):
        raise TraceCoverageError("invalid resampled quaternions")
    w, x, y, z = q.T
    # Heading and elevation of the rotated +Z headset-forward vector.
    forward_x = 2 * (x * z + w * y)
    forward_y = 2 * (y * z - w * x)
    forward_z = 1 - 2 * (x * x + y * y)
    horizontal = np.hypot(forward_x, forward_z)
    if np.any(horizontal < 1e-8):
        raise TraceCoverageError("undefined yaw at the vertical pole")
    yaw = np.unwrap(np.arctan2(forward_x, forward_z))
    pitch = np.arctan2(forward_y, horizontal)
    time = grid["time"].to_numpy()
    rate = (
        np.hypot(np.diff(yaw), np.diff(pitch))
        / np.diff(time) * (180 / np.pi) / (100 * np.sqrt(2))
    )
    starts = np.arange(int(WINDOW))[:, None]
    overlaps = np.maximum(
        0, np.minimum(starts + 1, time[1:]) - np.maximum(starts, time[:-1])
    )
    return (overlaps * rate).sum(axis=1) / overlaps.sum(axis=1)


def queue_states(inputs, drain):
    """Lin §II-B Eq. 3, reduced per repository literature notes pp. 10–11."""
    states = np.empty_like(inputs)
    state = np.zeros(len(inputs))
    for slot, arrivals in enumerate(inputs.T):
        state = np.maximum(0, state + arrivals - drain)
        states[:, slot] = state
    return states


def calibrated_probability(feature, labels, train, test):
    model = build_model()
    return model.fit(feature[train], labels[train]).predict_proba(feature[test])[:, 1]


def select_drain(inputs, labels, groups, train):
    inner_inputs, inner_labels = inputs[train], labels[train]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = tuple(cv.split(inner_inputs, inner_labels, groups[train]))
    probabilities = np.empty((len(DRAIN_QUANTILES), len(inner_labels)))
    for fit, validate in splits:
        drains = np.quantile(inner_inputs[fit], DRAIN_QUANTILES)
        for i, drain in enumerate(drains):
            feature = queue_states(inner_inputs, drain)[:, -1:]
            probabilities[i, validate] = calibrated_probability(
                feature, inner_labels, fit, validate
            )
    performance = [average_precision_score(inner_labels, p) for p in probabilities]
    selected = DRAIN_QUANTILES[int(np.argmax(performance))]
    return selected, float(np.quantile(inner_inputs, selected))


def evaluate(sessions, cohort, inputs=None):
    labels = sessions["high_risk_true"].to_numpy()
    groups = sessions["participant_id"].to_numpy()
    folds = sessions["cv_fold"].to_numpy()
    rotation = sessions[[ROTATION]].to_numpy()
    features = sessions[FEATURES].to_numpy()
    proposed = sessions["risk_probability"].to_numpy().copy()
    b0 = np.empty(len(labels))
    b1 = np.empty(len(labels))
    if inputs is not None:
        b2_states = queue_states(inputs, 0)
        b3_states = np.empty_like(inputs)
        b2 = np.empty(len(labels))
        b3 = np.empty(len(labels))

    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        b0[test] = labels[train].mean()
        b1[test] = calibrated_probability(rotation, labels, train, test)
        if inputs is None:
            continue

        # Refit the same proposed pipeline so its training cohort also matches.
        proposed[test] = calibrated_probability(features, labels, train, test)
        b2[test] = calibrated_probability(b2_states[:, -1:], labels, train, test)
        quantile, drain = select_drain(inputs, labels, groups, train)
        states = queue_states(inputs, drain)
        b3_states[test] = states[test]
        b3[test] = calibrated_probability(states[:, -1:], labels, train, test)
        print(f"Fold {fold}: n={len(test)}, positives={labels[test].sum()}, "
              f"drain quantile={quantile:g}, drain per slot={drain:.6f}")

    print(f"Sessions={len(sessions)}, participants={len(set(groups))}, "
          f"positives={labels.sum()}")
    scores = {"B0 constant risk": b0, "B1 cumulative rotation": b1}
    if inputs is not None:
        scores.update({"B2 no-drain queue": b2, "B3 drain queue": b3})
    scores["Proposed acceleration"] = proposed
    table = report_metrics(labels, scores)
    table.insert(0, "cohort", cohort)
    table.insert(1, "n_sessions", len(labels))
    table.insert(2, "n_participants", len(set(groups)))
    table.insert(3, "n_positives", int(labels.sum()))
    if inputs is not None:
        for name, states in (("B2", b2_states), ("B3", b3_states)):
            final = states[:, -1]
            final_zeros = int((final == 0).sum())
            any_zero = int((states == 0).any(axis=1).sum())
            print(f"{name} queue: final zeros={final_zeros}/{len(final)}, "
                  f"any zero after initialization={any_zero}/{len(final)}, "
                  f"distinct final scores={len(np.unique(final.round(8)))}, "
                  f"range=[{final.min():.6g}, {final.max():.6g}], SD={final.std():.6g}")
            row = table["model"].str.startswith(name)
            table.loc[row, "floor_final_zero"] = final_zeros
            table.loc[row, "floor_any_zero"] = any_zero
    return table


METRIC_NAMES = ("roc_auc", "average_precision", "balanced_accuracy",
                "precision", "recall", "f1")


def report_metrics(labels, scores):
    """Print one row per model and return the same metrics as a DataFrame."""
    print(f"{'Model':<25} {'ROC-AUC':>8} {'AP':>8} {'Bal.Acc':>8} "
          f"{'Prec.':>8} {'Recall':>8} {'F1':>8}")
    rows = []
    for name, probability in scores.items():
        predicted = probability >= 0.5
        metrics = (
            roc_auc_score(labels, probability),
            average_precision_score(labels, probability),
            balanced_accuracy_score(labels, predicted),
            precision_score(labels, predicted, zero_division=0),
            recall_score(labels, predicted, zero_division=0),
            f1_score(labels, predicted, zero_division=0),
        )
        print(f"{name:<25}" + "".join(f" {score:8.4f}" for score in metrics))
        rows.append({"model": name, **dict(zip(METRIC_NAMES, metrics))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path.home() / "Downloads/Questset.zip")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/tables/questset_lin_baselines.csv",
                        help="CSV for the metrics of both cohorts.")
    args = parser.parse_args()
    if int(pd.__version__.split(".")[0]) >= 3:
        parser.exit(2, "The existing quaternion resampler requires pandas 2.x.\n")
    sessions = load_sessions()
    print("Original 120-session comparison (saved proposed-model predictions)")
    tables = [evaluate(sessions, "all_120_saved_predictions")]
    print("\nMovement audit (no raw samples after 60 seconds)")
    traces = collect_traces(sessions, args.archive)
    if not traces:
        raise SystemExit("No eligible traces for the queue comparison.")
    print("\nCommon eligible-cohort comparison (original fold assignments)")
    tables.append(evaluate(sessions.loc[list(traces)], "eligible_queue_cohort",
                           np.stack(list(traces.values()))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)
    print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
