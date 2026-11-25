"""Audit released VRWalking data and prepare provisional rotation/FMS inputs.

Block numbers identify contiguous CSV row ranges. FMS is a minute-bin
annotation, and the files provide neither participant keys nor response times.
"""

import argparse
from functools import partial
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE), str(SOURCE / "models")]
from models.evaluate_lin_baselines import QUATERNION, WINDOW, queue_states, slot_inputs


FOLDER = "Grad/VRWalking"
ROTATION = [f"HMDRot_{axis}" for axis in "WXYZ"]
ROWS_PER_MINUTE = 60
MINUTES_PER_BLOCK = 14
BLOCK_ROWS = ROWS_PER_MINUTE * MINUTES_PER_BLOCK
CLOCK_TICKS_PER_SECOND = 10_000_000  # inferred from the published 1 Hz release
MIN_DT, MAX_DT = 0.5, 1.5  # provisional coverage rule for the nominal 1 Hz export
MIN_NORM = 0.5  # provisional rejection of strongly attenuated orientations
FMS_SUMMARY = ["First_FMS", "Second_FMS", "Overall_FMS"]


def load_archive(path):
    with ZipFile(path) as archive:
        with archive.open(f"{FOLDER}/all_data_reac_time.csv") as handle:
            motion = pd.read_csv(handle)
        with archive.open(f"{FOLDER}/SSQ_scores_NASA_Reaction_Time.csv") as handle:
            summary = pd.read_csv(handle)
    if len(motion) != len(summary) * BLOCK_ROWS:
        raise ValueError("Unexpected concatenated VRWalking row count.")
    starts = np.arange(BLOCK_ROWS, len(motion), BLOCK_ROWS)
    if not np.all(np.diff(motion["EyeFrames"])[starts - 1] < 0):
        raise ValueError("Expected EyeFrames reset missing at a candidate block boundary.")
    return motion, summary


def prepare_block(block_id, block):
    q = block[ROTATION].to_numpy(dtype=float)
    norm = np.linalg.norm(q, axis=1)
    valid_q = np.isfinite(norm) & (norm >= MIN_NORM)
    timing = block["ViveTiming"].to_numpy(dtype=float)
    time = (timing - timing[0]) / CLOCK_TICKS_PER_SECOND
    dt = np.diff(time)
    valid_dt = (dt >= MIN_DT) & (dt <= MAX_DT)
    valid_step = valid_dt & valid_q[:-1] & valid_q[1:]
    fms = block["fms"].to_numpy().reshape(-1, ROWS_PER_MINUTE)
    fms_value = fms[:, 0]
    fms_valid = np.isfinite(fms).all(axis=1) & (fms == fms_value[:, None]).all(axis=1)
    fms_value = np.where(fms_valid, fms_value, np.nan)
    minute_labels = pd.DataFrame({
        "block": block_id,
        "minute_bin": np.arange(1, MINUTES_PER_BLOCK + 1),
        "fms": fms_value,
    })

    # Each split marks a missing or unsupported motion interval. Never bridge it.
    bounds = np.r_[0, np.flatnonzero(~valid_step) + 1, len(block)]
    slots = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start < 2:
            continue
        segment_time = time[start:end] - time[start]
        n_seconds = int(np.floor(segment_time[-1]))
        if n_seconds == 0:
            continue
        q_segment = q[start:end] / norm[start:end, None]
        arrivals = []
        for offset in range(0, n_seconds, int(WINDOW)):
            first = np.searchsorted(segment_time, offset, side="right") - 1
            last = np.searchsorted(segment_time, offset + WINDOW, side="left") + 1
            grid = pd.DataFrame(q_segment[first:last], columns=QUATERNION)
            grid["time"] = segment_time[first:last] - offset
            # The Questset helper emits a fixed 60 slots. Keep only fully
            # covered seconds for a fragment's final partial window.
            count = min(int(WINDOW), n_seconds - offset)
            with np.errstate(invalid="ignore", divide="ignore"):
                arrivals.extend(slot_inputs(grid)[:count])

        seconds = np.arange(n_seconds)
        left_row = start + np.searchsorted(segment_time, seconds, side="right") - 1
        right_row = start + np.searchsorted(segment_time, seconds + 1, side="left")
        minute = left_row // ROWS_PER_MINUTE
        same_minute = minute == right_row // ROWS_PER_MINUTE
        slots.append(pd.DataFrame({
            "block": block_id,
            "fragment_start_row": start + 1,
            "fragment_second": seconds + 1,
            "block_clock_end_s": time[start] + seconds + 1,
            "rotation_input": arrivals,
            "fms_minute_bin": np.where(same_minute, minute + 1, np.nan),
            "fms_bin_annotation": np.where(same_minute, fms_value[minute], np.nan),
        }))

    return slots, minute_labels, dt, valid_dt, norm, valid_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path.home() / "OneDrive/Escritorio/Grad.zip")
    parser.add_argument("--show-slots", action="store_true",
                        help="Print the prepared motion/FMS slot table as CSV to stdout.")
    args = parser.parse_args()
    report = partial(print, file=sys.stderr if args.show_slots else sys.stdout)
    motion, summary = load_archive(args.archive)
    report(f"Motion rows={len(motion)}; summary rows={len(summary)}; "
          f"provisional blocks={len(summary)} of {BLOCK_ROWS} rows.")
    report("All between-block boundaries have an EyeFrames reset. "
          "No participant key or confirmed join is present.")

    results = [
        prepare_block(i + 1, motion.iloc[start:start + BLOCK_ROWS])
        for i, start in enumerate(range(0, len(motion), BLOCK_ROWS))
    ]
    slot_parts, minute_parts, dt_parts, valid_parts, norm_parts, step_parts = zip(*results)
    fragments = [part for block in slot_parts for part in block]
    if not fragments:
        raise SystemExit("No supported whole-second motion slots.")
    slots = pd.concat(fragments, ignore_index=True)
    if not np.isfinite(slots["rotation_input"]).all():
        raise ValueError("A prepared slot has no finite rotation input.")
    minutes = pd.concat(minute_parts, ignore_index=True)
    dt = np.concatenate(dt_parts)
    valid_dt = np.concatenate(valid_parts)
    norm = np.concatenate(norm_parts)
    report(f"Within-block timing: {valid_dt.sum()}/{len(dt)} intervals in "
          f"[{MIN_DT}, {MAX_DT}] s; median={np.median(dt[valid_dt]):.6f} s; "
          f"range={dt[valid_dt].min():.6f}–{dt[valid_dt].max():.6f} s; "
          f"zero={(dt == 0).sum()}, negative={(dt < 0).sum()}.")
    report("Blocks with no supported clock intervals:",
          [i + 1 for i, valid in enumerate(valid_parts) if not valid.any()])
    report(f"Quaternion norm min/median/max={np.nanmin(norm):.6f}/"
          f"{np.nanmedian(norm):.6f}/{np.nanmax(norm):.6f}; "
          f"nonfinite={(~np.isfinite(norm)).sum()}; below {MIN_NORM}={(norm < MIN_NORM).sum()}; "
          f"accepted timing and quaternion intervals={sum(step.sum() for step in step_parts)}.")

    candidates = summary[FMS_SUMMARY].to_numpy()
    values = minutes["fms"].to_numpy().reshape(-1, MINUTES_PER_BLOCK)
    signatures = np.column_stack((values[:, :7].mean(axis=1),
                                  values[:, 7:].mean(axis=1), values.mean(axis=1)))
    match_counts = np.isclose(signatures[:, None], candidates, atol=1e-6, rtol=0).all(axis=2).sum(axis=1)
    report(f"FMS constant and finite in {minutes['fms'].count()}/{len(minutes)} minute bins; "
          f"observed range={minutes['fms'].min():g}–{minutes['fms'].max():g}. "
          f"Unique FMS-summary candidates={np.count_nonzero(match_counts == 1)}/{len(summary)}.")
    complete = [part for part in fragments if len(part) >= WINDOW]
    covered_minutes = slots.dropna(subset=["fms_bin_annotation"])[
        ["block", "fms_minute_bin"]
    ].drop_duplicates()
    report(f"Prepared {len(slots)} whole-second slots in {len(fragments)} fragments "
          f"across {slots['block'].nunique()} blocks; "
          f"{slots['fms_bin_annotation'].count()} align to {len(covered_minutes)} FMS bins; "
          f"{len(complete)} fragments across {len({p['block'].iloc[0] for p in complete})} "
          "blocks support 60 consecutive seconds.")
    report("Provisional blocks with 60 supported seconds starting at the first CSV row:",
           [int(p["block"].iloc[0]) for p in complete if p["fragment_start_row"].iloc[0] == 1])
    if not complete:
        raise SystemExit("No supported 60-second fragment for the queue demonstration.")
    demo = max(complete, key=len).iloc[:int(WINDOW)].copy()
    inputs = demo["rotation_input"].to_numpy()
    # Smoke test only. The adapted drain is the fixed illustrative value in the
    # repository literature notes (p. 4), without selecting it from VRWalking.
    demo["queue_no_drain"] = queue_states(inputs[None, :], drain=0)[0]
    demo["queue_example_drain"] = queue_states(inputs[None, :], drain=0.2)[0]
    first = demo.iloc[0]
    last = demo.iloc[-1]
    report(f"Demo block={int(first['block'])}, starting row within block="
          f"{int(first['fragment_start_row'])}, block-clock window="
          f"{first['block_clock_end_s'] - 1:.3f}–{last['block_clock_end_s']:.3f} s, "
          f"rotation input min/mean/max={inputs.min():.6f}/{inputs.mean():.6f}/{inputs.max():.6f}.")
    report(f"Final queue state: drain=0 -> {last['queue_no_drain']:.6f}; "
          f"drain=0.2 -> {last['queue_example_drain']:.6f}; "
          f"FMS annotations={demo['fms_bin_annotation'].count()}/{len(demo)}.")
    report(demo.iloc[[0, 1, -1]].to_string(index=False))
    if args.show_slots:
        print(slots.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
