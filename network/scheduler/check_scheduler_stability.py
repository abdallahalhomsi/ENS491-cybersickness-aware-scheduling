import json
import random
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RISK_SCORES_PATH = (
    SCRIPT_DIR
    / "risk_scores.json"
)

OUTPUT_PATH = (
    Path("results/tables")
    / "scheduler_stability.csv"
)


# ============================================================
# Configuration
# ============================================================

NUM_RUNS = 100

NUM_FRAMES_PER_USER = 1000

FRAME_INTERVAL_MS = 14.0
FRAME_DEADLINE_MS = 30.0

SERVICE_TIME_MS = 4.72

PACKET_LOSS_RATE = 0.01
JITTER_MS = 2.0

EPSILON = 0.001

# Any positive value currently gives the same ordering.
# Keep alpha = 1.0 as the main planned configuration.
ALPHA = 1.0


# ============================================================
# Load real ML risk scores
# ============================================================

with open(
    RISK_SCORES_PATH,
    "r"
) as f:

    risk_scores = json.load(f)


users_sorted = sorted(
    risk_scores.keys(),
    key=lambda user:
        risk_scores[user]
)

low_user = users_sorted[0]

medium_user = min(
    users_sorted,
    key=lambda user:
        abs(
            risk_scores[user]
            - 0.50
        )
)

high_user = users_sorted[-1]


print("Selected sessions:")

print(
    f"LOW    : {low_user} "
    f"risk={risk_scores[low_user]:.3f}"
)

print(
    f"MEDIUM : {medium_user} "
    f"risk={risk_scores[medium_user]:.3f}"
)

print(
    f"HIGH   : {high_user} "
    f"risk={risk_scores[high_user]:.3f}"
)


# ============================================================
# Generate traffic
# ============================================================

def generate_packets(
    seed
):

    random.seed(
        seed
    )

    packets = []

    for (
        user,
        risk
    ) in risk_scores.items():

        for frame_id in range(
            NUM_FRAMES_PER_USER
        ):

            base_arrival = (
                frame_id
                * FRAME_INTERVAL_MS
            )

            arrival = (
                base_arrival
                + random.uniform(
                    -JITTER_MS,
                    JITTER_MS
                )
            )

            if (
                random.random()
                < PACKET_LOSS_RATE
            ):
                continue

            deadline = (
                base_arrival
                + FRAME_DEADLINE_MS
            )

            packets.append(
                {
                    "id":
                        f"{user}_{frame_id}",

                    "user":
                        user,

                    "arrival":
                        arrival,

                    "deadline":
                        deadline,

                    "risk":
                        float(
                            risk
                        )
                }
            )

    packets.sort(
        key=lambda packet: (
            packet["arrival"],
            packet["id"]
        )
    )

    return packets


# ============================================================
# Scheduler
# ============================================================

def run_simulation(
    packets,
    scheduler_type
):

    current_time = 0.0

    queue = []

    results = {
        user: {
            "on_time": 0,
            "missed": 0,
            "total": 0
        }
        for user
        in risk_scores
    }

    packet_idx = 0

    total_packets = len(
        packets
    )

    while (
        packet_idx
        < total_packets
        or queue
    ):

        while (
            packet_idx
            < total_packets
            and
            packets[
                packet_idx
            ]["arrival"]
            <= current_time
        ):

            packet = (
                packets[
                    packet_idx
                ]
            )

            queue.append(
                packet
            )

            results[
                packet["user"]
            ]["total"] += 1

            packet_idx += 1

        if not queue:

            if (
                packet_idx
                < total_packets
            ):

                current_time = (
                    packets[
                        packet_idx
                    ]["arrival"]
                )

            continue

        best_index = -1
        best_priority = float(
            "-inf"
        )

        for (
            index,
            packet
        ) in enumerate(
            queue
        ):

            time_to_deadline = max(
                packet["deadline"]
                - current_time,
                EPSILON
            )

            if (
                scheduler_type
                == "EDF"
            ):

                priority = (
                    1.0
                    / time_to_deadline
                )

            elif (
                scheduler_type
                == "RISK_AWARE"
            ):

                priority = (
                    (
                        1.0
                        + ALPHA
                        * packet["risk"]
                    )
                    / time_to_deadline
                )

            else:

                raise ValueError(
                    scheduler_type
                )

            if (
                priority
                > best_priority
            ):

                best_priority = (
                    priority
                )

                best_index = (
                    index
                )

        packet = queue.pop(
            best_index
        )

        if (
            current_time
            <= packet["deadline"]
        ):

            results[
                packet["user"]
            ]["on_time"] += 1

        else:

            results[
                packet["user"]
            ]["missed"] += 1

        current_time += (
            SERVICE_TIME_MS
        )

    return results


# ============================================================
# Jain fairness
# ============================================================

def jain_fairness(
    results
):

    success_rates = []

    for user in results:

        total = (
            results[user]["total"]
        )

        on_time = (
            results[user]["on_time"]
        )

        rate = (
            on_time / total
            if total > 0
            else 0.0
        )

        success_rates.append(
            rate
        )

    numerator = (
        sum(success_rates)
        ** 2
    )

    denominator = (
        len(success_rates)
        * sum(
            x ** 2
            for x
            in success_rates
        )
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# ============================================================
# Repeated experiment
# ============================================================

rows = []


for seed in range(
    NUM_RUNS
):

    packets = (
        generate_packets(
            seed
        )
    )

    edf = (
        run_simulation(
            packets,
            "EDF"
        )
    )

    risk = (
        run_simulation(
            packets,
            "RISK_AWARE"
        )
    )

    edf_total_misses = sum(
        data["missed"]
        for data
        in edf.values()
    )

    risk_total_misses = sum(
        data["missed"]
        for data
        in risk.values()
    )

    rows.append(
        {
            "seed":
                seed,

            "edf_total_misses":
                edf_total_misses,

            "risk_total_misses":
                risk_total_misses,

            "edf_fairness":
                jain_fairness(
                    edf
                ),

            "risk_fairness":
                jain_fairness(
                    risk
                ),

            "edf_low_misses":
                edf[
                    low_user
                ]["missed"],

            "risk_low_misses":
                risk[
                    low_user
                ]["missed"],

            "edf_medium_misses":
                edf[
                    medium_user
                ]["missed"],

            "risk_medium_misses":
                risk[
                    medium_user
                ]["missed"],

            "edf_high_misses":
                edf[
                    high_user
                ]["missed"],

            "risk_high_misses":
                risk[
                    high_user
                ]["missed"]
        }
    )


results = pd.DataFrame(
    rows
)


# ============================================================
# Derived metrics
# ============================================================

results[
    "high_miss_reduction"
] = (
    results[
        "edf_high_misses"
    ]
    - results[
        "risk_high_misses"
    ]
)


results[
    "fairness_change"
] = (
    results[
        "risk_fairness"
    ]
    - results[
        "edf_fairness"
    ]
)


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "SCHEDULER STABILITY OVER "
    f"{NUM_RUNS} RANDOM TRAFFIC RUNS"
)

print(
    "=" * 80
)


print(
    "\nTotal deadline misses"
)

print(
    "EDF mean:",
    round(
        results[
            "edf_total_misses"
        ].mean(),
        2
    )
)

print(
    "Risk-Aware mean:",
    round(
        results[
            "risk_total_misses"
        ].mean(),
        2
    )
)


print(
    "\nHIGH-RISK SESSION"
)

print(
    "EDF misses mean:",
    round(
        results[
            "edf_high_misses"
        ].mean(),
        2
    )
)

print(
    "Risk-Aware misses mean:",
    round(
        results[
            "risk_high_misses"
        ].mean(),
        2
    )
)

print(
    "Mean reduction:",
    round(
        results[
            "high_miss_reduction"
        ].mean(),
        2
    )
)

print(
    "Median reduction:",
    round(
        results[
            "high_miss_reduction"
        ].median(),
        2
    )
)

print(
    "Runs where Risk-Aware helped high-risk user:",
    round(
        100
        * (
            results[
                "high_miss_reduction"
            ]
            > 0
        ).mean(),
        1
    ),
    "%"
)


print(
    "\nMEDIUM-RISK SESSION"
)

print(
    "EDF misses mean:",
    round(
        results[
            "edf_medium_misses"
        ].mean(),
        2
    )
)

print(
    "Risk-Aware misses mean:",
    round(
        results[
            "risk_medium_misses"
        ].mean(),
        2
    )
)


print(
    "\nLOW-RISK SESSION"
)

print(
    "EDF misses mean:",
    round(
        results[
            "edf_low_misses"
        ].mean(),
        2
    )
)

print(
    "Risk-Aware misses mean:",
    round(
        results[
            "risk_low_misses"
        ].mean(),
        2
    )
)


print(
    "\nFAIRNESS"
)

print(
    "EDF mean:",
    round(
        results[
            "edf_fairness"
        ].mean(),
        4
    )
)

print(
    "Risk-Aware mean:",
    round(
        results[
            "risk_fairness"
        ].mean(),
        4
    )
)

print(
    "Mean fairness change:",
    round(
        results[
            "fairness_change"
        ].mean(),
        4
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)