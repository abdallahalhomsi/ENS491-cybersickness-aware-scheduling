import json
import random
from pathlib import Path

import pandas as pd


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
    / "scheduler_congestion_sweep.csv"
)


# ============================================================
# Configuration
# ============================================================

NUM_RUNS = 100

NUM_FRAMES_PER_USER = 1000

FRAME_INTERVAL_MS = 14.0
FRAME_DEADLINE_MS = 30.0

PACKET_LOSS_RATE = 0.01
JITTER_MS = 2.0

EPSILON = 0.001

ALPHA = 1.0


# Three congestion regimes
SERVICE_TIMES = {
    "light": 4.70,
    "moderate": 4.72,
    "heavy": 4.75
}


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


print(
    "Selected sessions:"
)

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
# Generate one traffic realization
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
    scheduler_type,
    service_time_ms
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

        # ----------------------------------------------------
        # Add all packets that have arrived
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Jump forward if queue is empty
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Select best packet
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Deadline outcome
        # ----------------------------------------------------

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
            service_time_ms
        )

    return results


# ============================================================
# Jain fairness
# ============================================================

def calculate_jain_fairness(
    results
):

    success_rates = []

    for user in results:

        total = (
            results[
                user
            ]["total"]
        )

        on_time = (
            results[
                user
            ]["on_time"]
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
        sum(
            success_rates
        )
        ** 2
    )

    denominator = (
        len(
            success_rates
        )
        * sum(
            value ** 2
            for value
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
# Run all congestion regimes
# ============================================================

rows = []


for (
    congestion_name,
    service_time_ms
) in SERVICE_TIMES.items():

    print(
        "\n"
        + "=" * 85
    )

    print(
        f"{congestion_name.upper()} CONGESTION "
        f"(service time = {service_time_ms} ms)"
    )

    print(
        "=" * 85
    )

    for seed in range(
        NUM_RUNS
    ):

        packets = (
            generate_packets(
                seed
            )
        )

        edf_results = (
            run_simulation(
                packets,
                "EDF",
                service_time_ms
            )
        )

        risk_results = (
            run_simulation(
                packets,
                "RISK_AWARE",
                service_time_ms
            )
        )

        edf_total_misses = sum(
            data["missed"]
            for data
            in edf_results.values()
        )

        risk_total_misses = sum(
            data["missed"]
            for data
            in risk_results.values()
        )

        edf_total_packets = sum(
            data["total"]
            for data
            in edf_results.values()
        )

        risk_total_packets = sum(
            data["total"]
            for data
            in risk_results.values()
        )

        rows.append(
            {
                "congestion":
                    congestion_name,

                "service_time_ms":
                    service_time_ms,

                "seed":
                    seed,

                "edf_total_packets":
                    edf_total_packets,

                "risk_total_packets":
                    risk_total_packets,

                "edf_total_misses":
                    edf_total_misses,

                "risk_total_misses":
                    risk_total_misses,

                "edf_miss_rate":
                    (
                        edf_total_misses
                        / edf_total_packets
                        if edf_total_packets > 0
                        else 0.0
                    ),

                "risk_miss_rate":
                    (
                        risk_total_misses
                        / risk_total_packets
                        if risk_total_packets > 0
                        else 0.0
                    ),

                "edf_fairness":
                    calculate_jain_fairness(
                        edf_results
                    ),

                "risk_fairness":
                    calculate_jain_fairness(
                        risk_results
                    ),

                "edf_low_misses":
                    edf_results[
                        low_user
                    ]["missed"],

                "risk_low_misses":
                    risk_results[
                        low_user
                    ]["missed"],

                "edf_medium_misses":
                    edf_results[
                        medium_user
                    ]["missed"],

                "risk_medium_misses":
                    risk_results[
                        medium_user
                    ]["missed"],

                "edf_high_misses":
                    edf_results[
                        high_user
                    ]["missed"],

                "risk_high_misses":
                    risk_results[
                        high_user
                    ]["missed"]
            }
        )


# ============================================================
# Create dataframe
# ============================================================

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
    "medium_miss_reduction"
] = (
    results[
        "edf_medium_misses"
    ]
    - results[
        "risk_medium_misses"
    ]
)


results[
    "low_miss_change"
] = (
    results[
        "risk_low_misses"
    ]
    - results[
        "edf_low_misses"
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
# Save full results
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
# Print summary
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "CONGESTION SWEEP SUMMARY"
)

print(
    "=" * 95
)


for congestion_name in SERVICE_TIMES:

    subset = results[
        results[
            "congestion"
        ]
        == congestion_name
    ]

    print(
        f"\n{congestion_name.upper()}"
    )

    print(
        "-" * 70
    )

    print(
        "EDF mean miss rate:",
        round(
            subset[
                "edf_miss_rate"
            ].mean(),
            4
        )
    )

    print(
        "Risk-Aware mean miss rate:",
        round(
            subset[
                "risk_miss_rate"
            ].mean(),
            4
        )
    )

    print(
        "EDF mean total misses:",
        round(
            subset[
                "edf_total_misses"
            ].mean(),
            2
        )
    )

    print(
        "Risk-Aware mean total misses:",
        round(
            subset[
                "risk_total_misses"
            ].mean(),
            2
        )
    )

    print(
        "\nHigh-risk user:"
    )

    print(
        "  EDF mean misses:",
        round(
            subset[
                "edf_high_misses"
            ].mean(),
            2
        )
    )

    print(
        "  Risk-Aware mean misses:",
        round(
            subset[
                "risk_high_misses"
            ].mean(),
            2
        )
    )

    print(
        "  Mean reduction:",
        round(
            subset[
                "high_miss_reduction"
            ].mean(),
            2
        )
    )

    print(
        "  Median reduction:",
        round(
            subset[
                "high_miss_reduction"
            ].median(),
            2
        )
    )

    print(
        "  Runs improved:",
        round(
            100
            * (
                subset[
                    "high_miss_reduction"
                ]
                > 0
            ).mean(),
            1
        ),
        "%"
    )

    print(
        "\nMedium-risk user:"
    )

    print(
        "  EDF mean misses:",
        round(
            subset[
                "edf_medium_misses"
            ].mean(),
            2
        )
    )

    print(
        "  Risk-Aware mean misses:",
        round(
            subset[
                "risk_medium_misses"
            ].mean(),
            2
        )
    )

    print(
        "\nLow-risk user:"
    )

    print(
        "  EDF mean misses:",
        round(
            subset[
                "edf_low_misses"
            ].mean(),
            2
        )
    )

    print(
        "  Risk-Aware mean misses:",
        round(
            subset[
                "risk_low_misses"
            ].mean(),
            2
        )
    )

    print(
        "\nFairness:"
    )

    print(
        "  EDF mean:",
        round(
            subset[
                "edf_fairness"
            ].mean(),
            4
        )
    )

    print(
        "  Risk-Aware mean:",
        round(
            subset[
                "risk_fairness"
            ].mean(),
            4
        )
    )

    print(
        "  Mean change:",
        round(
            subset[
                "fairness_change"
            ].mean(),
            4
        )
    )


# ============================================================
# Compact comparison table
# ============================================================

summary_rows = []


for congestion_name in SERVICE_TIMES:

    subset = results[
        results[
            "congestion"
        ]
        == congestion_name
    ]

    summary_rows.append(
        {
            "congestion":
                congestion_name,

            "service_time_ms":
                subset[
                    "service_time_ms"
                ].iloc[0],

            "edf_miss_rate":
                subset[
                    "edf_miss_rate"
                ].mean(),

            "risk_miss_rate":
                subset[
                    "risk_miss_rate"
                ].mean(),

            "high_risk_reduction":
                subset[
                    "high_miss_reduction"
                ].mean(),

            "high_risk_improved_pct":
                100
                * (
                    subset[
                        "high_miss_reduction"
                    ]
                    > 0
                ).mean(),

            "edf_fairness":
                subset[
                    "edf_fairness"
                ].mean(),

            "risk_fairness":
                subset[
                    "risk_fairness"
                ].mean()
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print(
    "\n"
    + "=" * 95
)

print(
    "COMPACT COMPARISON"
)

print(
    "=" * 95
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved full results:"
)

print(
    OUTPUT_PATH
)