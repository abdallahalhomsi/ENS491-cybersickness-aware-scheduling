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
    / "scheduler_alpha_sweep.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

NUM_FRAMES_PER_USER = 1000

FRAME_INTERVAL_MS = 14.0
FRAME_DEADLINE_MS = 30.0

# Keep the operating point we just identified.
SERVICE_TIME_MS = 4.72

PACKET_LOSS_RATE = 0.01
JITTER_MS = 2.0

EPSILON = 0.001


ALPHA_VALUES = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25
]


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
    "Selected users:"
)

print(
    f"LOW    : {low_user} "
    f"({risk_scores[low_user]:.3f})"
)

print(
    f"MEDIUM : {medium_user} "
    f"({risk_scores[medium_user]:.3f})"
)

print(
    f"HIGH   : {high_user} "
    f"({risk_scores[high_user]:.3f})"
)


# ============================================================
# Generate ONE fixed traffic realization
#
# Important:
# Every alpha value sees exactly the same traffic.
# ============================================================

random.seed(
    RANDOM_SEED
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
                    float(risk)
            }
        )


packets.sort(
    key=lambda packet: (
        packet["arrival"],
        packet["id"]
    )
)


print(
    "\nGenerated packets:",
    len(packets)
)


# ============================================================
# Scheduler
# ============================================================

def run_simulation(
    alpha
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
        # Add arrived packets
        # ----------------------------------------------------

        while (
            packet_idx
            < total_packets
            and
            packets[
                packet_idx
            ][
                "arrival"
            ]
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
                packet[
                    "user"
                ]
            ][
                "total"
            ] += 1

            packet_idx += 1

        # ----------------------------------------------------
        # Jump forward when queue is empty
        # ----------------------------------------------------

        if not queue:

            if (
                packet_idx
                < total_packets
            ):

                current_time = (
                    packets[
                        packet_idx
                    ][
                        "arrival"
                    ]
                )

            continue

        # ----------------------------------------------------
        # Select highest-priority packet
        # ----------------------------------------------------

        best_index = -1
        best_priority = float("-inf")

        for (
            index,
            packet
        ) in enumerate(
            queue
        ):

            time_to_deadline = max(
                packet[
                    "deadline"
                ]
                - current_time,
                EPSILON
            )

            priority = (
                (
                    1.0
                    + alpha
                    * packet[
                        "risk"
                    ]
                )
                / time_to_deadline
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
        # Deadline result
        # ----------------------------------------------------

        if (
            current_time
            <= packet[
                "deadline"
            ]
        ):

            results[
                packet[
                    "user"
                ]
            ][
                "on_time"
            ] += 1

        else:

            results[
                packet[
                    "user"
                ]
            ][
                "missed"
            ] += 1

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
            results[
                user
            ][
                "total"
            ]
        )

        on_time = (
            results[
                user
            ][
                "on_time"
            ]
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
# Alpha sweep
# ============================================================

rows = []


print(
    "\n"
    + "=" * 90
)

print(
    "RISK-AWARE SCHEDULER ALPHA SWEEP"
)

print(
    "=" * 90
)


for alpha in ALPHA_VALUES:

    results = (
        run_simulation(
            alpha
        )
    )

    total_packets = sum(
        data[
            "total"
        ]
        for data
        in results.values()
    )

    total_missed = sum(
        data[
            "missed"
        ]
        for data
        in results.values()
    )

    miss_rate = (
        total_missed
        / total_packets
        if total_packets > 0
        else 0.0
    )

    fairness = (
        jain_fairness(
            results
        )
    )

    low_misses = (
        results[
            low_user
        ][
            "missed"
        ]
    )

    medium_misses = (
        results[
            medium_user
        ][
            "missed"
        ]
    )

    high_misses = (
        results[
            high_user
        ][
            "missed"
        ]
    )

    rows.append(
        {
            "alpha":
                alpha,

            "total_misses":
                total_missed,

            "miss_rate":
                miss_rate,

            "jain_fairness":
                fairness,

            "low_risk":
                risk_scores[
                    low_user
                ],

            "low_risk_misses":
                low_misses,

            "medium_risk":
                risk_scores[
                    medium_user
                ],

            "medium_risk_misses":
                medium_misses,

            "high_risk":
                risk_scores[
                    high_user
                ],

            "high_risk_misses":
                high_misses
        }
    )

    print(
        f"\nalpha = {alpha}"
    )

    print(
        f"  total misses: "
        f"{total_missed}"
    )

    print(
        f"  miss rate: "
        f"{miss_rate:.4f}"
    )

    print(
        f"  fairness: "
        f"{fairness:.4f}"
    )

    print(
        f"  LOW misses: "
        f"{low_misses}"
    )

    print(
        f"  MEDIUM misses: "
        f"{medium_misses}"
    )

    print(
        f"  HIGH misses: "
        f"{high_misses}"
    )


# ============================================================
# Save
# ============================================================

output = pd.DataFrame(
    rows
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


output.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n"
    + "=" * 90
)

print(
    "FULL RESULTS"
)

print(
    "=" * 90
)

print(
    output.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)