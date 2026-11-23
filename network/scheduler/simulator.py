import json
import random

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RISK_SCORES_PATH = (
    SCRIPT_DIR
    / "risk_scores.json"
)


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42

random.seed(
    RANDOM_SEED
)


# ============================================================
# 1. LOAD ML RISK SCORES
# ============================================================

with open(
    RISK_SCORES_PATH,
    "r"
) as f:

    risk_scores = json.load(
        f
    )


if len(risk_scores) == 0:

    raise ValueError(
        "risk_scores.json is empty."
    )


print(
    f"Loaded {len(risk_scores)} "
    f"real ML risk scores."
)


# ============================================================
# 2. SIMULATION PARAMETERS
# ============================================================

NUM_FRAMES_PER_USER = 1000

FRAME_INTERVAL_MS = 14.0

FRAME_DEADLINE_MS = 30.0

SERVICE_TIME_MS = 4.72

ALPHA = 1.0

EPSILON = 0.001


# Realistic impairments from NeSt-VR
PACKET_LOSS_RATE = 0.01

JITTER_MS = 2.0


# ============================================================
# 3. GENERATE VR TRAFFIC
# ============================================================

packets = []


for (
    user,
    risk
) in risk_scores.items():

    for i in range(
        NUM_FRAMES_PER_USER
    ):

        base_arrival = (
            i
            * FRAME_INTERVAL_MS
        )

        # ----------------------------------------------------
        # Apply jitter
        # ----------------------------------------------------

        arrival_time = (
            base_arrival
            + random.uniform(
                -JITTER_MS,
                JITTER_MS
            )
        )

        # ----------------------------------------------------
        # Apply packet loss
        # ----------------------------------------------------

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
                    f"{user}_{i}",

                "user":
                    user,

                "arrival":
                    arrival_time,

                "deadline":
                    deadline,

                "risk":
                    float(
                        risk
                    )
            }
        )


# Sort packets chronologically.
packets.sort(
    key=lambda x: (
        x["arrival"],
        x["id"]
    )
)


print(
    f"Generated {len(packets)} "
    f"packets for simulation."
)


# ============================================================
# 4. SCHEDULERS
# ============================================================

def run_simulation(
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

        # ----------------------------------------------------
        # Add all packets that have arrived.
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
        # If queue is empty, move to next packet arrival.
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
        # Select packet according to scheduler.
        # ----------------------------------------------------

        best_packet = None

        best_priority = (
            -np.inf
        )

        best_index = -1

        for (
            i,
            q_packet
        ) in enumerate(
            queue
        ):

            time_to_deadline = max(
                q_packet[
                    "deadline"
                ]
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
                        * q_packet[
                            "risk"
                        ]
                    )
                    / time_to_deadline
                )

            else:

                raise ValueError(
                    "Unknown scheduler: "
                    f"{scheduler_type}"
                )

            if (
                priority
                > best_priority
            ):

                best_priority = (
                    priority
                )

                best_packet = (
                    q_packet
                )

                best_index = i

        # ----------------------------------------------------
        # Serve selected packet.
        # ----------------------------------------------------

        queue.pop(
            best_index
        )

        if (
            current_time
            <= best_packet[
                "deadline"
            ]
        ):

            results[
                best_packet[
                    "user"
                ]
            ][
                "on_time"
            ] += 1

        else:

            results[
                best_packet[
                    "user"
                ]
            ][
                "missed"
            ] += 1

        # ----------------------------------------------------
        # Advance time.
        # ----------------------------------------------------

        current_time += (
            SERVICE_TIME_MS
        )

    return results


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def calculate_jain_fairness(
    results
):

    x_i = []

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

        if total > 0:

            success_rate = (
                on_time
                / total
            )

        else:

            success_rate = 0.0

        x_i.append(
            success_rate
        )

    n = len(
        x_i
    )

    numerator = (
        sum(
            x_i
        )
        ** 2
    )

    denominator = (
        n
        * sum(
            x ** 2
            for x
            in x_i
        )
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_summary(
    results
):

    total_packets = sum(
        data[
            "total"
        ]
        for data
        in results.values()
    )

    total_on_time = sum(
        data[
            "on_time"
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

    if total_packets > 0:

        on_time_rate = (
            total_on_time
            / total_packets
        )

        miss_rate = (
            total_missed
            / total_packets
        )

    else:

        on_time_rate = 0.0
        miss_rate = 0.0

    return {
        "total_packets":
            total_packets,

        "on_time":
            total_on_time,

        "missed":
            total_missed,

        "on_time_rate":
            on_time_rate,

        "miss_rate":
            miss_rate,

        "jain_fairness":
            calculate_jain_fairness(
                results
            )
    }


# ============================================================
# 6. RUN EXPERIMENTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Running Baseline: EDF"
)

print(
    "=" * 70
)


edf_results = (
    run_simulation(
        "EDF"
    )
)

edf_summary = (
    calculate_summary(
        edf_results
    )
)


print(
    "Total packets:",
    edf_summary[
        "total_packets"
    ]
)

print(
    "Deadline misses:",
    edf_summary[
        "missed"
    ]
)

print(
    "On-time packets:",
    edf_summary[
        "on_time"
    ]
)

print(
    "Miss rate:",
    round(
        edf_summary[
            "miss_rate"
        ],
        4
    )
)

print(
    "Jain Fairness Index:",
    round(
        edf_summary[
            "jain_fairness"
        ],
        4
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "Running Treatment: "
    "Risk-Aware"
)

print(
    "=" * 70
)


risk_results = (
    run_simulation(
        "RISK_AWARE"
    )
)

risk_summary = (
    calculate_summary(
        risk_results
    )
)


print(
    "Total packets:",
    risk_summary[
        "total_packets"
    ]
)

print(
    "Deadline misses:",
    risk_summary[
        "missed"
    ]
)

print(
    "On-time packets:",
    risk_summary[
        "on_time"
    ]
)

print(
    "Miss rate:",
    round(
        risk_summary[
            "miss_rate"
        ],
        4
    )
)

print(
    "Jain Fairness Index:",
    round(
        risk_summary[
            "jain_fairness"
        ],
        4
    )
)


# ============================================================
# 7. HIGH-RISK USER ANALYSIS
# ============================================================

# Plot only the 10 highest-risk sessions.
# Plotting all 120 sessions would be unreadable.

top_users = sorted(
    risk_scores.keys(),
    key=lambda user:
        risk_scores[
            user
        ],
    reverse=True
)[:10]


print(
    "\n"
    + "=" * 70
)

print(
    "Top 10 Highest-Risk Sessions"
)

print(
    "=" * 70
)


for user in top_users:

    risk = (
        risk_scores[
            user
        ]
    )

    edf_missed = (
        edf_results[
            user
        ][
            "missed"
        ]
    )

    risk_missed = (
        risk_results[
            user
        ][
            "missed"
        ]
    )

    print(
        f"{user}"
        f" | risk={risk:.3f}"
        f" | EDF misses={edf_missed}"
        f" | Risk-Aware misses={risk_missed}"
    )


# ============================================================
# 8. PLOT RESULTS
# ============================================================

labels = [
    (
        f"{user}\n"
        f"R={risk_scores[user]:.2f}"
    )
    for user
    in top_users
]


edf_misses = [
    edf_results[
        user
    ][
        "missed"
    ]
    for user
    in top_users
]


risk_misses = [
    risk_results[
        user
    ][
        "missed"
    ]
    for user
    in top_users
]


x = np.arange(
    len(
        top_users
    )
)

width = 0.35


fig, ax = (
    plt.subplots(
        figsize=(
            14,
            7
        )
    )
)


rects1 = ax.bar(
    x
    - width / 2,
    edf_misses,
    width,
    label="Standard EDF"
)


rects2 = ax.bar(
    x
    + width / 2,
    risk_misses,
    width,
    label=(
        f"Risk-Aware "
        f"(alpha={ALPHA})"
    )
)


ax.set_ylabel(
    "Deadline Misses "
    "(Lower is Better)"
)

ax.set_title(
    "EDF vs Risk-Aware Scheduler\n"
    "Low, Medium, and High Predicted Cybersickness-Risk Sessions"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right"
)

ax.legend()


ax.bar_label(
    rects1,
    padding=3
)

ax.bar_label(
    rects2,
    padding=3
)


plt.tight_layout()

plt.show()