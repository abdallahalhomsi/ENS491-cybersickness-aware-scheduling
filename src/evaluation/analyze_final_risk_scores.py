import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


DATA_PATH = Path(
    "results/tables/questset_final_risk_scores.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_final_risk_score_analysis.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    y = df["high_risk_true"].to_numpy()

    scores = df["risk_score"].to_numpy()

    print("=" * 80)
    print("FINAL RISK SCORE DIAGNOSTIC")
    print("=" * 80)

    print("\nSessions:", len(df))
    print("True high-risk:", int(y.sum()))
    print("Prevalence:", round(y.mean(), 3))

    print("\nROC-AUC:")
    print(round(roc_auc_score(y, scores), 3))

    print("\nPR-AUC:")
    print(round(average_precision_score(y, scores), 3))

    print("\nBrier score:")
    print(round(brier_score_loss(y, scores), 3))

    # ========================================================
    # True-risk score distributions
    # ========================================================

    print("\nRisk-score distribution by true class:")

    summary = (
        df.groupby("high_risk_true")["risk_score"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .round(3)
    )

    print(summary)

    # ========================================================
    # Ranking / top-k analysis
    # ========================================================

    ranked = (
        df.sort_values(
            "risk_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    top_k_values = [
        5,
        10,
        15,
        20,
        25,
        30,
        40
    ]

    top_k_results = []

    print("\n" + "=" * 80)
    print("TOP-K RANKING PERFORMANCE")
    print("=" * 80)

    total_positive = y.sum()

    for k in top_k_values:

        top = ranked.head(k)

        tp = int(
            top["high_risk_true"].sum()
        )

        precision_at_k = (
            tp / k
        )

        recall_at_k = (
            tp / total_positive
        )

        lift = (
            precision_at_k
            / y.mean()
        )

        top_k_results.append(
            {
                "k":
                    k,

                "true_high_risk":
                    tp,

                "precision_at_k":
                    precision_at_k,

                "recall_at_k":
                    recall_at_k,

                "lift_vs_prevalence":
                    lift
            }
        )

        print(
            f"Top {k:>2}: "
            f"TP={tp:>2} | "
            f"Precision={precision_at_k:.3f} | "
            f"Recall={recall_at_k:.3f} | "
            f"Lift={lift:.2f}x"
        )

    # ========================================================
    # Precision-recall threshold analysis
    # ========================================================

    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            scores
        )
    )

    threshold_rows = []

    for i, threshold in enumerate(
        thresholds
    ):

        p = precision[i]
        r = recall[i]

        if (
            p + r
        ) == 0:
            f1 = 0
        else:
            f1 = (
                2
                * p
                * r
                / (
                    p + r
                )
            )

        threshold_rows.append(
            {
                "threshold":
                    threshold,

                "precision":
                    p,

                "recall":
                    r,

                "f1":
                    f1
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    best_f1 = (
        threshold_df
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 80)
    print("BEST THRESHOLD BY F1")
    print("=" * 80)

    print(
        "Threshold:",
        round(
            best_f1[
                "threshold"
            ],
            3
        )
    )

    print(
        "Precision:",
        round(
            best_f1[
                "precision"
            ],
            3
        )
    )

    print(
        "Recall:",
        round(
            best_f1[
                "recall"
            ],
            3
        )
    )

    print(
        "F1:",
        round(
            best_f1[
                "f1"
            ],
            3
        )
    )

    # ========================================================
    # Thresholds maintaining useful recall
    # ========================================================

    print("\nThreshold options with recall >= 0.70:")

    useful = (
        threshold_df[
            threshold_df[
                "recall"
            ] >= 0.70
        ]
        .sort_values(
            [
                "precision",
                "f1"
            ],
            ascending=False
        )
        .head(10)
    )

    print(
        useful
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Highest-ranked true positives
    # ========================================================

    true_positive_ranked = (
        ranked[
            ranked[
                "high_risk_true"
            ] == 1
        ]
        .copy()
    )

    true_positive_ranked[
        "rank"
    ] = (
        true_positive_ranked.index
        + 1
    )

    print("\nRanks of true high-risk sessions:")

    print(
        true_positive_ranked[
            [
                "rank",
                "participant_id",
                "game_name",
                "post_ssq_total",
                "risk_score"
            ]
        ]
        .head(22)
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save combined analysis
    # ========================================================

    top_k_df = pd.DataFrame(
        top_k_results
    )

    top_k_df[
        "analysis_type"
    ] = "top_k"

    threshold_df[
        "analysis_type"
    ] = "threshold"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined = pd.concat(
        [
            top_k_df,
            threshold_df
        ],
        ignore_index=True,
        sort=False
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()