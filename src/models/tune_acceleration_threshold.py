import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_acceleration_threshold_sweep.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
MODEL_C = 0.1
N_SPLITS = 5
RANDOM_STATE = 42

FEATURES = [
    "1min_acceleration_std"
]

DECISION_THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.025
)


# ============================================================
# Model
# ============================================================

def build_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    C=MODEL_C,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    df = (
        df[
            df["window"] == "1min"
        ]
        .copy()
        .reset_index(drop=True)
    )

    X = df[
        FEATURES
    ]

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(int).to_numpy()

    groups = (
        df["participant_id"]
        .to_numpy()
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    probabilities = np.zeros(
        len(df)
    )

    # --------------------------------------------------------
    # Generate participant-independent out-of-fold probabilities
    # --------------------------------------------------------

    for (
        train_idx,
        test_idx
    ) in cv.split(
        X,
        y,
        groups
    ):

        model = build_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ]
        )

        probabilities[
            test_idx
        ] = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

    print(
        "=" * 85
    )

    print(
        "ACCELERATION STD DECISION-THRESHOLD SWEEP"
    )

    print(
        "=" * 85
    )

    print(
        "Sessions:",
        len(df)
    )

    print(
        "High-risk sessions:",
        int(
            y.sum()
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y,
                probabilities
            ),
            3
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y,
                probabilities
            ),
            3
        )
    )

    print()

    rows = []

    for threshold in DECISION_THRESHOLDS:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y,
                predictions
            )
        )

        predicted_positive = int(
            predictions.sum()
        )

        rows.append(
            {
                "threshold":
                    threshold,

                "predicted_positive":
                    predicted_positive,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "balanced_accuracy":
                    balanced_accuracy
            }
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save all thresholds
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Print full sweep
    # --------------------------------------------------------

    print(
        results.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x:
                        f"{x:.3f}",

                "precision":
                    lambda x:
                        f"{x:.3f}",

                "recall":
                    lambda x:
                        f"{x:.3f}",

                "f1":
                    lambda x:
                        f"{x:.3f}",

                "balanced_accuracy":
                    lambda x:
                        f"{x:.3f}"
            }
        )
    )

    # --------------------------------------------------------
    # Best threshold by F1
    # --------------------------------------------------------

    best_f1 = (
        results
        .sort_values(
            [
                "f1",
                "balanced_accuracy"
            ],
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\n"
        + "=" * 85
    )

    print(
        "BEST BY F1"
    )

    print(
        "=" * 85
    )

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.3f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.3f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.3f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.3f}"
    )

    print(
        f"Balanced accuracy: "
        f"{best_f1['balanced_accuracy']:.3f}"
    )

    # --------------------------------------------------------
    # Best threshold with recall >= 0.70
    # --------------------------------------------------------

    recall_candidates = results[
        results[
            "recall"
        ]
        >= 0.70
    ].copy()

    if len(
        recall_candidates
    ) > 0:

        best_precision_with_recall = (
            recall_candidates
            .sort_values(
                [
                    "precision",
                    "f1"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print(
            "\n"
            + "=" * 85
        )

        print(
            "BEST PRECISION WITH RECALL >= 0.70"
        )

        print(
            "=" * 85
        )

        print(
            f"Threshold: "
            f"{best_precision_with_recall['threshold']:.3f}"
        )

        print(
            f"Precision: "
            f"{best_precision_with_recall['precision']:.3f}"
        )

        print(
            f"Recall: "
            f"{best_precision_with_recall['recall']:.3f}"
        )

        print(
            f"F1: "
            f"{best_precision_with_recall['f1']:.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{best_precision_with_recall['balanced_accuracy']:.3f}"
        )

    # --------------------------------------------------------
    # Best threshold with recall >= 0.80
    # --------------------------------------------------------

    recall80_candidates = results[
        results[
            "recall"
        ]
        >= 0.80
    ].copy()

    if len(
        recall80_candidates
    ) > 0:

        best_precision_recall80 = (
            recall80_candidates
            .sort_values(
                [
                    "precision",
                    "f1"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print(
            "\n"
            + "=" * 85
        )

        print(
            "BEST PRECISION WITH RECALL >= 0.80"
        )

        print(
            "=" * 85
        )

        print(
            f"Threshold: "
            f"{best_precision_recall80['threshold']:.3f}"
        )

        print(
            f"Precision: "
            f"{best_precision_recall80['precision']:.3f}"
        )

        print(
            f"Recall: "
            f"{best_precision_recall80['recall']:.3f}"
        )

        print(
            f"F1: "
            f"{best_precision_recall80['f1']:.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{best_precision_recall80['balanced_accuracy']:.3f}"
        )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()