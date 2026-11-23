import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_acceleration_stability.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
MODEL_C = 0.1
RANDOM_STATE = 42

N_REPEATS = 200
TEST_SIZE = 0.20


# ============================================================
# Features
# ============================================================

FEATURES = [
    "1min_acceleration_mean",
    "1min_acceleration_std",
    "1min_acceleration_p95"
]


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

    splitter = GroupShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = []

    skipped = 0

    for split_id, (
        train_idx,
        test_idx
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # Need both classes in train and test.
        if (
            len(
                np.unique(
                    y_train
                )
            ) < 2
            or
            len(
                np.unique(
                    y_test
                )
            ) < 2
        ):
            skipped += 1
            continue

        model = build_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            y_train
        )

        probabilities = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        predictions = (
            probabilities
            >= 0.5
        ).astype(int)

        prevalence = float(
            y_test.mean()
        )

        results.append(
            {
                "split":
                    split_id,

                "n_test":
                    len(test_idx),

                "test_prevalence":
                    prevalence,

                "roc_auc":
                    roc_auc_score(
                        y_test,
                        probabilities
                    ),

                "pr_auc":
                    average_precision_score(
                        y_test,
                        probabilities
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        predictions
                    ),

                "precision":
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )
            }
        )

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "=" * 80
    )

    print(
        "ACCELERATION MODEL STABILITY"
    )

    print(
        "=" * 80
    )

    print(
        "Features:",
        FEATURES
    )

    print(
        "Total requested splits:",
        N_REPEATS
    )

    print(
        "Valid splits:",
        len(results_df)
    )

    print(
        "Skipped splits:",
        skipped
    )

    print()

    metrics = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1"
    ]

    for metric in metrics:

        values = (
            results_df[
                metric
            ]
        )

        print(
            metric
        )

        print(
            "  mean:",
            round(
                values.mean(),
                3
            )
        )

        print(
            "  std:",
            round(
                values.std(),
                3
            )
        )

        print(
            "  median:",
            round(
                values.median(),
                3
            )
        )

        print(
            "  p05:",
            round(
                values.quantile(
                    0.05
                ),
                3
            )
        )

        print(
            "  p25:",
            round(
                values.quantile(
                    0.25
                ),
                3
            )
        )

        print(
            "  p75:",
            round(
                values.quantile(
                    0.75
                ),
                3
            )
        )

        print(
            "  p95:",
            round(
                values.quantile(
                    0.95
                ),
                3
            )
        )

        print()

    roc = (
        results_df[
            "roc_auc"
        ]
    )

    bal = (
        results_df[
            "balanced_accuracy"
        ]
    )

    pr_advantage = (
        results_df[
            "pr_auc"
        ]
        - results_df[
            "test_prevalence"
        ]
    )

    print(
        "ROC-AUC > 0.50:",
        round(
            100
            * (
                roc > 0.50
            ).mean(),
            1
        ),
        "%"
    )

    print(
        "ROC-AUC >= 0.60:",
        round(
            100
            * (
                roc >= 0.60
            ).mean(),
            1
        ),
        "%"
    )

    print(
        "Balanced accuracy > 0.50:",
        round(
            100
            * (
                bal > 0.50
            ).mean(),
            1
        ),
        "%"
    )

    print(
        "Mean PR-AUC minus prevalence:",
        round(
            pr_advantage.mean(),
            3
        )
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