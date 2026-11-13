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
    "results/tables/questset_risk_model_stability.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0

N_REPEATS = 200

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# Current best 1-minute movement feature set
# ============================================================

FEATURES = [
    "1min_velocity_mean",
    "1min_velocity_std",
    "1min_velocity_p95",
    "1min_angular_velocity_mean",
    "1min_angular_velocity_std",
    "1min_angular_velocity_p95",
    "1min_net_displacement"
]


# ============================================================
# Model
# ============================================================

def make_model():

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
                    C=0.1,
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

    X = df[FEATURES]

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(int).to_numpy()

    groups = (
        df["participant_id"]
        .to_numpy()
    )

    print(
        "=" * 80
    )

    print(
        "1-MINUTE RISK MODEL STABILITY ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nSessions:",
        len(df)
    )

    print(
        "Participants:",
        df["participant_id"].nunique()
    )

    print(
        "High-risk sessions:",
        int(y.sum())
    )

    print(
        "Prevalence:",
        round(y.mean(), 3)
    )

    splitter = GroupShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = []

    skipped = 0

    for repeat, (
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

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ROC-AUC cannot be computed if a test split
        # contains only one class.
        if (
            len(np.unique(y_train)) < 2
            or len(np.unique(y_test)) < 2
        ):
            skipped += 1
            continue

        model = make_model()

        model.fit(
            X.iloc[train_idx],
            y_train
        )

        prob = model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

        pred = (
            prob >= 0.5
        ).astype(int)

        results.append(
            {
                "repeat":
                    repeat,

                "n_train":
                    len(train_idx),

                "n_test":
                    len(test_idx),

                "test_high_risk":
                    int(y_test.sum()),

                "test_prevalence":
                    y_test.mean(),

                "roc_auc":
                    roc_auc_score(
                        y_test,
                        prob
                    ),

                "pr_auc":
                    average_precision_score(
                        y_test,
                        prob
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        pred
                    ),

                "precision":
                    precision_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y_test,
                        pred,
                        zero_division=0
                    )
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nValid repeated splits:",
        len(results_df)
    )

    print(
        "Skipped splits:",
        skipped
    )

    # ========================================================
    # Summary
    # ========================================================

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1"
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "METRIC STABILITY"
    )

    print(
        "=" * 80
    )

    summary_rows = []

    for metric in metric_columns:

        values = (
            results_df[metric]
            .dropna()
            .to_numpy()
        )

        row = {
            "metric":
                metric,

            "mean":
                np.mean(values),

            "std":
                np.std(values),

            "median":
                np.median(values),

            "p05":
                np.percentile(
                    values,
                    5
                ),

            "p25":
                np.percentile(
                    values,
                    25
                ),

            "p75":
                np.percentile(
                    values,
                    75
                ),

            "p95":
                np.percentile(
                    values,
                    95
                )
        }

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Additional interpretation statistics
    # ========================================================

    print(
        "\nFraction of splits with ROC-AUC > 0.50:"
    )

    print(
        round(
            (
                results_df[
                    "roc_auc"
                ] > 0.50
            ).mean(),
            3
        )
    )

    print(
        "\nFraction of splits with ROC-AUC >= 0.60:"
    )

    print(
        round(
            (
                results_df[
                    "roc_auc"
                ] >= 0.60
            ).mean(),
            3
        )
    )

    print(
        "\nFraction of splits with balanced accuracy > 0.50:"
    )

    print(
        round(
            (
                results_df[
                    "balanced_accuracy"
                ] > 0.50
            ).mean(),
            3
        )
    )

    print(
        "\nMean PR-AUC minus mean test prevalence:"
    )

    print(
        round(
            results_df[
                "pr_auc"
            ].mean()
            -
            results_df[
                "test_prevalence"
            ].mean(),
            3
        )
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()