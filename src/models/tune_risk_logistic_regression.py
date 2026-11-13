import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

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
    "results/tables/questset_logistic_c_tuning.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
]

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# Final compact 1-minute movement feature set
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

def build_model(c_value):

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
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )


# ============================================================
# Cross-validation
# ============================================================

def evaluate_c(
    df,
    c_value
):

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

    all_true = []
    all_prob = []
    all_pred = []

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups
    ):

        model = build_model(
            c_value
        )

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ]
        )

        prob = model.predict_proba(
            X.iloc[
                test_idx
            ]
        )[:, 1]

        pred = (
            prob >= 0.5
        ).astype(int)

        all_true.extend(
            y[
                test_idx
            ]
        )

        all_prob.extend(
            prob
        )

        all_pred.extend(
            pred
        )

    all_true = np.asarray(
        all_true
    )

    all_prob = np.asarray(
        all_prob
    )

    all_pred = np.asarray(
        all_pred
    )

    return {
        "C":
            c_value,

        "roc_auc":
            roc_auc_score(
                all_true,
                all_prob
            ),

        "pr_auc":
            average_precision_score(
                all_true,
                all_prob
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                all_true,
                all_pred
            ),

        "precision":
            precision_score(
                all_true,
                all_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                all_true,
                all_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                all_true,
                all_pred,
                zero_division=0
            )
    }


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    df = (
        df[
            df["window"]
            == "1min"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "=" * 80
    )

    print(
        "LOGISTIC REGRESSION REGULARIZATION TUNING"
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
        df[
            "participant_id"
        ].nunique()
    )

    print(
        "High-risk sessions:",
        int(
            (
                df[
                    "post_ssq_total"
                ]
                > RISK_THRESHOLD
            ).sum()
        )
    )

    results = []

    for c_value in C_VALUES:

        metrics = evaluate_c(
            df,
            c_value
        )

        results.append(
            metrics
        )

        print(
            f"\nC = {c_value}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.3f}"
        )

        print(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.3f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.3f}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.3f}"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Rank primarily by ROC-AUC, then PR-AUC
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "roc_auc",
                "pr_auc"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RANKED RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        results_df
        .round(3)
        .to_string(
            index=False
        )
    )

    best = (
        results_df
        .iloc[0]
    )

    print(
        "\nBest C:",
        best["C"]
    )

    print(
        "Best ROC-AUC:",
        round(
            best["roc_auc"],
            3
        )
    )

    print(
        "Best PR-AUC:",
        round(
            best["pr_auc"],
            3
        )
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
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()