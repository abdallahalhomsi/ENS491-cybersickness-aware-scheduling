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
    "results/tables/questset_acceleration_feature_ablation.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
MODEL_C = 0.1
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Feature sets
# ============================================================

FEATURE_SETS = {

    "mean_only": [
        "1min_acceleration_mean"
    ],

    "std_only": [
        "1min_acceleration_std"
    ],

    "p95_only": [
        "1min_acceleration_p95"
    ],

    "mean_plus_std": [
        "1min_acceleration_mean",
        "1min_acceleration_std"
    ],

    "mean_plus_p95": [
        "1min_acceleration_mean",
        "1min_acceleration_p95"
    ],

    "std_plus_p95": [
        "1min_acceleration_std",
        "1min_acceleration_p95"
    ],

    "all_three": [
        "1min_acceleration_mean",
        "1min_acceleration_std",
        "1min_acceleration_p95"
    ]
}


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
# Evaluate feature set
# ============================================================

def evaluate(
    df,
    name,
    features
):

    X = df[
        features
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

    predictions = np.zeros(
        len(df),
        dtype=int
    )

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

        prob = model.predict_proba(
            X.iloc[
                test_idx
            ]
        )[:, 1]

        pred = (
            prob >= 0.5
        ).astype(int)

        probabilities[
            test_idx
        ] = prob

        predictions[
            test_idx
        ] = pred

    return {

        "feature_set":
            name,

        "n_features":
            len(features),

        "roc_auc":
            roc_auc_score(
                y,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y,
                probabilities
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                predictions
            ),

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                predictions,
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
            df["window"] == "1min"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "=" * 80
    )

    print(
        "ACCELERATION FEATURE ABLATION"
    )

    print(
        "=" * 80
    )

    print(
        "Sessions:",
        len(df)
    )

    print(
        "Participants:",
        df[
            "participant_id"
        ].nunique()
    )

    print()

    results = []

    for (
        name,
        features
    ) in FEATURE_SETS.items():

        result = evaluate(
            df,
            name,
            features
        )

        results.append(
            result
        )

        print(
            name
        )

        print(
            "  features:",
            result["n_features"]
        )

        print(
            "  ROC-AUC:",
            round(
                result["roc_auc"],
                3
            )
        )

        print(
            "  PR-AUC:",
            round(
                result["pr_auc"],
                3
            )
        )

        print(
            "  Balanced accuracy:",
            round(
                result[
                    "balanced_accuracy"
                ],
                3
            )
        )

        print(
            "  Precision:",
            round(
                result["precision"],
                3
            )
        )

        print(
            "  Recall:",
            round(
                result["recall"],
                3
            )
        )

        print(
            "  F1:",
            round(
                result["f1"],
                3
            )
        )

        print()

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "roc_auc",
            ascending=False
        )
        .reset_index(drop=True)
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
        "RANKING BY ROC-AUC"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
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