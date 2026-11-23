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
    "results/tables/questset_acceleration_class_weight_tuning.csv"
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

DECISION_THRESHOLD = 0.525


# ============================================================
# Class-weight settings
# ============================================================

CLASS_WEIGHTS = {

    "none":
        None,

    "positive_1.25":
        {
            0: 1.0,
            1: 1.25
        },

    "positive_1.5":
        {
            0: 1.0,
            1: 1.5
        },

    "positive_2":
        {
            0: 1.0,
            1: 2.0
        },

    "positive_2.5":
        {
            0: 1.0,
            1: 2.5
        },

    "positive_3":
        {
            0: 1.0,
            1: 3.0
        },

    "balanced":
        "balanced"
}


# ============================================================
# Model
# ============================================================

def build_model(
    class_weight
):

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
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )


# ============================================================
# Evaluate one class-weight setting
# ============================================================

def evaluate(
    df,
    weight_name,
    class_weight
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

        model = build_model(
            class_weight
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
            prob
            >= DECISION_THRESHOLD
        ).astype(int)

        probabilities[
            test_idx
        ] = prob

        predictions[
            test_idx
        ] = pred

    return {

        "weight_name":
            weight_name,

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
            ),

        "predicted_positive":
            int(
                predictions.sum()
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
        "=" * 90
    )

    print(
        "ACCELERATION STD CLASS-WEIGHT TUNING"
    )

    print(
        "=" * 90
    )

    print(
        "Decision threshold:",
        DECISION_THRESHOLD
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

    print()

    results = []

    for (
        weight_name,
        class_weight
    ) in CLASS_WEIGHTS.items():

        result = evaluate(
            df,
            weight_name,
            class_weight
        )

        results.append(
            result
        )

        print(
            weight_name
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
            "  Predicted positive:",
            result[
                "predicted_positive"
            ]
        )

        print()

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Ranking: prioritize precision, but keep recall useful
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "precision",
                "recall",
                "f1"
            ],
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
        "=" * 90
    )

    print(
        "RANKING BY PRECISION"
    )

    print(
        "=" * 90
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best precision while recall >= 0.70
    # --------------------------------------------------------

    candidates = results_df[
        results_df[
            "recall"
        ]
        >= 0.70
    ].copy()

    if len(candidates) > 0:

        best = (
            candidates
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
            + "=" * 90
        )

        print(
            "BEST PRECISION WITH RECALL >= 0.70"
        )

        print(
            "=" * 90
        )

        print(
            "Weight:",
            best[
                "weight_name"
            ]
        )

        print(
            "Precision:",
            round(
                best[
                    "precision"
                ],
                3
            )
        )

        print(
            "Recall:",
            round(
                best[
                    "recall"
                ],
                3
            )
        )

        print(
            "F1:",
            round(
                best[
                    "f1"
                ],
                3
            )
        )

        print(
            "Balanced accuracy:",
            round(
                best[
                    "balanced_accuracy"
                ],
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
