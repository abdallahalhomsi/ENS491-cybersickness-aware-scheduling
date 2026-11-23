import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    "results/tables/questset_acceleration_context_comparison.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
MODEL_C = 0.1
N_SPLITS = 5
RANDOM_STATE = 42

DECISION_THRESHOLD = 0.525


# ============================================================
# Feature sets
# ============================================================

ACCELERATION_STD = [
    "1min_acceleration_std"
]


FEATURE_SETS = {

    "acceleration_std_only":
        ACCELERATION_STD,

    "plus_pre_ssq":
        ACCELERATION_STD
        + [
            "pre_ssq_total"
        ],

    "plus_game":
        ACCELERATION_STD
        + [
            "game_name"
        ],

    "plus_session":
        ACCELERATION_STD
        + [
            "session_number"
        ],

    "plus_pre_ssq_game":
        ACCELERATION_STD
        + [
            "pre_ssq_total",
            "game_name"
        ],

    "plus_pre_ssq_session":
        ACCELERATION_STD
        + [
            "pre_ssq_total",
            "session_number"
        ],

    "plus_game_session":
        ACCELERATION_STD
        + [
            "game_name",
            "session_number"
        ],

    "plus_pre_ssq_game_session":
        ACCELERATION_STD
        + [
            "pre_ssq_total",
            "game_name",
            "session_number"
        ]
}


# ============================================================
# Build preprocessing + model pipeline
# ============================================================

def build_model(
    df,
    features
):

    numeric_features = [
        feature
        for feature in features
        if pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    categorical_features = [
        feature
        for feature in features
        if feature not in numeric_features
    ]

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
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
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocess = ColumnTransformer(
        transformers
    )

    model = LogisticRegression(
        C=MODEL_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    return Pipeline(
        [
            (
                "preprocess",
                preprocess
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# Evaluate one feature set
# ============================================================

def evaluate_feature_set(
    df,
    feature_set_name,
    features
):

    X = df[
        features
    ]

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(
        int
    ).to_numpy()

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
            df,
            features
        )

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ]
        )

        prob = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        pred = (
            prob
            >= DECISION_THRESHOLD
        ).astype(
            int
        )

        probabilities[
            test_idx
        ] = prob

        predictions[
            test_idx
        ] = pred

    return {

        "feature_set":
            feature_set_name,

        "n_features":
            len(
                features
            ),

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
        "ACCELERATION + CONTEXT FEATURE COMPARISON"
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
        feature_set_name,
        features
    ) in FEATURE_SETS.items():

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:

            print(
                "Skipping",
                feature_set_name,
                "- missing:",
                missing
            )

            continue

        result = (
            evaluate_feature_set(
                df,
                feature_set_name,
                features
            )
        )

        results.append(
            result
        )

        print(
            feature_set_name
        )

        print(
            "  ROC-AUC:",
            round(
                result[
                    "roc_auc"
                ],
                3
            )
        )

        print(
            "  PR-AUC:",
            round(
                result[
                    "pr_auc"
                ],
                3
            )
        )

        print(
            "  Precision:",
            round(
                result[
                    "precision"
                ],
                3
            )
        )

        print(
            "  Recall:",
            round(
                result[
                    "recall"
                ],
                3
            )
        )

        print(
            "  F1:",
            round(
                result[
                    "f1"
                ],
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

    results_df = (
        results_df
        .sort_values(
            [
                "precision",
                "recall",
                "roc_auc"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
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

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()