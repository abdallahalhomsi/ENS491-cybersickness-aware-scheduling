import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

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
    "results/tables/questset_acceleration_model_comparison.csv"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Feature sets
# ============================================================

FEATURE_SETS = {

    "std_only": [
        "1min_acceleration_std"
    ],

    "all_acceleration": [
        "1min_acceleration_mean",
        "1min_acceleration_std",
        "1min_acceleration_p95"
    ]
}


# ============================================================
# Models
# ============================================================

def get_models():

    models = {

        "logistic_regression":

            Pipeline(
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
            ),

        "random_forest":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=3,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                    )
                ]
            ),

        "gradient_boosting":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=80,
                            learning_rate=0.03,
                            max_depth=2,
                            min_samples_leaf=5,
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

        "hist_gradient_boosting":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=100,
                            max_leaf_nodes=7,
                            min_samples_leaf=10,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            )
    }

    return models


# ============================================================
# Find exploratory best threshold
# ============================================================

def find_best_threshold(
    y,
    probabilities
):

    thresholds = np.arange(
        0.20,
        0.81,
        0.025
    )

    rows = []

    for threshold in thresholds:

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

        rows.append(
            {
                "threshold":
                    threshold,

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

    threshold_df = pd.DataFrame(
        rows
    )

    best = (
        threshold_df
        .sort_values(
            [
                "f1",
                "balanced_accuracy"
            ],
            ascending=False
        )
        .iloc[0]
    )

    return best


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    df,
    feature_set_name,
    features,
    model_name,
    model
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

    # --------------------------------------------------------
    # Participant-independent OOF predictions
    # --------------------------------------------------------

    for (
        train_idx,
        test_idx
    ) in cv.split(
        X,
        y,
        groups
    ):

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

    # --------------------------------------------------------
    # Default threshold = 0.5
    # --------------------------------------------------------

    default_predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Exploratory threshold optimization
    # --------------------------------------------------------

    best_threshold = (
        find_best_threshold(
            y,
            probabilities
        )
    )

    return {

        "feature_set":
            feature_set_name,

        "model":
            model_name,

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

        "default_precision":
            precision_score(
                y,
                default_predictions,
                zero_division=0
            ),

        "default_recall":
            recall_score(
                y,
                default_predictions,
                zero_division=0
            ),

        "default_f1":
            f1_score(
                y,
                default_predictions,
                zero_division=0
            ),

        "default_balanced_accuracy":
            balanced_accuracy_score(
                y,
                default_predictions
            ),

        "best_threshold":
            best_threshold[
                "threshold"
            ],

        "best_precision":
            best_threshold[
                "precision"
            ],

        "best_recall":
            best_threshold[
                "recall"
            ],

        "best_f1":
            best_threshold[
                "f1"
            ],

        "best_balanced_accuracy":
            best_threshold[
                "balanced_accuracy"
            ]
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
        "=" * 100
    )

    print(
        "ACCELERATION MODEL COMPARISON"
    )

    print(
        "=" * 100
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
                df["post_ssq_total"]
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

        print(
            "\n"
            + "=" * 100
        )

        print(
            "FEATURE SET:",
            feature_set_name
        )

        print(
            "=" * 100
        )

        models = get_models()

        for (
            model_name,
            model
        ) in models.items():

            result = evaluate_model(
                df,
                feature_set_name,
                features,
                model_name,
                model
            )

            results.append(
                result
            )

            print(
                "\n",
                model_name,
                sep=""
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
                "  Default threshold 0.5:"
            )

            print(
                "    Precision:",
                round(
                    result[
                        "default_precision"
                    ],
                    3
                )
            )

            print(
                "    Recall:",
                round(
                    result[
                        "default_recall"
                    ],
                    3
                )
            )

            print(
                "    F1:",
                round(
                    result[
                        "default_f1"
                    ],
                    3
                )
            )

            print(
                "  Exploratory best threshold:"
            )

            print(
                "    Threshold:",
                round(
                    result[
                        "best_threshold"
                    ],
                    3
                )
            )

            print(
                "    Precision:",
                round(
                    result[
                        "best_precision"
                    ],
                    3
                )
            )

            print(
                "    Recall:",
                round(
                    result[
                        "best_recall"
                    ],
                    3
                )
            )

            print(
                "    F1:",
                round(
                    result[
                        "best_f1"
                    ],
                    3
                )
            )

    # ========================================================
    # Results table
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            [
                "pr_auc",
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
        "\n"
        + "=" * 100
    )

    print(
        "FINAL RANKING BY PR-AUC"
    )

    print(
        "=" * 100
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

    print()

    print(
        "NOTE:"
    )

    print(
        "Best-threshold metrics are exploratory because the threshold "
        "was selected using the same out-of-fold predictions."
    )

    print(
        "ROC-AUC and PR-AUC should be used as the primary model-comparison metrics."
    )


if __name__ == "__main__":
    main()