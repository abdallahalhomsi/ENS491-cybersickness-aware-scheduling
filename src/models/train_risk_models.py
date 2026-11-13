import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_risk_classification_results.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# Exploratory risk threshold.
# This is NOT being claimed as a clinical SSQ threshold.
RISK_THRESHOLD = 30.0


# ============================================================
# Context
# ============================================================

CONTEXT_FEATURES = [
    "game_name",
    "session_number",
    "group",
    "order",
    "age",
    "gender",
    "vr_experience",
    "gaming_experience",
    "pre_ssq_total"
]


# ============================================================
# Remove unusable columns
# ============================================================

def clean_features(df, columns):

    result = []

    for col in columns:

        if col not in df.columns:
            continue

        if df[col].isna().all():
            continue

        values = df[col].dropna()

        if (
            len(values) > 0
            and values.nunique() <= 1
        ):
            continue

        result.append(col)

    return result


# ============================================================
# Feature sets
# ============================================================

def get_feature_sets(df):

    context = clean_features(
        df,
        CONTEXT_FEATURES
    )

    movement = [
        c for c in df.columns
        if (
            c.startswith("1min_")
            or c.startswith("3min_")
            or c.startswith("5min_")
        )
    ]

    movement = clean_features(
        df,
        movement
    )

    temporal = [
        c for c in movement
        if (
            "_slope" in c
            or "_early_late_change" in c
            or "_temporal_range" in c
            or "_temporal_std" in c
            or "_max_block" in c
        )
    ]

    base_movement = [
        c for c in movement
        if c not in temporal
    ]

    return {
        "pre_ssq_only": [
            "pre_ssq_total"
        ],

        "context_only":
            context,

        "movement_only":
            base_movement,

        "movement_context":
            list(
                dict.fromkeys(
                    base_movement
                    + context
                )
            ),

        "movement_context_temporal":
            list(
                dict.fromkeys(
                    base_movement
                    + temporal
                    + context
                )
            )
    }


# ============================================================
# Pipeline
# ============================================================

def make_pipeline(
    df,
    features,
    model
):

    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    categorical = [
        c for c in features
        if c not in numeric
    ]

    numeric_pipe = Pipeline(
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

    categorical_pipe = Pipeline(
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

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipe,
                numeric
            ),
            (
                "categorical",
                categorical_pipe,
                categorical
            )
        ]
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
# Cross-validation
# ============================================================

def evaluate(
    df,
    features,
    model
):

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(int).to_numpy()

    groups = df[
        "participant_id"
    ].to_numpy()

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    all_true = []
    all_prob = []
    all_pred = []

    for train_idx, test_idx in cv.split(
        df,
        y,
        groups
    ):

        train = df.iloc[
            train_idx
        ]

        test = df.iloc[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        pipeline = make_pipeline(
            train,
            features,
            model
        )

        pipeline.fit(
            train[
                features
            ],
            y_train
        )

        prob = pipeline.predict_proba(
            test[
                features
            ]
        )[:, 1]

        pred = (
            prob >= 0.5
        ).astype(int)

        all_true.extend(
            y[test_idx]
        )

        all_prob.extend(
            prob
        )

        all_pred.extend(
            pred
        )

    all_true = np.array(
        all_true
    )

    all_prob = np.array(
        all_prob
    )

    all_pred = np.array(
        all_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        all_true,
        all_pred,
        labels=[0, 1]
    ).ravel()

    return {
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
            ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    results = []

    print(
        "=" * 80
    )
    print(
        "CYBERSICKNESS RISK CLASSIFICATION"
    )
    print(
        "=" * 80
    )

    print(
        "\nExploratory high-risk definition:"
    )

    print(
        f"post_ssq_total > {RISK_THRESHOLD}"
    )

    for window in [
        "1min",
        "3min",
        "5min"
    ]:

        temp = (
            df[
                df["window"]
                == window
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        y = (
            temp[
                "post_ssq_total"
            ]
            > RISK_THRESHOLD
        ).astype(int)

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"WINDOW: {window}"
        )

        print(
            "=" * 80
        )

        print(
            "Sessions:",
            len(temp)
        )

        print(
            "High risk:",
            int(y.sum())
        )

        print(
            "Not high risk:",
            int(
                len(y)
                - y.sum()
            )
        )

        feature_sets = get_feature_sets(
            temp
        )

        for (
            feature_set_name,
            features
        ) in feature_sets.items():

            if len(features) == 0:
                continue

            print(
                f"\nFeature set: "
                f"{feature_set_name} "
                f"({len(features)} features)"
            )

            models = {
                "Logistic Regression":
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=
                            RANDOM_STATE
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=
                            RANDOM_STATE,
                        n_jobs=-1
                    ),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=4,
                        random_state=
                            RANDOM_STATE
                    )
            }

            for (
                model_name,
                model
            ) in models.items():

                metrics = evaluate(
                    temp,
                    features,
                    model
                )

                result = {
                    "window":
                        window,

                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,

                    "n_features":
                        len(features),

                    **metrics
                }

                results.append(
                    result
                )

                print(
                    f"  {model_name:<22} "
                    f"ROC-AUC={metrics['roc_auc']:.3f} "
                    f"PR-AUC={metrics['pr_auc']:.3f} "
                    f"BalAcc={metrics['balanced_accuracy']:.3f} "
                    f"Recall={metrics['recall']:.3f} "
                    f"F1={metrics['f1']:.3f}"
                )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            [
                "window",
                "pr_auc"
            ],
            ascending=[
                True,
                False
            ]
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
        + "=" * 80
    )

    print(
        "BEST RISK MODELS BY WINDOW"
    )

    print(
        "=" * 80
    )

    for window in [
        "1min",
        "3min",
        "5min"
    ]:

        subset = results_df[
            results_df[
                "window"
            ] == window
        ]

        print(
            f"\n{window}"
        )

        print(
            subset[
                [
                    "feature_set",
                    "model",
                    "n_features",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "tp",
                    "fn",
                    "fp",
                    "tn"
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print(
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()