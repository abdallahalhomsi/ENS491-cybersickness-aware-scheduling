import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
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


DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_risk_threshold_sensitivity.csv"
)

RISK_THRESHOLDS = [
    20.0,
    30.0,
    40.0
]

WINDOWS = [
    "1min",
    "3min",
    "5min"
]

RANDOM_STATE = 42


def clean_features(df, columns):

    valid = []

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

        valid.append(col)

    return valid


def movement_features(df):

    features = [
        c for c in df.columns
        if (
            c.startswith("1min_")
            or c.startswith("3min_")
            or c.startswith("5min_")
        )
    ]

    # For this sensitivity test, use BASE movement only.
    # Temporal features will be analyzed separately.
    features = [
        c for c in features
        if not (
            "_slope" in c
            or "_early_late_change" in c
            or "_temporal_range" in c
            or "_temporal_std" in c
            or "_max_block" in c
        )
    ]

    return clean_features(
        df,
        features
    )


def make_pipeline():

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

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipe,
                slice(0, None)
            )
        ]
    )

    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    return Pipeline(
        [
            (
                "preprocess",
                preprocessing
            ),
            (
                "model",
                model
            )
        ]
    )


def evaluate(
    df,
    features,
    risk_threshold
):

    X = df[
        features
    ]

    y = (
        df["post_ssq_total"]
        > risk_threshold
    ).astype(int).to_numpy()

    groups = df[
        "participant_id"
    ].to_numpy()

    cv = GroupKFold(
        n_splits=5
    )

    all_true = []
    all_prob = []
    all_pred = []

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups
    ):

        train_X = X.iloc[
            train_idx
        ]

        test_X = X.iloc[
            test_idx
        ]

        train_y = y[
            train_idx
        ]

        pipeline = make_pipeline()

        pipeline.fit(
            train_X,
            train_y
        )

        probabilities = (
            pipeline.predict_proba(
                test_X
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        all_true.extend(
            y[test_idx]
        )

        all_prob.extend(
            probabilities
        )

        all_pred.extend(
            predictions
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
        "n_high_risk":
            int(
                all_true.sum()
            ),

        "prevalence":
            float(
                all_true.mean()
            ),

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


def main():

    df = pd.read_csv(
        DATA_PATH
    )

    results = []

    print(
        "=" * 80
    )
    print(
        "RISK THRESHOLD SENSITIVITY ANALYSIS"
    )
    print(
        "=" * 80
    )

    for risk_threshold in RISK_THRESHOLDS:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"HIGH RISK: post_ssq_total > "
            f"{risk_threshold}"
        )

        print(
            "=" * 80
        )

        for window in WINDOWS:

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

            features = movement_features(
                temp
            )

            metrics = evaluate(
                temp,
                features,
                risk_threshold
            )

            result = {
                "risk_threshold":
                    risk_threshold,

                "window":
                    window,

                "n_features":
                    len(features),

                **metrics
            }

            results.append(
                result
            )

            print(
                f"{window:<5} | "
                f"High={metrics['n_high_risk']:>3} | "
                f"Prev={metrics['prevalence']:.3f} | "
                f"ROC={metrics['roc_auc']:.3f} | "
                f"PR={metrics['pr_auc']:.3f} | "
                f"BalAcc={metrics['balanced_accuracy']:.3f} | "
                f"Prec={metrics['precision']:.3f} | "
                f"Recall={metrics['recall']:.3f} | "
                f"F1={metrics['f1']:.3f}"
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
        "\n"
        + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        results_df[
            [
                "risk_threshold",
                "window",
                "n_high_risk",
                "prevalence",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1"
            ]
        ]
        .round(3)
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