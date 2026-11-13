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


DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_final_risk_scores.csv"
)

SUMMARY_PATH = Path(
    "results/tables/questset_final_risk_model_summary.csv"
)


RISK_THRESHOLD = 30.0
MODEL_C = 0.1
N_SPLITS = 5
RANDOM_STATE = 42


FEATURES = [
    "1min_velocity_mean",
    "1min_velocity_std",
    "1min_velocity_p95",
    "1min_angular_velocity_mean",
    "1min_angular_velocity_std",
    "1min_angular_velocity_p95",
    "1min_net_displacement"
]


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

    predicted_classes = np.zeros(
        len(df),
        dtype=int
    )

    fold_ids = np.zeros(
        len(df),
        dtype=int
    )

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1
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

        predicted_classes[
            test_idx
        ] = pred

        fold_ids[
            test_idx
        ] = fold

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {
        "risk_threshold":
            RISK_THRESHOLD,

        "model":
            "Logistic Regression",

        "C":
            MODEL_C,

        "n_sessions":
            len(df),

        "n_participants":
            df[
                "participant_id"
            ].nunique(),

        "n_high_risk":
            int(
                y.sum()
            ),

        "prevalence":
            float(
                y.mean()
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
                predicted_classes
            ),

        "precision":
            precision_score(
                y,
                predicted_classes,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                predicted_classes,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                predicted_classes,
                zero_division=0
            )
    }

    # ========================================================
    # Export risk scores
    # ========================================================

    output = df[
        [
            "participant_id",
            "session_number",
            "game_name",
            "game_speed",
            "pre_ssq_total",
            "post_ssq_total",
            "delta_ssq_total"
        ]
    ].copy()

    output[
        "high_risk_true"
    ] = y

    output[
        "risk_score"
    ] = probabilities

    output[
        "high_risk_predicted"
    ] = predicted_classes

    output[
        "cv_fold"
    ] = fold_ids

    output[
        "absolute_risk_error"
    ] = np.abs(
        y
        - probabilities
    )

    output = output.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "FINAL 1-MINUTE CYBERSICKNESS RISK MODEL"
    )

    print(
        "=" * 80
    )

    print(
        "\nConfiguration:"
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        f"C: {MODEL_C}"
    )

    print(
        f"Risk label: post_ssq_total > "
        f"{RISK_THRESHOLD}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        "\nCross-validated performance:"
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

    print(
        "\nTop 10 highest predicted risk sessions:"
    )

    print(
        output[
            [
                "participant_id",
                "game_name",
                "post_ssq_total",
                "high_risk_true",
                "risk_score"
            ]
        ]
        .head(10)
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved scores: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()