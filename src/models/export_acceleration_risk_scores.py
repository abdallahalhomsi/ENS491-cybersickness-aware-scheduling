import json

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

CSV_OUTPUT_PATH = Path(
    "results/tables/questset_acceleration_risk_scores.csv"
)

SUMMARY_OUTPUT_PATH = Path(
    "results/tables/questset_acceleration_risk_summary.csv"
)

JSON_OUTPUT_PATH = Path(
    "network/scheduler/risk_scores.json"
)


# ============================================================
# Configuration
# ============================================================

RISK_THRESHOLD = 30.0
MODEL_C = 0.1
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Final feature set
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

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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

    X = df[
        FEATURES
    ]

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(
        int
    ).to_numpy()

    groups = (
        df[
            "participant_id"
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Participant-independent cross-validation
    # --------------------------------------------------------

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
            prob
            >= 0.5
        ).astype(
            int
        )

        probabilities[
            test_idx
        ] = prob

        predictions[
            test_idx
        ] = pred

        fold_ids[
            test_idx
        ] = fold

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "model":
            "Logistic Regression",

        "feature_set":
            "1-minute acceleration only",

        "risk_threshold":
            RISK_THRESHOLD,

        "C":
            MODEL_C,

        "n_features":
            len(
                FEATURES
            ),

        "n_sessions":
            len(
                df
            ),

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

    # --------------------------------------------------------
    # Build detailed CSV output
    # --------------------------------------------------------

    output = df[
        [
            "participant_id",
            "session_number",
            "user",
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
        "risk_probability"
    ] = probabilities

    output[
        "predicted_high_risk"
    ] = predictions

    output[
        "cv_fold"
    ] = fold_ids

    # Include final feature values for traceability.
    for feature in FEATURES:

        output[
            feature
        ] = df[
            feature
        ]

    # --------------------------------------------------------
    # Save CSV results
    # --------------------------------------------------------

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        CSV_OUTPUT_PATH,
        index=False
    )

    pd.DataFrame(
        [
            metrics
        ]
    ).to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Build scheduler JSON
    #
    # Use a unique session identifier because every Questset
    # participant has two sessions.
    # --------------------------------------------------------

    scheduler_scores = {}

    for _, row in output.iterrows():

        user_id = str(
            row[
                "participant_id"
            ]
        )

        session_number = int(
            row[
                "session_number"
            ]
        )

        key = (
            f"{user_id}"
            f"_session{session_number}"
        )

        scheduler_scores[
            key
        ] = round(
            float(
                row[
                    "risk_probability"
                ]
            ),
            6
        )

    JSON_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        JSON_OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            scheduler_scores,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "ACCELERATION RISK SCORE EXPORT"
    )

    print(
        "=" * 80
    )

    print(
        "Sessions:",
        metrics[
            "n_sessions"
        ]
    )

    print(
        "Participants:",
        metrics[
            "n_participants"
        ]
    )

    print(
        "High-risk sessions:",
        metrics[
            "n_high_risk"
        ]
    )

    print()

    print(
        "ROC-AUC:",
        round(
            metrics[
                "roc_auc"
            ],
            3
        )
    )

    print(
        "PR-AUC:",
        round(
            metrics[
                "pr_auc"
            ],
            3
        )
    )

    print(
        "Balanced accuracy:",
        round(
            metrics[
                "balanced_accuracy"
            ],
            3
        )
    )

    print(
        "Precision:",
        round(
            metrics[
                "precision"
            ],
            3
        )
    )

    print(
        "Recall:",
        round(
            metrics[
                "recall"
            ],
            3
        )
    )

    print(
        "F1:",
        round(
            metrics[
                "f1"
            ],
            3
        )
    )

    print()

    print(
        "Risk score range:"
    )

    print(
        "  min:",
        round(
            probabilities.min(),
            3
        )
    )

    print(
        "  mean:",
        round(
            probabilities.mean(),
            3
        )
    )

    print(
        "  max:",
        round(
            probabilities.max(),
            3
        )
    )

    print()

    print(
        "CSV:"
    )

    print(
        CSV_OUTPUT_PATH
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print()

    print(
        "Scheduler JSON:"
    )

    print(
        JSON_OUTPUT_PATH
    )

    print()

    print(
        "First 10 scheduler entries:"
    )

    for (
        key,
        value
    ) in list(
        scheduler_scores.items()
    )[:10]:

        print(
            f"  {key}: {value}"
        )


if __name__ == "__main__":
    main()