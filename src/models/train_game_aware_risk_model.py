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
    f1_score,
    confusion_matrix
)


DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_game_aware_risk_results.csv"
)

RANDOM_STATE = 42
RISK_THRESHOLD = 30.0


# ============================================================
# Candidate features justified by EDA
# ============================================================

BASE_FEATURES = [
    "1min_velocity_mean",
    "1min_velocity_std",
    "1min_velocity_p95",
    "1min_angular_velocity_mean",
    "1min_angular_velocity_std",
    "1min_angular_velocity_p95"
]

CONTEXT_FEATURES = [
    "game_name",
    "pre_ssq_total",
    "session_number",
    "order"
]


# ============================================================
# Create compact game interactions
# ============================================================

def add_game_interactions(df):

    out = df.copy()

    games = [
        "Beat Saber",
        "Cooking Simulator",
        "Forklift Simulator",
        "Medal of Honor"
    ]

    selected_motion_features = [
        "1min_velocity_mean",
        "1min_angular_velocity_mean",
        "1min_angular_velocity_std",
        "1min_angular_velocity_p95"
    ]

    for game in games:

        slug = (
            game
            .lower()
            .replace(" ", "_")
        )

        indicator = (
            out["game_name"] == game
        ).astype(int)

        for feature in selected_motion_features:

            out[
                f"{feature}_x_{slug}"
            ] = (
                out[feature]
                * indicator
            )

    return out


# ============================================================
# Pipeline
# ============================================================

def make_pipeline(
    df,
    features
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

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
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


# ============================================================
# Evaluate with GroupKFold
# ============================================================

def evaluate(
    df,
    features
):

    y = (
        df["post_ssq_total"]
        > RISK_THRESHOLD
    ).astype(int).to_numpy()

    groups = (
        df["participant_id"]
        .to_numpy()
    )

    cv = GroupKFold(
        n_splits=5
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

        pipeline = make_pipeline(
            train,
            features
        )

        pipeline.fit(
            train[features],
            y[train_idx]
        )

        prob = (
            pipeline.predict_proba(
                test[features]
            )[:, 1]
        )

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

    all_true = np.asarray(
        all_true
    )

    all_prob = np.asarray(
        all_prob
    )

    all_pred = np.asarray(
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

        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn
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

    df = add_game_interactions(
        df
    )

    interaction_features = [
        c for c in df.columns
        if "_x_" in c
    ]

    feature_sets = {
        "movement_only":
            BASE_FEATURES,

        "movement_context":
            BASE_FEATURES
            + CONTEXT_FEATURES,

        "game_aware":
            BASE_FEATURES
            + CONTEXT_FEATURES
            + interaction_features
    }

    results = []

    print(
        "=" * 80
    )

    print(
        "GAME-AWARE EARLY RISK MODEL"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRisk threshold: "
        f"post_ssq_total > "
        f"{RISK_THRESHOLD}"
    )

    print(
        "\nHigh-risk sessions:",
        int(
            (
                df["post_ssq_total"]
                > RISK_THRESHOLD
            ).sum()
        )
    )

    for (
        name,
        features
    ) in feature_sets.items():

        metrics = evaluate(
            df,
            features
        )

        results.append(
            {
                "feature_set":
                    name,

                "n_features":
                    len(features),

                **metrics
            }
        )

        print(
            f"\n{name}"
        )

        print(
            f"Features: "
            f"{len(features)}"
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
            f"TP={metrics['tp']} "
            f"FN={metrics['fn']} "
            f"FP={metrics['fp']} "
            f"TN={metrics['tn']}"
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
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()