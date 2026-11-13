import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_second_generation_results.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "post_ssq_total"

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# Columns that should never be model inputs
# ============================================================

NON_FEATURE_COLUMNS = {
    "participant_id",
    "post_ssq_total",
    "delta_ssq_total",
    "window",
    "window_seconds",
    "user"
}


# ============================================================
# Context features
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
# Helper: metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "mae":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "rmse":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "r2":
            r2_score(
                y_true,
                y_pred
            )
    }


# ============================================================
# Remove unavailable and constant features
# ============================================================

def clean_feature_columns(
    df,
    columns
):

    valid = []

    for col in columns:

        if col not in df.columns:
            continue

        # Completely unavailable at current horizon
        if df[col].isna().all():
            continue

        # Constant numeric/categorical feature
        non_missing = (
            df[col]
            .dropna()
        )

        if (
            len(non_missing) > 0
            and non_missing.nunique() <= 1
        ):
            continue

        valid.append(col)

    return valid


# ============================================================
# Feature families
# ============================================================

def get_feature_sets(
    df
):

    # --------------------------------------------------------
    # Everything available except identifiers and target
    # --------------------------------------------------------

    candidate_columns = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    candidate_columns = (
        clean_feature_columns(
            df,
            candidate_columns
        )
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = [
        c for c in CONTEXT_FEATURES
        if c in candidate_columns
    ]

    # --------------------------------------------------------
    # Movement features
    # --------------------------------------------------------

    movement = [
        c for c in candidate_columns
        if (
            c.startswith("1min_")
            or c.startswith("3min_")
            or c.startswith("5min_")
        )
        and not (
            "_slope" in c
            or "_early_late_change" in c
            or "_temporal_range" in c
            or "_temporal_std" in c
            or "_max_block" in c
        )
    ]

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    temporal = [
        c for c in candidate_columns
        if (
            "_slope" in c
            or "_early_late_change" in c
            or "_temporal_range" in c
            or "_temporal_std" in c
            or "_max_block" in c
        )
    ]

    return {
        "pre_ssq_only":
            [
                "pre_ssq_total"
            ],

        "context_only":
            context,

        "movement_only":
            movement,

        "movement_context":
            list(
                dict.fromkeys(
                    movement
                    + context
                )
            ),

        "movement_context_temporal":
            list(
                dict.fromkeys(
                    movement
                    + temporal
                    + context
                )
            )
    }


# ============================================================
# Build preprocessing/model pipeline
# ============================================================

def make_pipeline(
    df,
    features,
    model
):

    numeric_features = [
        c for c in features
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    categorical_features = [
        c for c in features
        if c not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
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
        steps=[
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# Cross-validation evaluation
# ============================================================

def evaluate_model(
    df,
    features,
    model_name,
    model
):

    X = df[
        features
    ]

    y = df[
        TARGET
    ].to_numpy()

    groups = df[
        "participant_id"
    ].to_numpy()

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_results = []

    all_true = []
    all_pred = []

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

        train_df = df.iloc[
            train_idx
        ]

        test_df = df.iloc[
            test_idx
        ]

        pipeline = make_pipeline(
            train_df,
            features,
            model
        )

        pipeline.fit(
            train_df[
                features
            ],
            train_df[
                TARGET
            ]
        )

        pred = pipeline.predict(
            test_df[
                features
            ]
        )

        y_true = test_df[
            TARGET
        ].to_numpy()

        metrics = (
            calculate_metrics(
                y_true,
                pred
            )
        )

        fold_results.append(
            {
                "fold":
                    fold,

                **metrics
            }
        )

        all_true.extend(
            y_true
        )

        all_pred.extend(
            pred
        )

    overall = calculate_metrics(
        np.array(all_true),
        np.array(all_pred)
    )

    return {
        "model":
            model_name,

        "mae":
            overall["mae"],

        "rmse":
            overall["rmse"],

        "r2":
            overall["r2"]
    }


# ============================================================
# Mean baseline
# ============================================================

def evaluate_mean_baseline(
    df
):

    y = df[
        TARGET
    ].to_numpy()

    groups = df[
        "participant_id"
    ].to_numpy()

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    all_true = []
    all_pred = []

    for train_idx, test_idx in cv.split(
        df,
        y,
        groups
    ):

        train_y = y[
            train_idx
        ]

        test_y = y[
            test_idx
        ]

        prediction = np.full(
            len(test_y),
            np.mean(
                train_y
            )
        )

        all_true.extend(
            test_y
        )

        all_pred.extend(
            prediction
        )

    metrics = calculate_metrics(
        np.array(all_true),
        np.array(all_pred)
    )

    return {
        "model":
            "Mean baseline",

        "mae":
            metrics["mae"],

        "rmse":
            metrics["rmse"],

        "r2":
            metrics["r2"]
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
        "SECOND-GENERATION MODEL EVALUATION"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Evaluate each prediction horizon separately
    # ========================================================

    for window in [
        "1min",
        "3min",
        "5min"
    ]:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"PREDICTION WINDOW: {window}"
        )

        print(
            "=" * 80
        )

        window_df = (
            df[
                df["window"]
                == window
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        print(
            "Rows:",
            len(
                window_df
            )
        )

        print(
            "Participants:",
            window_df[
                "participant_id"
            ].nunique()
        )

        feature_sets = (
            get_feature_sets(
                window_df
            )
        )

        # ====================================================
        # Mean baseline
        # ====================================================

        baseline = (
            evaluate_mean_baseline(
                window_df
            )
        )

        baseline[
            "window"
        ] = window

        baseline[
            "feature_set"
        ] = "mean_only"

        baseline[
            "n_features"
        ] = 0

        results.append(
            baseline
        )

        print(
            "\nMean baseline:"
        )

        print(
            baseline
        )

        # ====================================================
        # Feature-set evaluation
        # ====================================================

        for (
            feature_set_name,
            features
        ) in feature_sets.items():

            if len(features) == 0:
                continue

            print(
                f"\nFeature set: "
                f"{feature_set_name}"
            )

            print(
                "Features:",
                len(features)
            )

            models = {
                "Ridge":
                    Ridge(
                        alpha=10.0
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=4,
                        random_state=
                            RANDOM_STATE,
                        n_jobs=-1
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
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

                result = (
                    evaluate_model(
                        window_df,
                        features,
                        model_name,
                        model
                    )
                )

                result[
                    "window"
                ] = window

                result[
                    "feature_set"
                ] = (
                    feature_set_name
                )

                result[
                    "n_features"
                ] = len(
                    features
                )

                results.append(
                    result
                )

                print(
                    f"  "
                    f"{model_name:<20} "
                    f"MAE={result['mae']:.3f} "
                    f"RMSE={result['rmse']:.3f} "
                    f"R2={result['r2']:.3f}"
                )

    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "window",
            "feature_set",
            "model",
            "n_features",
            "mae",
            "rmse",
            "r2"
        ]
    ]

    results_df = (
        results_df
        .sort_values(
            [
                "window",
                "mae"
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
        "BEST RESULTS BY WINDOW"
    )

    print(
        "=" * 80
    )

    for window in [
        "1min",
        "3min",
        "5min"
    ]:

        temp = results_df[
            results_df[
                "window"
            ] == window
        ]

        print(
            f"\n{window}"
        )

        print(
            temp[
                [
                    "feature_set",
                    "model",
                    "n_features",
                    "mae",
                    "rmse",
                    "r2"
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