import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


DATA_PATH = Path(
    "data/processed/questset_second_generation_features.csv"
)

OUTPUT_PATH = Path(
    "results/tables/questset_second_generation_error_analysis.csv"
)

RANDOM_STATE = 42


# ============================================================
# Helper
# ============================================================

def build_pipeline(
    df,
    features,
    model
):

    numeric_features = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical_features = [
        c for c in features
        if c not in numeric_features
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

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipe,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipe,
                categorical_features
            )
        ]
    )

    return Pipeline(
        [
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


# ============================================================
# Feature selection
# ============================================================

def available_features(
    df,
    feature_type
):

    context = [
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

    movement = [
        c for c in df.columns
        if (
            c.startswith("1min_")
            or c.startswith("3min_")
            or c.startswith("5min_")
        )
        and not df[c].isna().all()
        and df[c].dropna().nunique() > 1
    ]

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

    if feature_type == "context_only":
        return context

    if feature_type == "full":
        return list(
            dict.fromkeys(
                context
                + base_movement
                + temporal
            )
        )

    raise ValueError(
        feature_type
    )


# ============================================================
# Cross-validated predictions
# ============================================================

def cross_validated_predictions(
    df,
    features,
    model
):

    groups = df[
        "participant_id"
    ]

    y = df[
        "post_ssq_total"
    ]

    cv = GroupKFold(
        n_splits=5
    )

    predictions = np.zeros(
        len(df)
    )

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

        pipeline = build_pipeline(
            train,
            features,
            model
        )

        pipeline.fit(
            train[
                features
            ],
            train[
                "post_ssq_total"
            ]
        )

        predictions[
            test_idx
        ] = pipeline.predict(
            test[
                features
            ]
        )

    return predictions


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    all_results = []

    configurations = [
        (
            "1min",
            "context_only",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=4,
                random_state=RANDOM_STATE
            )
        ),

        (
            "3min",
            "context_only",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=4,
                random_state=RANDOM_STATE
            )
        ),

        (
            "5min",
            "full",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]

    for (
        window,
        feature_type,
        model
    ) in configurations:

        temp = (
            df[
                df[
                    "window"
                ] == window
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        features = available_features(
            temp,
            feature_type
        )

        predictions = (
            cross_validated_predictions(
                temp,
                features,
                model
            )
        )

        temp[
            "prediction"
        ] = predictions

        temp[
            "error"
        ] = (
            temp[
                "prediction"
            ]
            - temp[
                "post_ssq_total"
            ]
        )

        temp[
            "absolute_error"
        ] = np.abs(
            temp[
                "error"
            ]
        )

        temp[
            "model_configuration"
        ] = feature_type

        all_results.append(
            temp[
                [
                    "participant_id",
                    "session_number",
                    "window",
                    "game_name",
                    "pre_ssq_total",
                    "post_ssq_total",
                    "delta_ssq_total",
                    "prediction",
                    "error",
                    "absolute_error",
                    "model_configuration"
                ]
            ]
        )

    errors = pd.concat(
        all_results,
        ignore_index=True
    )

    # ========================================================
    # Overall summary
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nOverall MAE:"
    )

    print(
        errors
        .groupby("window")[
            "absolute_error"
        ]
        .mean()
        .round(3)
    )

    # ========================================================
    # Error by game
    # ========================================================

    print(
        "\nMAE by game and window:"
    )

    game_summary = (
        errors
        .groupby(
            [
                "window",
                "game_name"
            ]
        )[
            "absolute_error"
        ]
        .agg(
            [
                "mean",
                "median",
                "max"
            ]
        )
        .round(3)
    )

    print(
        game_summary
    )

    # ========================================================
    # Bias by game
    # ========================================================

    print(
        "\nMean signed error by game:"
    )

    bias_summary = (
        errors
        .groupby(
            [
                "window",
                "game_name"
            ]
        )[
            "error"
        ]
        .mean()
        .round(3)
    )

    print(
        bias_summary
    )

    # ========================================================
    # High vs low sickness
    # ========================================================

    errors[
        "sickness_level"
    ] = pd.cut(
        errors[
            "post_ssq_total"
        ],
        bins=[
            -np.inf,
            10,
            30,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    print(
        "\nMAE by sickness level:"
    )

    sickness_summary = (
        errors
        .groupby(
            [
                "window",
                "sickness_level"
            ],
            observed=False
        )[
            "absolute_error"
        ]
        .agg(
            [
                "count",
                "mean",
                "median"
            ]
        )
        .round(3)
    )

    print(
        sickness_summary
    )

    # ========================================================
    # Prediction bias by sickness level
    # ========================================================

    print(
        "\nMean signed error by sickness level:"
    )

    sickness_bias = (
        errors
        .groupby(
            [
                "window",
                "sickness_level"
            ],
            observed=False
        )[
            "error"
        ]
        .mean()
        .round(3)
    )

    print(
        sickness_bias
    )

    # ========================================================
    # Worst predicted sessions
    # ========================================================

    print(
        "\nWorst 15 predictions:"
    )

    worst = (
        errors
        .sort_values(
            "absolute_error",
            ascending=False
        )
        [
            [
                "participant_id",
                "window",
                "game_name",
                "pre_ssq_total",
                "post_ssq_total",
                "prediction",
                "error",
                "absolute_error"
            ]
        ]
        .head(15)
    )

    print(
        worst.to_string(
            index=False
        )
    )

    # ========================================================
    # Extreme SSQ sensitivity
    # ========================================================

    print(
        "\nMAE excluding post-SSQ > 60:"
    )

    non_extreme = errors[
        errors[
            "post_ssq_total"
        ] <= 60
    ]

    print(
        non_extreme
        .groupby("window")[
            "absolute_error"
        ]
        .mean()
        .round(3)
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    errors.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()