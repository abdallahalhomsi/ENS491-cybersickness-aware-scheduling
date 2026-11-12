import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


DATA_PATH = Path("data/processed/questset_ml_dataset.csv")
OUTPUT_PATH = Path("results/tables/questset_baseline_results.csv")

TARGET = "post_ssq_total"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Features to exclude
# --------------------------------------------------

exclude_cols = [
    # identifiers
    "participant_id",
    "group",
    "order",
    "user",
    "session_number",

    # window labels / constants
    "window",
    "window_seconds",
    "window_duration_actual",
    "n_samples",
    "median_dt",
    "effective_hz",
    "resample_hz",

    # tracking quality indicators
    "tracking_jump_count",
    "tracking_jump_fraction",

    # targets
    "post_ssq_total",
    "delta_ssq_total"
]


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "MeanBaseline": None,

    "Ridge": Ridge(
        alpha=1.0
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    )
}


# --------------------------------------------------
# Run separately for 1, 3, and 5 minutes
# --------------------------------------------------

results = []

for window in ["1min", "3min", "5min"]:

    print("\n" + "=" * 70)
    print("WINDOW:", window)
    print("=" * 70)

    data = df[df["window"] == window].copy()

    groups = data["participant_id"]
    y = data[TARGET]

    X = data.drop(columns=exclude_cols)

    categorical_cols = [
        c for c in X.columns
        if X[c].dtype == "object"
    ]

    numeric_cols = [
        c for c in X.columns
        if c not in categorical_cols
    ]

    print("Rows:", len(data))
    print("Participants:", groups.nunique())
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", categorical_cols)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # --------------------------------------------------
    # 5-fold participant-independent CV
    # --------------------------------------------------

    cv = GroupKFold(n_splits=5)

    for model_name, model in models.items():

        fold_metrics = []

        print("\nModel:", model_name)

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            overlap = train_groups.intersection(test_groups)

            if len(overlap) != 0:
                raise RuntimeError(
                    f"Participant leakage detected: {overlap}"
                )

            # ------------------------------------------
            # Mean baseline
            # ------------------------------------------

            if model_name == "MeanBaseline":

                prediction = np.repeat(
                    y_train.mean(),
                    len(y_test)
                )

            # ------------------------------------------
            # ML models
            # ------------------------------------------

            else:

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ]
                )

                pipeline.fit(
                    X_train,
                    y_train
                )

                prediction = pipeline.predict(
                    X_test
                )

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            rmse = mean_squared_error(
                y_test,
                prediction
            ) ** 0.5

            r2 = r2_score(
                y_test,
                prediction
            )

            fold_metrics.append(
                {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }
            )

            print(
                f"  Fold {fold}: "
                f"MAE={mae:.3f}, "
                f"RMSE={rmse:.3f}, "
                f"R2={r2:.3f}"
            )

        fold_df = pd.DataFrame(
            fold_metrics
        )

        result = {
            "window": window,
            "model": model_name,

            "MAE_mean": fold_df["MAE"].mean(),
            "MAE_std": fold_df["MAE"].std(),

            "RMSE_mean": fold_df["RMSE"].mean(),
            "RMSE_std": fold_df["RMSE"].std(),

            "R2_mean": fold_df["R2"].mean(),
            "R2_std": fold_df["R2"].std()
        }

        results.append(result)

        print(
            f"  Average: "
            f"MAE={result['MAE_mean']:.3f}, "
            f"RMSE={result['RMSE_mean']:.3f}, "
            f"R2={result['R2_mean']:.3f}"
        )


# --------------------------------------------------
# Final results table
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "window",
            "model",
            "MAE_mean",
            "RMSE_mean",
            "R2_mean"
        ]
    ].to_string(index=False)
)

print(f"\nSaved: {OUTPUT_PATH}")