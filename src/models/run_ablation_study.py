import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


DATA = Path("data/processed/questset_ml_dataset.csv")
OUTPUT = Path("results/tables/questset_ablation_results.csv")

df = pd.read_csv(DATA)

TARGET = "post_ssq_total"

context_features = [
    "game_speed",
    "game_name",
    "age",
    "gender",
    "vr_experience",
    "gaming_experience"
]

pre_features = [
    "pre_ssq_total"
]

exclude = {
    "participant_id",
    "group",
    "order",
    "user",
    "session_number",
    "window",
    "window_seconds",
    "window_duration_actual",
    "n_samples",
    "median_dt",
    "effective_hz",
    "resample_hz",
    "tracking_jump_count",
    "tracking_jump_fraction",
    "pre_ssq_total",
    "post_ssq_total",
    "delta_ssq_total",
    "game_speed",
    "game_name",
    "age",
    "gender",
    "vr_experience",
    "gaming_experience"
}

movement_features = [
    c for c in df.columns
    if c not in exclude
]

feature_sets = {
    "pre_only": pre_features,
    "context_only": context_features,
    "movement_only": movement_features,
    "movement_context": movement_features + context_features,
    "all": movement_features + context_features + pre_features
}

results = []

for window in ["1min", "3min", "5min"]:

    data = df[df["window"] == window].copy()

    y = data[TARGET]
    groups = data["participant_id"]

    print("\n" + "=" * 70)
    print("WINDOW:", window)
    print("=" * 70)

    for set_name, features in feature_sets.items():

        X = data[features].copy()

        categorical = [
            c for c in X.columns
            if X[c].dtype == "object"
        ]

        numeric = [
            c for c in X.columns
            if c not in categorical
        ]

        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical)
            ]
        )

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        cv = GroupKFold(n_splits=5)

        maes = []
        rmses = []
        r2s = []

        for train_idx, test_idx in cv.split(X, y, groups):

            pipe.fit(
                X.iloc[train_idx],
                y.iloc[train_idx]
            )

            pred = pipe.predict(
                X.iloc[test_idx]
            )

            maes.append(
                mean_absolute_error(
                    y.iloc[test_idx],
                    pred
                )
            )

            rmses.append(
                mean_squared_error(
                    y.iloc[test_idx],
                    pred
                ) ** 0.5
            )

            r2s.append(
                r2_score(
                    y.iloc[test_idx],
                    pred
                )
            )

        result = {
            "window": window,
            "feature_set": set_name,
            "MAE": np.mean(maes),
            "RMSE": np.mean(rmses),
            "R2": np.mean(r2s)
        }

        results.append(result)

        print(
            f"{set_name:20s} "
            f"MAE={result['MAE']:.3f} "
            f"RMSE={result['RMSE']:.3f} "
            f"R2={result['R2']:.3f}"
        )


results_df = pd.DataFrame(results)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nFINAL ABLATION RESULTS")
print(results_df.to_string(index=False))

print(f"\nSaved: {OUTPUT}")