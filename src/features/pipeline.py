"""Feature transformation pipeline using Scikit-Learn."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for domain-specific feature engineering."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        # 1. Flag high-risk nighttime hours (11 PM - 5 AM)
        X_out["is_night_transaction"] = (
            (X_out["hour_of_day"] >= 23) | (X_out["hour_of_day"] <= 5)
        ).astype(int)

        # 2. Risk interaction: high velocity with high distance
        X_out["velocity_distance_interaction"] = (
            X_out["velocity_last_24h"] * X_out["distance_from_home_km"]
        )
        return X_out


def build_preprocessor() -> Pipeline:
    """Constructs the Scikit-learn preprocessing pipeline."""
    numeric_features = [
        "amount",
        "distance_from_home_km",
        "velocity_last_24h",
        "hour_of_day",
        "velocity_distance_interaction",
    ]
    categorical_features = ["merchant_category", "device_type"]
    binary_features = ["is_foreign_transaction", "is_night_transaction"]

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", column_preprocessor),
        ]
    )
    return pipeline