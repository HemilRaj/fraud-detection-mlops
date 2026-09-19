"""Unit tests for data generator and feature transformation pipeline."""

import numpy as np
from src.data.generator import generate_transactions
from src.features.pipeline import build_preprocessor


def test_data_generator_shape_and_fraud_ratio():
    df = generate_transactions(n_samples=1000, fraud_ratio=0.05, random_state=42)
    assert len(df) == 1000
    assert df["is_fraud"].sum() == 50
    assert "amount" in df.columns
    assert (df["amount"] > 0).all()


def test_feature_pipeline_fit_transform():
    df = generate_transactions(n_samples=500, random_state=42)
    X = df.drop(columns=["is_fraud", "transaction_id", "user_id"])

    pipeline = build_preprocessor()
    X_transformed = pipeline.fit_transform(X)

    # Output should be a 2D numpy array with no NaN values
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 500
    assert not np.isnan(X_transformed).any()