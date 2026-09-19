"""Training pipeline with MLflow tracking and artifact serialization."""

import os
import joblib
import mlflow
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.generator import generate_transactions
from src.features.pipeline import build_preprocessor
from src.models.evaluate import compute_metrics


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)

    # 1. Setup MLflow Tracking URI from config (fixes Windows path space bug)
    if "tracking_uri" in config:
        mlflow.set_tracking_uri(config["tracking_uri"])

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run():
        print(" Generating synthetic dataset...")
        df = generate_transactions(
            n_samples=config["data"]["n_samples"],
            fraud_ratio=config["data"]["fraud_ratio"],
            random_state=config["random_state"],
        )

        # Separate target & features
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["random_state"],
            stratify=y,
        )

        # Save baseline reference dataset for drift monitoring
        os.makedirs("data/reference", exist_ok=True)
        ref_data = X_test.copy()
        ref_data["is_fraud"] = y_test
        ref_data.to_parquet(config["data"]["reference_data_path"])
        print(f" Baseline reference dataset saved to {config['data']['reference_data_path']}")

        # 2. Build End-to-End Pipeline (Preprocessor + LightGBM Classifier)
        preprocessor = build_preprocessor()
        classifier = LGBMClassifier(
            n_estimators=config["model"]["n_estimators"],
            learning_rate=config["model"]["learning_rate"],
            num_leaves=config["model"]["num_leaves"],
            max_depth=config["model"]["max_depth"],
            scale_pos_weight=config["model"]["scale_pos_weight"],
            random_state=config["random_state"],
            verbose=-1,
        )

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        # 3. Fit Pipeline
        print(" Training LightGBM Fraud Detection Pipeline...")
        feature_cols = [c for c in X_train.columns if c not in ["transaction_id", "user_id"]]
        full_pipeline.fit(X_train[feature_cols], y_train)

        # 4. Evaluate
        y_pred_proba = full_pipeline.predict_proba(X_test[feature_cols])[:, 1]
        metrics = compute_metrics(
            y_true=y_test.values,
            y_pred_proba=y_pred_proba,
            threshold=config["model"]["decision_threshold"],
        )

        print(f"\n Model Performance:")
        print(f"   PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")

        # 5. Log Parameters & Metrics to MLflow
        mlflow.log_params(config["model"])
        mlflow.log_params(config["data"])
        mlflow.log_metrics(metrics)

        # 6. Save Model Artifact
        os.makedirs("models", exist_ok=True)
        model_path = config["model"]["output_model_path"]
        joblib.dump(full_pipeline, model_path)
        mlflow.log_artifact(model_path)
        print(f"\n Model pipeline successfully saved to {model_path}")


if __name__ == "__main__":
    train_model()