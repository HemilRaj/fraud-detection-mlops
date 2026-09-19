"""API routes for real-time and batch fraud prediction."""

import os
import time
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from src.data.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    PredictionOutput,
    RiskLevel,
    TransactionInput,
)

router = APIRouter()


def score_transactions(model, transactions_df: pd.DataFrame, threshold: float, model_version: str):
    """Utility function to score a DataFrame of transactions."""
    feature_cols = [c for c in transactions_df.columns if c not in ["transaction_id", "user_id"]]
    probabilities = model.predict_proba(transactions_df[feature_cols])[:, 1]

    results = []
    for tx_id, prob in zip(transactions_df["transaction_id"], probabilities):
        is_fraud = bool(prob >= threshold)
        if prob >= 0.75:
            risk = RiskLevel.HIGH
        elif prob >= threshold:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        results.append(
            PredictionOutput(
                transaction_id=tx_id,
                fraud_probability=round(float(prob), 4),
                is_fraud=is_fraud,
                risk_level=risk,
                model_version=model_version,
            )
        )
    return results, probabilities


@router.get("/health", tags=["Monitoring"])
async def health_check(request: Request):
    """Health check probe for container orchestrators (Kubernetes / ECS)."""
    model_loaded = request.app.state.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_version": request.app.state.config["model"]["model_version"],
    }


@router.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_single(transaction: TransactionInput, request: Request):
    """
    Score a single transaction in real-time (< 20ms SLA).
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Convert Pydantic object to single-row DataFrame
    tx_dict = transaction.model_dump()
    df = pd.DataFrame([tx_dict])

    config = request.app.state.config
    predictions, probs = score_transactions(
        model=request.app.state.model,
        transactions_df=df,
        threshold=config["model"]["decision_threshold"],
        model_version=config["model"]["model_version"],
    )

    # Log prediction asynchronously for drift monitoring
    if config["monitoring"]["log_predictions"]:
        df_log = df.copy()
        df_log["fraud_probability"] = probs[0]
        df_log["timestamp"] = time.time()
        
        log_file = config["monitoring"]["prediction_log_path"]
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        df_log.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)

    return predictions[0]


@router.post("/batch_predict", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(batch: BatchTransactionInput, request: Request):
    """
    High-throughput batch transaction scoring.
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    tx_dicts = [t.model_dump() for t in batch.transactions]
    df = pd.DataFrame(tx_dicts)

    config = request.app.state.config
    predictions, _ = score_transactions(
        model=request.app.state.model,
        transactions_df=df,
        threshold=config["model"]["decision_threshold"],
        model_version=config["model"]["model_version"],
    )

    return BatchPredictionOutput(
        predictions=predictions,
        total_processed=len(predictions),
    )