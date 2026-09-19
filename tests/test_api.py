"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_endpoint(client):
    payload = {
        "transaction_id": "tx_test_001",
        "user_id": "usr_999",
        "amount": 450.0,
        "merchant_category": "electronics",
        "device_type": "mobile",
        "is_foreign_transaction": 1,
        "hour_of_day": 3,
        "distance_from_home_km": 85.0,
        "velocity_last_24h": 7,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_test_001"
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_single_invalid_payload(client):
    # Invalid amount (<= 0)
    payload = {
        "transaction_id": "tx_invalid",
        "user_id": "usr_001",
        "amount": -20.0,
        "merchant_category": "grocery",
        "device_type": "desktop",
        "is_foreign_transaction": 0,
        "hour_of_day": 12,
        "distance_from_home_km": 1.0,
        "velocity_last_24h": 1,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)