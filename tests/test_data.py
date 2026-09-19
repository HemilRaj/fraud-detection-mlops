"""Data schemas and contracts for transaction fraud detection."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator


class MerchantCategory(str, Enum):
    GROCERY = "grocery"
    ELECTRONICS = "electronics"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    RESTAURANT = "restaurant"


class DeviceType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TransactionInput(BaseModel):
    """Schema for a single incoming transaction request."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Unique user identifier")
    amount: float = Field(..., gt=0.0, description="Transaction amount in USD (must be > 0)")
    merchant_category: MerchantCategory = Field(..., description="Category of merchant")
    device_type: DeviceType = Field(..., description="Device used for transaction")
    is_foreign_transaction: int = Field(
        ..., ge=0, le=1, description="1 if transaction is international, else 0"
    )
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    distance_from_home_km: float = Field(
        ..., ge=0.0, description="Distance from cardholder's home address in km"
    )
    velocity_last_24h: int = Field(
        ..., ge=0, description="Number of transactions by user in last 24 hours"
    )

    @field_validator("transaction_id", "user_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID cannot be empty or blank")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "tx_12345",
                "user_id": "usr_987",
                "amount": 250.75,
                "merchant_category": "electronics",
                "device_type": "mobile",
                "is_foreign_transaction": 0,
                "hour_of_day": 14,
                "distance_from_home_km": 12.5,
                "velocity_last_24h": 3,
            }
        }
    }


class BatchTransactionInput(BaseModel):
    """Schema for batch prediction requests."""

    transactions: List[TransactionInput] = Field(
        ..., min_length=1, description="List of transactions to score"
    )


class PredictionOutput(BaseModel):
    """Schema for a single transaction prediction response."""

    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    risk_level: RiskLevel
    model_version: str


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction responses."""

    predictions: List[PredictionOutput]
    total_processed: int