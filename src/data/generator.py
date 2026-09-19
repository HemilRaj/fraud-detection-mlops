"""Synthetic transaction dataset generator with drift simulation capabilities."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_transactions(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    drift: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Proportion of fraudulent transactions (e.g., 0.02 = 2%).
        drift: If True, injects covariate drift (amount shifts) and concept drift.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing synthetic transactions matching TransactionInput schema.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # 1. Legitimate Transactions
    legit_amounts = rng.exponential(scale=45.0, size=n_legit) + 2.0
    legit_distances = rng.exponential(scale=8.0, size=n_legit)
    legit_velocities = rng.poisson(lam=1.5, size=n_legit)
    legit_hours = rng.integers(low=6, high=23, size=n_legit)  # mostly daytime
    legit_foreign = rng.choice([0, 1], size=n_legit, p=[0.97, 0.03])

    # 2. Fraudulent Transactions (Baseline pattern)
    if not drift:
        fraud_amounts = rng.normal(loc=350.0, scale=120.0, size=n_fraud).clip(min=10.0)
        fraud_distances = rng.exponential(scale=60.0, size=n_fraud) + 20.0
        fraud_velocities = rng.poisson(lam=6.0, size=n_fraud)
        fraud_hours = rng.integers(low=0, high=24, size=n_fraud)
        fraud_foreign = rng.choice([0, 1], size=n_fraud, p=[0.60, 0.40])
    else:
        # DRIFT SCENARIO: High-velocity micro-transactions and night surges
        fraud_amounts = rng.exponential(scale=15.0, size=n_fraud) + 1.0  # micro-amounts
        fraud_distances = rng.exponential(scale=120.0, size=n_fraud)
        fraud_velocities = rng.poisson(lam=12.0, size=n_fraud)  # rapid attacks
        fraud_hours = rng.integers(low=0, high=6, size=n_fraud)  # late night attacks
        fraud_foreign = rng.choice([0, 1], size=n_fraud, p=[0.30, 0.70])

    # Combine Legit & Fraud
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    distances = np.concatenate([legit_distances, fraud_distances])
    velocities = np.concatenate([legit_velocities, fraud_velocities])
    hours = np.concatenate([legit_hours, fraud_hours])
    foreign = np.concatenate([legit_foreign, fraud_foreign])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    categories = ["grocery", "electronics", "travel", "entertainment", "utilities", "restaurant"]
    devices = ["mobile", "desktop", "tablet"]

    cat_choices = rng.choice(categories, size=n_samples, p=[0.35, 0.15, 0.10, 0.15, 0.10, 0.15])
    dev_choices = rng.choice(devices, size=n_samples, p=[0.65, 0.25, 0.10])

    tx_ids = [f"tx_{i:06d}" for i in range(n_samples)]
    usr_ids = [f"usr_{rng.integers(100, 999)}" for _ in range(n_samples)]

    df = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "user_id": usr_ids,
            "amount": np.round(amounts, 2),
            "merchant_category": cat_choices,
            "device_type": dev_choices,
            "is_foreign_transaction": foreign,
            "hour_of_day": hours,
            "distance_from_home_km": np.round(distances, 2),
            "velocity_last_24h": velocities,
            "is_fraud": labels,
        }
    )

    # Shuffle dataset
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)