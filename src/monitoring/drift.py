"""Automated data and target drift detection using Evidently AI."""

import os
import json
import pandas as pd
from typing import Dict, Any

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from src.data.generator import generate_transactions


def run_drift_analysis(
    reference_path: str = "data/reference/baseline_reference.parquet",
    current_data: pd.DataFrame = None,
    output_html_path: str = "reports/drift_report.html",
    output_json_path: str = "reports/drift_summary.json",
) -> Dict[str, Any]:
    """
    Compares current production data against the baseline reference dataset.
    Generates an interactive HTML dashboard and a JSON summary.
    """
    # 1. Load Reference Dataset
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"Reference dataset not found at {reference_path}. Please train the model first."
        )
    
    reference_df = pd.read_parquet(reference_path)

    # 2. If no current data passed, simulate drifted production traffic
    if current_data is None:
        print("⚡ Simulating production traffic with active covariate & concept drift...")
        current_data = generate_transactions(n_samples=2500, fraud_ratio=0.06, drift=True, random_state=99)

    # Drop non-feature identifiers for statistical comparisons
    feature_cols = [
        c for c in reference_df.columns 
        if c not in ["transaction_id", "user_id"]
    ]
    ref_features = reference_df[feature_cols]
    curr_features = current_data[[c for c in feature_cols if c in current_data.columns]]

    print(" Running Evidently AI Statistical Drift Tests...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=ref_features, current_data=curr_features)

    # 3. Save Interactive HTML Report
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    report.save_html(output_html_path)
    print(f"📊 Interactive HTML report generated at: {output_html_path}")

    # 4. Extract Machine-Readable Summary JSON
    report_dict = report.as_dict()
    drift_metrics = report_dict["metrics"][0]["result"]
    dataset_drift_detected = drift_metrics.get("dataset_drift", False)
    drifted_columns_count = drift_metrics.get("number_of_drifted_columns", 0)
    share_of_drifted_columns = drift_metrics.get("share_of_drifted_columns", 0.0)

    summary = {
        "dataset_drift_detected": bool(dataset_drift_detected),
        "number_of_drifted_columns": int(drifted_columns_count),
        "share_of_drifted_columns": float(share_of_drifted_columns),
        "total_columns_evaluated": len(feature_cols),
    }

    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f" Summary JSON saved to: {output_json_path}")

    return summary


if __name__ == "__main__":
    summary = run_drift_analysis()
    print("\n🚨 Drift Analysis Results:")
    print(json.dumps(summary, indent=2))