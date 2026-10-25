#!/usr/bin/env python3
"""
TelemetrySentry — Model Training & Serialization Script
Trains an Isolation Forest model on telemetry data, calibrates decision threshold
on validation set to maximize F1-score, and exports the joblib artifact.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telemetry_data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "telemetrysentry_model.joblib"

FEATURES = [
    "request_duration_ms",
    "cpu_usage_pct",
    "memory_usage_pct",
    "requests_per_second",
    "error_rate_pct",
]


def train_and_export():
    print(f"Loading telemetry dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Hard anomaly injection experiment
    df_hard = df.copy()
    rng = np.random.default_rng(123)

    candidate_rows = df_hard.index[
        (df_hard["is_anomaly"] == 0) &
        (df_hard["cpu_usage_pct"].between(40, 60)) &
        (df_hard["request_duration_ms"].between(200, 350)) &
        (df_hard["error_rate_pct"] < 3)
    ].to_numpy()

    selected_rows = rng.choice(candidate_rows, size=100, replace=False)
    df_hard.loc[selected_rows, "cpu_usage_pct"] += 10
    df_hard.loc[selected_rows, "request_duration_ms"] += 100
    df_hard.loc[selected_rows, "error_rate_pct"] += 4
    df_hard.loc[selected_rows, "is_anomaly"] = 1
    df_hard.loc[selected_rows, "incident_type"] = "subtle_correlated_incident"

    X = df_hard[FEATURES]
    y = df_hard["is_anomaly"]

    train_end = int(len(df_hard) * 0.60)
    val_end = int(len(df_hard) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"Training set: {len(X_train)} samples ({y_train.sum()} anomalies)")
    print(f"Validation set: {len(X_val)} samples ({y_val.sum()} anomalies)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} anomalies)")

    print("Fitting IsolationForest (200 estimators)...")
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(X_train)

    print("Calibrating decision boundary on validation set...")
    val_scores = model.decision_function(X_val)
    thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.30, 100)))

    best_thresh = -0.11404
    best_f1 = -1.0
    best_prec = -1.0

    for t in thresholds:
        pred = (val_scores < t).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        p = precision_score(y_val, pred, zero_division=0)
        if f > best_f1 or (f == best_f1 and p > best_prec):
            best_f1 = f
            best_prec = p
            best_thresh = float(t)

    print(f"Selected Threshold: {best_thresh:.6f} (Val F1: {best_f1:.4f}, Prec: {best_prec:.4f})")

    # Evaluate on test set
    test_scores = model.decision_function(X_test)
    test_pred = (test_scores < best_thresh).astype(int)
    print(f"Test Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"Test Recall:    {recall_score(y_test, test_pred):.4f}")
    print(f"Test F1-Score:  {f1_score(y_test, test_pred):.4f}")

    # Export artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "threshold": best_thresh,
        "features": FEATURES,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Exported production artifact to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_export()
