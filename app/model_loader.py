import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from app.schemas import (
    TelemetryInput,
    AnomalyPredictionResponse,
    MetricContribution,
    BatchPredictionResponse,
    ModelMetadata,
)

# Reference baseline thresholds
RULE_THRESHOLDS = {
    "request_duration_ms": {"elevated": 500.0, "critical": 1000.0},
    "cpu_usage_pct": {"elevated": 75.0, "critical": 85.0},
    "memory_usage_pct": {"elevated": 80.0, "critical": 90.0},
    "requests_per_second": {"elevated": 180.0, "critical": 250.0},
    "error_rate_pct": {"elevated": 3.0, "critical": 10.0},
}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "telemetrysentry_model.joblib"


class TelemetryAnomalyDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.threshold: float = -0.11404
        self.features: List[str] = [
            "request_duration_ms",
            "cpu_usage_pct",
            "memory_usage_pct",
            "requests_per_second",
            "error_rate_pct",
        ]
        self.loaded = False
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        
        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict):
            self.model = artifact.get("model")
            self.threshold = float(artifact.get("threshold", -0.11404))
            self.features = artifact.get("features", self.features)
        else:
            self.model = artifact
            self.threshold = 0.0

        self.loaded = True

    def get_metadata(self) -> ModelMetadata:
        n_estimators = getattr(self.model, "n_estimators", 200) if self.model else 200
        return ModelMetadata(
            model_name="TelemetrySentry-IsolationForest",
            model_type="IsolationForest",
            n_estimators=n_estimators,
            features=self.features,
            threshold=round(self.threshold, 6),
            trained_on="Production-grade synthetic telemetry with multi-incident injection",
        )

    def _analyze_metrics(self, data: Dict[str, float]) -> Tuple[List[MetricContribution], bool]:
        metric_analyses = []
        rule_triggered = False

        for feat in self.features:
            val = float(data.get(feat, 0.0))
            limits = RULE_THRESHOLDS.get(feat, {"elevated": 80.0, "critical": 90.0})
            
            exceeded = val > limits["critical"]
            if exceeded:
                rule_triggered = True
                status = "CRITICAL"
            elif val > limits["elevated"]:
                status = "ELEVATED"
            else:
                status = "NORMAL"

            metric_analyses.append(
                MetricContribution(
                    metric=feat,
                    value=round(val, 2),
                    status=status,
                    rule_threshold_exceeded=exceeded,
                )
            )

        return metric_analyses, rule_triggered

    def predict_single(self, telemetry: TelemetryInput) -> AnomalyPredictionResponse:
        row_dict = telemetry.model_dump()
        df = pd.DataFrame([row_dict])[self.features]

        # Compute decision score (lower means more anomalous)
        score = float(self.model.decision_function(df)[0])
        is_anomaly = bool(score < self.threshold)

        metric_analyses, rule_triggered = self._analyze_metrics(row_dict)

        # Severity & confidence assessment
        margin = self.threshold - score
        if not is_anomaly:
            severity = "NORMAL"
            confidence = min(1.0, max(0.5, 0.5 + abs(margin) * 1.5))
            summary = "Telemetry profile within standard operational parameters."
        else:
            if margin < 0.04:
                severity = "LOW_ANOMALY"
            elif margin < 0.10:
                severity = "MEDIUM_ANOMALY"
            else:
                severity = "CRITICAL_ANOMALY"
            
            confidence = min(0.99, max(0.65, 0.65 + margin * 1.8))
            
            crit_metrics = [m.metric for m in metric_analyses if m.status in ("ELEVATED", "CRITICAL")]
            if crit_metrics:
                summary = f"Anomaly detected ({severity}) with elevated signals on: {', '.join(crit_metrics)}."
            else:
                summary = f"Subtle multivariate anomaly detected ({severity}); correlated deviation identified across metric envelope."

        return AnomalyPredictionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 5),
            threshold=round(self.threshold, 5),
            severity=severity,
            confidence=round(confidence, 4),
            rule_based_triggered=rule_triggered,
            summary=summary,
            metric_analysis=metric_analyses,
        )

    def predict_batch(self, batch: List[TelemetryInput]) -> BatchPredictionResponse:
        records = [item.model_dump() for item in batch]
        df = pd.DataFrame(records)[self.features]

        scores = self.model.decision_function(df)
        predictions: List[AnomalyPredictionResponse] = []
        anomalies_count = 0

        for item, score in zip(batch, scores):
            score_val = float(score)
            is_anomaly = bool(score_val < self.threshold)
            if is_anomaly:
                anomalies_count += 1
            
            row_dict = item.model_dump()
            metric_analyses, rule_triggered = self._analyze_metrics(row_dict)

            margin = self.threshold - score_val
            if not is_anomaly:
                severity = "NORMAL"
                confidence = min(1.0, max(0.5, 0.5 + abs(margin) * 1.5))
                summary = "Telemetry profile within standard operational parameters."
            else:
                if margin < 0.04:
                    severity = "LOW_ANOMALY"
                elif margin < 0.10:
                    severity = "MEDIUM_ANOMALY"
                else:
                    severity = "CRITICAL_ANOMALY"
                confidence = min(0.99, max(0.65, 0.65 + margin * 1.8))
                crit_metrics = [m.metric for m in metric_analyses if m.status in ("ELEVATED", "CRITICAL")]
                if crit_metrics:
                    summary = f"Anomaly detected ({severity}) with elevated signals on: {', '.join(crit_metrics)}."
                else:
                    summary = f"Subtle multivariate anomaly detected ({severity})."

            predictions.append(
                AnomalyPredictionResponse(
                    is_anomaly=is_anomaly,
                    anomaly_score=round(score_val, 5),
                    threshold=round(self.threshold, 5),
                    severity=severity,
                    confidence=round(confidence, 4),
                    rule_based_triggered=rule_triggered,
                    summary=summary,
                    metric_analysis=metric_analyses,
                )
            )

        total = len(predictions)
        return BatchPredictionResponse(
            total_samples=total,
            anomalies_detected=anomalies_count,
            normal_samples=total - anomalies_count,
            predictions=predictions,
        )


# Global singleton instance
detector = TelemetryAnomalyDetector()
