from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class TelemetryInput(BaseModel):
    request_duration_ms: float = Field(
        ...,
        description="Average request duration in milliseconds (latency)",
        examples=[145.20],
        ge=0.0
    )
    cpu_usage_pct: float = Field(
        ...,
        description="CPU utilization percentage (0-100%)",
        examples=[32.50],
        ge=0.0,
        le=100.0
    )
    memory_usage_pct: float = Field(
        ...,
        description="Memory utilization percentage (0-100%)",
        examples=[48.10],
        ge=0.0,
        le=100.0
    )
    requests_per_second: float = Field(
        ...,
        description="Total inbound request rate (requests per second)",
        examples=[85.00],
        ge=0.0
    )
    error_rate_pct: float = Field(
        ...,
        description="HTTP 5xx / application error rate percentage (0-100%)",
        examples=[0.45],
        ge=0.0,
        le=100.0
    )


class MetricContribution(BaseModel):
    metric: str
    value: float
    status: Literal["NORMAL", "ELEVATED", "CRITICAL"]
    rule_threshold_exceeded: bool


class AnomalyPredictionResponse(BaseModel):
    is_anomaly: bool = Field(..., description="True if telemetry is anomalous, False otherwise")
    anomaly_score: float = Field(
        ...,
        description="Isolation Forest decision score. Lower/negative scores indicate higher anomaly severity."
    )
    threshold: float = Field(..., description="Calibrated decision threshold used for classification")
    severity: Literal["NORMAL", "LOW_ANOMALY", "MEDIUM_ANOMALY", "CRITICAL_ANOMALY"] = Field(
        ...,
        description="Qualitative incident severity grade"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score (0.0 to 1.0) derived from margin to threshold"
    )
    rule_based_triggered: bool = Field(
        ...,
        description="True if static rule-based monitoring thresholds were also triggered"
    )
    summary: str = Field(..., description="Human-readable assessment summary")
    metric_analysis: List[MetricContribution] = Field(
        ...,
        description="Per-metric diagnostic breakdown against normal operating envelopes"
    )


class BatchTelemetryInput(BaseModel):
    samples: List[TelemetryInput] = Field(
        ...,
        description="Array of telemetry data points to classify in batch",
        min_length=1
    )


class BatchPredictionResponse(BaseModel):
    total_samples: int
    anomalies_detected: int
    normal_samples: int
    predictions: List[AnomalyPredictionResponse]


class ModelMetadata(BaseModel):
    model_name: str
    model_type: str
    n_estimators: int
    features: List[str]
    threshold: float
    trained_on: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    metadata: Optional[ModelMetadata] = None
