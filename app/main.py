import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    TelemetryInput,
    AnomalyPredictionResponse,
    BatchTelemetryInput,
    BatchPredictionResponse,
    HealthResponse,
)
from app.model_loader import detector, RULE_THRESHOLDS

app = FastAPI(
    title="TelemetrySentry API",
    description=(
        "Production-ready anomaly detection API for cloud infrastructure telemetry. "
        "Powered by an Isolation Forest model optimized with calibrated decision boundaries."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend / dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Root endpoint welcoming clients and directing to docs."""
    return {
        "service": "TelemetrySentry Anomaly Detection Microservice",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "model": detector.get_metadata().model_dump(),
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Returns service health status and model artifact metadata."""
    return HealthResponse(
        status="healthy" if detector.loaded else "degraded",
        version="1.0.0",
        model_loaded=detector.loaded,
        metadata=detector.get_metadata() if detector.loaded else None,
    )


@app.get("/rules", tags=["Baseline Monitoring"])
async def get_rule_thresholds() -> Dict[str, Any]:
    """Returns static rule-based monitoring thresholds used as baseline benchmark."""
    return {
        "description": "Static threshold heuristics used for legacy alerting vs machine learning comparison",
        "thresholds": RULE_THRESHOLDS,
    }


@app.post(
    "/predict",
    response_model=AnomalyPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict anomaly for a single telemetry record",
)
async def predict_telemetry(payload: TelemetryInput) -> AnomalyPredictionResponse:
    """
    Accepts real-time server telemetry and returns:
    - `is_anomaly`: Boolean indicating if the system behavior is anomalous
    - `anomaly_score`: Isolation Forest decision score (lower values = higher anomaly severity)
    - `severity`: Qualitative category (NORMAL, LOW_ANOMALY, MEDIUM_ANOMALY, CRITICAL_ANOMALY)
    - `confidence`: Confidence rating based on distance from tuned threshold
    - `rule_based_triggered`: Whether classical static rules fired
    - `metric_analysis`: Metric-by-metric inspection against operational limits
    """
    if not detector.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized",
        )

    try:
        return detector.predict_single(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Batch anomaly prediction for multiple telemetry records",
)
async def predict_telemetry_batch(payload: BatchTelemetryInput) -> BatchPredictionResponse:
    """
    Accepts an array of telemetry observations and processes them in batch.
    Returns aggregate summary statistics and individual classification results.
    """
    if not detector.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized",
        )

    try:
        return detector.predict_batch(payload.samples)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )
