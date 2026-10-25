import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "model" in data
    assert data["model"]["model_type"] == "IsolationForest"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "metadata" in data
    assert len(data["metadata"]["features"]) == 5
    assert data["metadata"]["threshold"] < 0


def test_rules_endpoint():
    response = client.get("/rules")
    assert response.status_code == 200
    data = response.json()
    assert "thresholds" in data
    assert "request_duration_ms" in data["thresholds"]


def test_predict_normal_telemetry():
    """Test standard operating conditions return normal status."""
    payload = {
        "request_duration_ms": 115.4,
        "cpu_usage_pct": 28.5,
        "memory_usage_pct": 46.2,
        "requests_per_second": 68.0,
        "error_rate_pct": 0.25,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_anomaly"] is False
    assert data["severity"] == "NORMAL"
    assert data["anomaly_score"] >= data["threshold"]
    assert data["rule_based_triggered"] is False
    assert "within standard operational parameters" in data["summary"].lower()


def test_predict_latency_spike_anomaly():
    """Test latency spike incident is detected as anomaly."""
    payload = {
        "request_duration_ms": 1550.0,
        "cpu_usage_pct": 45.0,
        "memory_usage_pct": 52.0,
        "requests_per_second": 110.0,
        "error_rate_pct": 1.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_anomaly"] is True
    assert data["anomaly_score"] < data["threshold"]
    lat_analysis = next(m for m in data["metric_analysis"] if m["metric"] == "request_duration_ms")
    assert lat_analysis["status"] == "CRITICAL"
    assert lat_analysis["rule_threshold_exceeded"] is True


def test_predict_error_rate_spike_anomaly():
    """Test error rate spike incident is detected as anomaly."""
    payload = {
        "request_duration_ms": 290.0,
        "cpu_usage_pct": 52.0,
        "memory_usage_pct": 50.0,
        "requests_per_second": 95.0,
        "error_rate_pct": 29.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_anomaly"] is True
    assert data["anomaly_score"] < data["threshold"]
    err_analysis = next(m for m in data["metric_analysis"] if m["metric"] == "error_rate_pct")
    assert err_analysis["status"] == "CRITICAL"
    assert err_analysis["rule_threshold_exceeded"] is True


def test_predict_traffic_surge_anomaly():
    """Test traffic surge overloading server queue is detected as anomaly."""
    payload = {
        "request_duration_ms": 520.0,
        "cpu_usage_pct": 78.0,
        "memory_usage_pct": 68.0,
        "requests_per_second": 290.0,
        "error_rate_pct": 4.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_anomaly"] is True
    assert data["anomaly_score"] < data["threshold"]
    assert data["severity"] in ["MEDIUM_ANOMALY", "CRITICAL_ANOMALY"]


def test_predict_combined_critical_incident():
    """Test catastrophic multi-metric failure."""
    payload = {
        "request_duration_ms": 1650.0,
        "cpu_usage_pct": 92.0,
        "memory_usage_pct": 72.0,
        "requests_per_second": 240.0,
        "error_rate_pct": 32.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_anomaly"] is True
    assert data["anomaly_score"] < data["threshold"]
    assert data["severity"] == "CRITICAL_ANOMALY"
    assert data["rule_based_triggered"] is True


def test_batch_prediction():
    """Test batch prediction handles heterogeneous workloads."""
    batch_payload = {
        "samples": [
            {
                "request_duration_ms": 115.4,
                "cpu_usage_pct": 28.5,
                "memory_usage_pct": 46.2,
                "requests_per_second": 68.0,
                "error_rate_pct": 0.25,
            },
            {
                "request_duration_ms": 1550.0,
                "cpu_usage_pct": 45.0,
                "memory_usage_pct": 52.0,
                "requests_per_second": 110.0,
                "error_rate_pct": 1.8,
            },
            {
                "request_duration_ms": 1650.0,
                "cpu_usage_pct": 92.0,
                "memory_usage_pct": 72.0,
                "requests_per_second": 240.0,
                "error_rate_pct": 32.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_samples"] == 3
    assert data["normal_samples"] == 1
    assert data["anomalies_detected"] == 2
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["is_anomaly"] is False
    assert data["predictions"][1]["is_anomaly"] is True
    assert data["predictions"][2]["is_anomaly"] is True


def test_invalid_telemetry_validation():
    """Test Pydantic validation rejects negative or out-of-range metrics."""
    # Negative CPU
    bad_payload = {
        "request_duration_ms": 100.0,
        "cpu_usage_pct": -5.0,
        "memory_usage_pct": 45.0,
        "requests_per_second": 50.0,
        "error_rate_pct": 1.0,
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422

    # CPU > 100%
    bad_payload2 = {
        "request_duration_ms": 100.0,
        "cpu_usage_pct": 105.0,
        "memory_usage_pct": 45.0,
        "requests_per_second": 50.0,
        "error_rate_pct": 1.0,
    }
    res2 = client.post("/predict", json=bad_payload2)
    assert res2.status_code == 422
