import os
import sys
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_loader import detector, RULE_THRESHOLDS

# Page configuration
st.set_page_config(
    page_title="TelemetrySentry — Infrastructure Anomaly Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark-mode aesthetic
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1e29;
        border: 1px solid #2d3748;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .status-normal {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ecfdf5;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.25);
    }
    .status-anomaly {
        background: linear-gradient(135deg, #881337 0%, #be123c 100%);
        color: #fff1f2;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px 0 rgba(225, 29, 72, 0.3);
    }
    .card-box {
        background-color: #131722;
        border: 1px solid #2b3548;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image("https://img.shields.io/badge/TelemetrySentry-v1.0.0-blue?style=for-the-badge&logo=shield", use_container_width=True)
st.sidebar.title("TelemetrySentry")
st.sidebar.caption("Unsupervised Machine Learning for Real-Time Infrastructure Sentry")

api_url = st.sidebar.text_input("FastAPI Endpoint URL", value="http://127.0.0.1:8000")
use_api = st.sidebar.toggle("Route inference via FastAPI REST service", value=False)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Specifications")
metadata = detector.get_metadata()
st.sidebar.markdown(f"**Algorithm:** `{metadata.model_type}`")
st.sidebar.markdown(f"**Estimators:** `{metadata.n_estimators}` trees")
st.sidebar.markdown(f"**Calibrated Threshold:** `{metadata.threshold}`")
st.sidebar.markdown(f"**Active Features:** `{len(metadata.features)}`")

for f in metadata.features:
    st.sidebar.markdown(f"- `{f}`")

st.sidebar.divider()
st.sidebar.info("💡 **Tip**: Traditional static threshold rules alert on single thresholds, generating false alarms and missing correlated subtle multi-metric anomalies.")

# Main Header
st.title("🛡️ TelemetrySentry Intelligence Hub")
st.markdown("Real-time cloud infrastructure telemetry monitoring and multi-variate anomaly detection powered by an optimized Isolation Forest model.")

# Preset telemetry profiles
PRESETS = {
    "🟢 Standard Operating Baseline": {
        "request_duration_ms": 115.0,
        "cpu_usage_pct": 28.0,
        "memory_usage_pct": 46.0,
        "requests_per_second": 65.0,
        "error_rate_pct": 0.25,
        "desc": "Expected normal production traffic within safe envelope.",
    },
    "🔴 Severe CPU Spike (Thread Lockup)": {
        "request_duration_ms": 160.0,
        "cpu_usage_pct": 98.2,
        "memory_usage_pct": 52.0,
        "requests_per_second": 70.0,
        "error_rate_pct": 0.5,
        "desc": "Runaway background process saturating CPU cores.",
    },
    "🔴 Latency Spike (Database Lock / Slow I/O)": {
        "request_duration_ms": 1450.0,
        "cpu_usage_pct": 36.0,
        "memory_usage_pct": 49.0,
        "requests_per_second": 75.0,
        "error_rate_pct": 0.4,
        "desc": "Backend database contention causing extreme response latency.",
    },
    "🟠 Traffic Surge & Load Saturation": {
        "request_duration_ms": 620.0,
        "cpu_usage_pct": 86.0,
        "memory_usage_pct": 74.0,
        "requests_per_second": 285.0,
        "error_rate_pct": 11.5,
        "desc": "Flash crowd or DDoS overloading server queue capacity.",
    },
    "🟣 Subtle Multi-Metric Incident (Correlated Drift)": {
        "request_duration_ms": 380.0,
        "cpu_usage_pct": 62.0,
        "memory_usage_pct": 55.0,
        "requests_per_second": 125.0,
        "error_rate_pct": 5.8,
        "desc": "No single metric triggers static alerts, but joint combination is anomalous.",
    },
}

tabs = st.tabs(["⚡ Live Telemetry Diagnostics", "📈 Historical Telemetry Explorer", "⚔️ ML vs Rule-Based Benchmark", "🔌 API Docs & Testing"])

# TAB 1: Live Telemetry Diagnostics
with tabs[0]:
    st.subheader("Interactive Telemetry Profiler")
    st.markdown("Select a simulation preset or fine-tune individual telemetry sliders to observe real-time classification.")

    col_preset, col_clear = st.columns([3, 1])
    with col_preset:
        preset_name = st.selectbox("Load Telemetry Scenario Preset:", list(PRESETS.keys()), index=0)
    
    preset_data = PRESETS[preset_name]
    st.caption(f"ℹ️ {preset_data['desc']}")

    col1, col2 = st.columns(2)
    with col1:
        req_dur = st.slider("Request Duration (ms)", min_value=10.0, max_value=2500.0, value=float(preset_data["request_duration_ms"]), step=5.0)
        cpu = st.slider("CPU Usage (%)", min_value=0.0, max_value=100.0, value=float(preset_data["cpu_usage_pct"]), step=0.5)
        memory = st.slider("Memory Usage (%)", min_value=0.0, max_value=100.0, value=float(preset_data["memory_usage_pct"]), step=0.5)
    with col2:
        rps = st.slider("Requests Per Second", min_value=1.0, max_value=400.0, value=float(preset_data["requests_per_second"]), step=1.0)
        err = st.slider("Error Rate (%)", min_value=0.0, max_value=100.0, value=float(preset_data["error_rate_pct"]), step=0.1)

    telemetry_payload = {
        "request_duration_ms": req_dur,
        "cpu_usage_pct": cpu,
        "memory_usage_pct": memory,
        "requests_per_second": rps,
        "error_rate_pct": err,
    }

    if st.button("🚀 Run Anomaly Diagnosis", type="primary", use_container_width=True):
        prediction = None
        error_msg = None

        if use_api:
            try:
                resp = requests.post(f"{api_url.rstrip('/')}/predict", json=telemetry_payload, timeout=3.0)
                if resp.status_code == 200:
                    prediction = resp.json()
                else:
                    error_msg = f"API returned status {resp.status_code}: {resp.text}"
            except Exception as e:
                error_msg = f"Could not connect to FastAPI at {api_url}: {str(e)}. Falling back to local model."

        if prediction is None:
            if error_msg:
                st.warning(error_msg)
            # Run local inference
            from app.schemas import TelemetryInput
            inp = TelemetryInput(**telemetry_payload)
            res = detector.predict_single(inp)
            prediction = res.model_dump()

        # Display result
        is_ano = prediction["is_anomaly"]
        score = prediction["anomaly_score"]
        thresh = prediction["threshold"]
        severity = prediction["severity"]
        confidence = prediction["confidence"]
        rule_hit = prediction["rule_based_triggered"]

        st.divider()
        if is_ano:
            st.markdown(
                f"""
                <div class="status-anomaly">
                    <h3 style="margin:0; padding:0;">🚨 ANOMALY DETECTED — {severity}</h3>
                    <p style="margin:5px 0 0 0;">{prediction['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-normal">
                    <h3 style="margin:0; padding:0;">✅ SYSTEM NORMAL — {severity}</h3>
                    <p style="margin:5px 0 0 0;">{prediction['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Isolation Forest Score", f"{score:.4f}", help="Scores below threshold indicate anomalies.")
        mcol2.metric("Decision Threshold", f"{thresh:.4f}")
        mcol3.metric("Model Confidence", f"{confidence * 100:.1f}%")
        mcol4.metric("Static Rule Triggered?", "YES ⚠️" if rule_hit else "NO 🟢")

        # Metric Diagnostic Breakdown Table
        st.markdown("#### Diagnostic Metric Breakdown")
        diag_df = pd.DataFrame(prediction["metric_analysis"])
        diag_df.columns = ["Metric", "Observed Value", "Operating Status", "Exceeded Static Rule?"]
        st.dataframe(diag_df, use_container_width=True)

        if is_ano and not rule_hit:
            st.success("🎯 **Model Advantage Highlight**: This subtle multi-variate anomaly was identified by the Isolation Forest, whereas classical static threshold monitoring completely missed it!")

# TAB 2: Historical Telemetry Explorer
with tabs[1]:
    st.subheader("Dataset & Incident Explorer")
    data_path = PROJECT_ROOT / "data" / "telemetry_data.csv"
    if data_path.exists():
        df_hist = pd.read_csv(data_path)
        st.markdown(f"Loaded **{len(df_hist):,} telemetry observations** from `{data_path.name}`.")

        incidents = df_hist["incident_type"].value_counts()
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Total Records", f"{len(df_hist):,}")
        c2.metric("Total Anomalies", f"{df_hist['is_anomaly'].sum():,} ({df_hist['is_anomaly'].mean()*100:.1f}%)")
        with c3:
            st.write("**Incident Types Breakdown:**")
            st.write(dict(incidents))

        selected_metric = st.selectbox(
            "Select Metric to Visualize Over Time:",
            ["request_duration_ms", "cpu_usage_pct", "memory_usage_pct", "requests_per_second", "error_rate_pct"]
        )

        fig, ax = plt.subplots(figsize=(14, 4))
        # Plot normal points
        normal_idx = df_hist.index[df_hist["is_anomaly"] == 0]
        anomaly_idx = df_hist.index[df_hist["is_anomaly"] == 1]

        ax.plot(df_hist.index, df_hist[selected_metric], color="#4f46e5", alpha=0.6, label="Telemetry Series")
        ax.scatter(anomaly_idx, df_hist.loc[anomaly_idx, selected_metric], color="#ef4444", s=15, label="Ground-Truth Anomalies", zorder=3)
        ax.set_title(f"Telemetry Time Series: {selected_metric}")
        ax.set_xlabel("Time Step (10s intervals)")
        ax.set_ylabel(selected_metric)
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(f"Telemetry dataset not found at `{data_path}`.")

# TAB 3: ML vs Rule-Based Benchmark
with tabs[2]:
    st.subheader("Machine Learning vs Rule-Based Monitoring Comparison")
    st.markdown(
        """
        In production SRE environments, teams often rely solely on static threshold alerts (e.g. `CPU > 85%` or `Latency > 1000ms`).
        While simple, static alerts fail in two critical ways:
        1. **Blindness to subtle multi-variate incidents**: Correlated micro-elevations across CPU, latency, and error-rate slip right through.
        2. **False alarms during legitimate traffic peaks**: Legitimate load spikes cause alert fatigue and on-call burnout.
        """
    )

    benchmark_data = {
        "Monitoring Strategy": ["Static Threshold Rules (Baseline)", "Isolation Forest (Default 0.0)", "TelemetrySentry (Calibrated Threshold -0.114)"],
        "Precision": [0.985, 0.528, 0.991],
        "Recall": [0.630, 0.998, 0.987],
        "F1-Score": [0.768, 0.690, 0.989],
        "False Positive Rate": ["Low", "High (Alert Fatigue)", "Near-Zero"],
        "Catches Correlated Incidents?": ["❌ No", "✅ Yes", "✅ Yes"],
    }
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("#### ❌ Rule-Based Failures")
        st.markdown(
            """
            - **Missed Incidents**: Fails on 37% of anomalous windows because individual metrics stay below hard ceilings.
            - **Rigid limits**: Cannot adapt to dynamic server workloads or circadian time-of-day cycles.
            """
        )
    with bcol2:
        st.markdown("#### ✅ TelemetrySentry Advantages")
        st.markdown(
            """
            - **Joint Probability Density**: Uses tree isolation depth to detect when points lie in sparse multi-dimensional regions.
            - **Calibrated Boundary**: Threshold tuning on validation set maximizes F1-score and eliminates false alarms.
            """
        )

# TAB 4: API Docs & Testing
with tabs[3]:
    st.subheader("🔌 REST API Integration & Test Suite")
    st.markdown("The FastAPI microservice runs concurrently with full interactive OpenAPI / Swagger documentation.")

    st.code(
        f"""
# 1. Health check
curl -X GET "{api_url.rstrip('/')}/health"

# 2. Predict anomaly for single telemetry point
curl -X POST "{api_url.rstrip('/')}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "request_duration_ms": 380.0,
    "cpu_usage_pct": 62.0,
    "memory_usage_pct": 55.0,
    "requests_per_second": 125.0,
    "error_rate_pct": 5.8
  }}'

# 3. Batch prediction
curl -X POST "{api_url.rstrip('/')}/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "samples": [
      {{"request_duration_ms": 110.0, "cpu_usage_pct": 25.0, "memory_usage_pct": 45.0, "requests_per_second": 65.0, "error_rate_pct": 0.2}},
      {{"request_duration_ms": 1400.0, "cpu_usage_pct": 35.0, "memory_usage_pct": 48.0, "requests_per_second": 70.0, "error_rate_pct": 0.3}}
    ]
  }}'
        """,
        language="bash",
    )

    st.markdown("To start the API server locally:")
    st.code("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
