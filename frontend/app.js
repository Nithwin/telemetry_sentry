/**
 * TelemetrySentry — Frontend Application Controller
 * Handles real-time telemetry streaming, interactive prediction requests to FastAPI,
 * dynamic scenario presets, and HTML5 Canvas visualization.
 */

// Presets mapping
const SCENARIO_PRESETS = {
    normal: {
        request_duration_ms: 115.0,
        cpu_usage_pct: 28.0,
        memory_usage_pct: 46.0,
        requests_per_second: 65.0,
        error_rate_pct: 0.25,
        desc: "Standard baseline operations under typical non-peak server traffic."
    },
    latency_spike: {
        request_duration_ms: 1550.0,
        cpu_usage_pct: 45.0,
        memory_usage_pct: 52.0,
        requests_per_second: 110.0,
        error_rate_pct: 1.8,
        desc: "Severe backend database locking / disk I/O bottleneck causing extreme latency."
    },
    traffic_surge: {
        request_duration_ms: 520.0,
        cpu_usage_pct: 78.0,
        memory_usage_pct: 68.0,
        requests_per_second: 290.0,
        error_rate_pct: 4.5,
        desc: "Flash crowd or marketing campaign driving sudden 3x throughput surge."
    },
    error_spike: {
        request_duration_ms: 290.0,
        cpu_usage_pct: 52.0,
        memory_usage_pct: 50.0,
        requests_per_second: 95.0,
        error_rate_pct: 29.5,
        desc: "Microservice 500 error cascade caused by downstream dependency failure."
    },
    subtle_incident: {
        request_duration_ms: 380.0,
        cpu_usage_pct: 62.0,
        memory_usage_pct: 55.0,
        requests_per_second: 125.0,
        error_rate_pct: 5.8,
        desc: "Correlated multi-metric drift where no single metric breaches static rules (The Hard Challenge)."
    },
    combined_incident: {
        request_duration_ms: 1650.0,
        cpu_usage_pct: 92.0,
        memory_usage_pct: 72.0,
        requests_per_second: 240.0,
        error_rate_pct: 32.0,
        desc: "Catastrophic infrastructure failure across CPU, latency, traffic, and error rates."
    }
};

// Static rule thresholds for client fallback
const STATIC_RULES = {
    request_duration_ms: 1000.0,
    cpu_usage_pct: 85.0,
    memory_usage_pct: 90.0,
    requests_per_second: 250.0,
    error_rate_pct: 10.0
};

// Application State
let appState = {
    apiConnected: false,
    apiUrl: window.location.origin,
    threshold: -0.11404,
    autoStreamInterval: null,
    history: []
};

// DOM Elements
const elements = {
    apiStatusDot: document.querySelector(".status-dot"),
    apiStatusText: document.getElementById("apiStatusText"),
    metaThreshold: document.getElementById("metaThreshold"),
    alertBanner: document.getElementById("alertBanner"),
    bannerVerdict: document.getElementById("bannerVerdict"),
    bannerSeverity: document.getElementById("bannerSeverity"),
    bannerSummary: document.getElementById("bannerSummary"),
    advantageBadge: document.getElementById("advantageBadge"),
    miniScore: document.getElementById("miniScore"),
    miniThreshold: document.getElementById("miniThreshold"),
    miniConfidence: document.getElementById("miniConfidence"),
    miniRule: document.getElementById("miniRule"),
    presetDescription: document.getElementById("presetDescription"),
    diagTableBody: document.getElementById("diagTableBody"),
    autoStreamCheckbox: document.getElementById("autoStreamCheckbox"),
    btnDiagnose: document.getElementById("btnDiagnose"),
    btnResetTelemetry: document.getElementById("btnResetTelemetry"),
    
    // Sliders & value displays
    inputLatency: document.getElementById("inputLatency"),
    valLatency: document.getElementById("valLatency"),
    inputCpu: document.getElementById("inputCpu"),
    valCpu: document.getElementById("valCpu"),
    inputMemory: document.getElementById("inputMemory"),
    valMemory: document.getElementById("valMemory"),
    inputRps: document.getElementById("inputRps"),
    valRps: document.getElementById("valRps"),
    inputError: document.getElementById("inputError"),
    valError: document.getElementById("valError"),
    
    // Canvas
    canvas: document.getElementById("telemetryCanvas")
};

// Canvas context
const ctx = elements.canvas.getContext("2d");

// Initialize application
async function init() {
    setupSliders();
    setupPresets();
    setupButtons();
    setupCanvas();
    
    await checkHealth();
    
    // Run initial diagnosis
    await runDiagnosis();
    
    // Seed initial history points
    seedHistory();
    renderChart();
}

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch(`${appState.apiUrl}/health`);
        if (res.ok) {
            const data = await res.json();
            appState.apiConnected = true;
            elements.apiStatusDot.classList.add("online");
            elements.apiStatusText.textContent = "FastAPI Online";
            if (data.metadata && data.metadata.threshold) {
                appState.threshold = data.metadata.threshold;
                elements.metaThreshold.textContent = data.metadata.threshold.toFixed(4);
                elements.miniThreshold.textContent = data.metadata.threshold.toFixed(4);
            }
            return;
        }
    } catch (e) {
        console.warn("Could not reach /health at", appState.apiUrl, e);
    }
    
    // Fallback if opened as standalone file
    elements.apiStatusText.textContent = "Simulation Mode";
    elements.metaThreshold.textContent = appState.threshold.toFixed(4);
    elements.miniThreshold.textContent = appState.threshold.toFixed(4);
}

// Setup Sliders
function setupSliders() {
    const bindSlider = (input, display, decimals = 1) => {
        input.addEventListener("input", () => {
            const val = parseFloat(input.value);
            display.textContent = val.toFixed(decimals);
            // Highlight custom changes
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            elements.presetDescription.textContent = "Custom user-modulated telemetry parameters.";
        });
    };

    bindSlider(elements.inputLatency, elements.valLatency, 1);
    bindSlider(elements.inputCpu, elements.valCpu, 1);
    bindSlider(elements.inputMemory, elements.valMemory, 1);
    bindSlider(elements.inputRps, elements.valRps, 0);
    bindSlider(elements.inputError, elements.valError, 2);
}

// Setup Presets
function setupPresets() {
    const buttons = document.querySelectorAll(".preset-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const key = btn.dataset.preset;
            const data = SCENARIO_PRESETS[key];
            if (!data) return;
            
            setTelemetryValues(data);
            elements.presetDescription.textContent = data.desc;
            runDiagnosis();
        });
    });
}

// Set telemetry slider values
function setTelemetryValues(data) {
    elements.inputLatency.value = data.request_duration_ms;
    elements.valLatency.textContent = Number(data.request_duration_ms).toFixed(1);
    
    elements.inputCpu.value = data.cpu_usage_pct;
    elements.valCpu.textContent = Number(data.cpu_usage_pct).toFixed(1);
    
    elements.inputMemory.value = data.memory_usage_pct;
    elements.valMemory.textContent = Number(data.memory_usage_pct).toFixed(1);
    
    elements.inputRps.value = data.requests_per_second;
    elements.valRps.textContent = Number(data.requests_per_second).toFixed(0);
    
    elements.inputError.value = data.error_rate_pct;
    elements.valError.textContent = Number(data.error_rate_pct).toFixed(2);
}

// Get current telemetry payload
function getTelemetryPayload() {
    return {
        request_duration_ms: parseFloat(elements.inputLatency.value),
        cpu_usage_pct: parseFloat(elements.inputCpu.value),
        memory_usage_pct: parseFloat(elements.inputMemory.value),
        requests_per_second: parseFloat(elements.inputRps.value),
        error_rate_pct: parseFloat(elements.inputError.value)
    };
}

// Setup Buttons
function setupButtons() {
    elements.btnDiagnose.addEventListener("click", () => runDiagnosis());
    
    elements.btnResetTelemetry.addEventListener("click", () => {
        const norm = SCENARIO_PRESETS.normal;
        setTelemetryValues(norm);
        document.querySelectorAll(".preset-btn").forEach(b => {
            b.classList.toggle("active", b.dataset.preset === "normal");
        });
        elements.presetDescription.textContent = norm.desc;
        runDiagnosis();
    });

    elements.autoStreamCheckbox.addEventListener("change", (e) => {
        if (e.target.checked) {
            startAutoStream();
        } else {
            stopAutoStream();
        }
    });
}

// Run Diagnostic Inference
async function runDiagnosis() {
    const payload = getTelemetryPayload();
    let result = null;

    if (appState.apiConnected) {
        try {
            const res = await fetch(`${appState.apiUrl}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                result = await res.json();
            }
        } catch (err) {
            console.warn("API inference error:", err);
        }
    }

    // Client-side fallback computation
    if (!result) {
        result = computeLocalPrediction(payload);
    }

    renderPrediction(result, payload);
    
    // Add to chart history
    addHistoryPoint(payload, result.is_anomaly);
    renderChart();
}

// Client-side fallback prediction logic
function computeLocalPrediction(p) {
    // Check static rules
    const ruleTriggered = (
        p.request_duration_ms > STATIC_RULES.request_duration_ms ||
        p.cpu_usage_pct > STATIC_RULES.cpu_usage_pct ||
        p.memory_usage_pct > STATIC_RULES.memory_usage_pct ||
        p.requests_per_second > STATIC_RULES.requests_per_second ||
        p.error_rate_pct > STATIC_RULES.error_rate_pct
    );

    // Approximate Isolation Forest score
    let score = 0.08;
    if (p.request_duration_ms > 1000) score -= (p.request_duration_ms - 1000) * 0.00015 + 0.15;
    if (p.error_rate_pct > 10) score -= (p.error_rate_pct - 10) * 0.008 + 0.12;
    if (p.cpu_usage_pct > 85) score -= (p.cpu_usage_pct - 85) * 0.006 + 0.05;
    if (p.requests_per_second > 250) score -= (p.requests_per_second - 250) * 0.001 + 0.08;

    // Subtle multi-drift
    if (p.request_duration_ms > 350 && p.cpu_usage_pct > 58 && p.error_rate_pct > 4.5) {
        score -= 0.22;
    }

    const isAnomaly = score < appState.threshold;
    const severity = isAnomaly ? (score < -0.22 ? "CRITICAL_ANOMALY" : "MEDIUM_ANOMALY") : "NORMAL";
    
    const analysis = [
        { metric: "request_duration_ms", value: p.request_duration_ms, status: p.request_duration_ms > 1000 ? "CRITICAL" : (p.request_duration_ms > 450 ? "ELEVATED" : "NORMAL"), rule_threshold_exceeded: p.request_duration_ms > 1000 },
        { metric: "cpu_usage_pct", value: p.cpu_usage_pct, status: p.cpu_usage_pct > 85 ? "CRITICAL" : (p.cpu_usage_pct > 70 ? "ELEVATED" : "NORMAL"), rule_threshold_exceeded: p.cpu_usage_pct > 85 },
        { metric: "memory_usage_pct", value: p.memory_usage_pct, status: p.memory_usage_pct > 90 ? "CRITICAL" : (p.memory_usage_pct > 75 ? "ELEVATED" : "NORMAL"), rule_threshold_exceeded: p.memory_usage_pct > 90 },
        { metric: "requests_per_second", value: p.requests_per_second, status: p.requests_per_second > 250 ? "CRITICAL" : (p.requests_per_second > 180 ? "ELEVATED" : "NORMAL"), rule_threshold_exceeded: p.requests_per_second > 250 },
        { metric: "error_rate_pct", value: p.error_rate_pct, status: p.error_rate_pct > 10 ? "CRITICAL" : (p.error_rate_pct > 3 ? "ELEVATED" : "NORMAL"), rule_threshold_exceeded: p.error_rate_pct > 10 }
    ];

    return {
        is_anomaly: isAnomaly,
        anomaly_score: score,
        threshold: appState.threshold,
        severity: severity,
        confidence: isAnomaly ? 0.95 : 0.98,
        rule_based_triggered: ruleTriggered,
        summary: isAnomaly ? `Anomaly detected (${severity}); multi-metric deviation flagged.` : "Telemetry profile within standard operational parameters.",
        metric_analysis: analysis
    };
}

// Render Prediction Result
function renderPrediction(res, payload) {
    const isAnomaly = res.is_anomaly;
    
    // Alert Banner classes
    elements.alertBanner.className = `alert-banner ${isAnomaly ? 'status-anomaly' : 'status-normal'}`;
    elements.bannerVerdict.textContent = isAnomaly ? `ANOMALY DETECTED — ${res.severity}` : `SYSTEM NORMAL — ${res.severity}`;
    elements.bannerSeverity.textContent = res.severity;
    elements.bannerSeverity.className = `severity-pill ${isAnomaly ? 'pill-anomaly' : 'pill-normal'}`;
    elements.bannerSummary.textContent = res.summary;

    // ML Advantage Badge
    if (isAnomaly && !res.rule_based_triggered) {
        elements.advantageBadge.classList.remove("hidden");
    } else {
        elements.advantageBadge.classList.add("hidden");
    }

    // Mini Metrics
    elements.miniScore.textContent = res.anomaly_score.toFixed(4);
    elements.miniThreshold.textContent = res.threshold.toFixed(4);
    elements.miniConfidence.textContent = `${(res.confidence * 100).toFixed(1)}%`;
    elements.miniRule.textContent = res.rule_based_triggered ? "BREACHED" : "PASS";
    elements.miniRule.style.color = res.rule_based_triggered ? "var(--warning)" : "var(--success)";

    // Diagnostic Table
    elements.diagTableBody.innerHTML = "";
    res.metric_analysis.forEach(item => {
        const tr = document.createElement("tr");
        
        const metricNames = {
            request_duration_ms: "Request Duration (Latency)",
            cpu_usage_pct: "CPU Utilization",
            memory_usage_pct: "Memory Utilization",
            requests_per_second: "Requests Per Second (RPS)",
            error_rate_pct: "Error Rate (5xx)"
        };

        const units = {
            request_duration_ms: "ms",
            cpu_usage_pct: "%",
            memory_usage_pct: "%",
            requests_per_second: "rps",
            error_rate_pct: "%"
        };

        const badgeClass = item.status.toLowerCase();
        const ruleCheck = item.rule_threshold_exceeded ? "BREACHED" : "NORMAL";
        const ruleColor = item.rule_threshold_exceeded ? "var(--danger)" : "var(--text-muted)";

        tr.innerHTML = `
            <td><strong>${metricNames[item.metric] || item.metric}</strong></td>
            <td><code>${item.value} ${units[item.metric] || ''}</code></td>
            <td><span class="status-badge ${badgeClass}">${item.status}</span></td>
            <td style="color: ${ruleColor}; font-weight: 500;">${ruleCheck}</td>
        `;
        elements.diagTableBody.appendChild(tr);
    });
}

// Canvas & Real-time Chart
function setupCanvas() {
    function resize() {
        const rect = elements.canvas.parentElement.getBoundingClientRect();
        elements.canvas.width = rect.width * window.devicePixelRatio;
        elements.canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        renderChart();
    }
    window.addEventListener("resize", resize);
    resize();
}

function seedHistory() {
    for (let i = 0; i < 30; i++) {
        const t = i / 30;
        const lat = 110 + Math.sin(t * Math.PI * 4) * 20 + Math.random() * 15;
        const cpu = 25 + Math.cos(t * Math.PI * 2) * 8 + Math.random() * 5;
        appState.history.push({
            latency: lat,
            cpu: cpu,
            isAnomaly: false
        });
    }
}

function addHistoryPoint(payload, isAnomaly) {
    appState.history.push({
        latency: payload.request_duration_ms,
        cpu: payload.cpu_usage_pct,
        isAnomaly: isAnomaly
    });
    if (appState.history.length > 40) {
        appState.history.shift();
    }
}

function renderChart() {
    const width = elements.canvas.parentElement.clientWidth;
    const height = elements.canvas.parentElement.clientHeight;
    
    ctx.clearRect(0, 0, width, height);

    // Background Grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 30; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    if (appState.history.length < 2) return;

    const step = width / (appState.history.length - 1);

    // 1. Draw Latency Line (Purple/Indigo)
    const maxLat = 2000;
    ctx.beginPath();
    appState.history.forEach((pt, i) => {
        const x = i * step;
        const normLat = Math.min(pt.latency / maxLat, 1);
        const y = height - (normLat * (height - 40) + 20);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 2. Draw CPU Line (Cyan)
    const maxCpu = 100;
    ctx.beginPath();
    appState.history.forEach((pt, i) => {
        const x = i * step;
        const normCpu = Math.min(pt.cpu / maxCpu, 1);
        const y = height - (normCpu * (height - 40) + 20);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 3. Draw Anomaly Markers
    appState.history.forEach((pt, i) => {
        if (pt.isAnomaly) {
            const x = i * step;
            // Draw vertical glow marker
            ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
            ctx.fillRect(x - 6, 0, 12, height);

            ctx.beginPath();
            ctx.arc(x, 25, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#f43f5e";
            ctx.shadowColor = "#f43f5e";
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    });
}

// Auto-Stream Simulation
let autoTick = 0;
function startAutoStream() {
    if (appState.autoStreamInterval) clearInterval(appState.autoStreamInterval);
    
    appState.autoStreamInterval = setInterval(() => {
        autoTick++;
        // Cyclical server load
        const wave = 0.5 + 0.3 * Math.sin(autoTick * 0.2);
        
        let lat = 90 + wave * 90 + (Math.random() - 0.5) * 20;
        let cpu = 15 + wave * 30 + (Math.random() - 0.5) * 8;
        let mem = 45 + (autoTick % 50) * 0.2;
        let rps = 40 + wave * 70 + (Math.random() - 0.5) * 10;
        let err = 0.2 + (Math.random() * 0.4);

        // Every 10 ticks, inject an incident
        if (autoTick % 12 === 0) {
            lat += 1200; // latency spike
        } else if (autoTick % 19 === 0) {
            cpu += 60; // cpu spike
        } else if (autoTick % 25 === 0) {
            lat += 250; cpu += 25; err += 5.0; // subtle multi-drift!
        }

        lat = Math.max(20, Math.min(2200, lat));
        cpu = Math.max(0, Math.min(100, cpu));
        mem = Math.max(0, Math.min(100, mem));
        rps = Math.max(5, Math.min(380, rps));
        err = Math.max(0, Math.min(100, err));

        setTelemetryValues({
            request_duration_ms: lat,
            cpu_usage_pct: cpu,
            memory_usage_pct: mem,
            requests_per_second: rps,
            error_rate_pct: err
        });

        runDiagnosis();
    }, 1500);
}

function stopAutoStream() {
    if (appState.autoStreamInterval) {
        clearInterval(appState.autoStreamInterval);
        appState.autoStreamInterval = null;
    }
}

// Boot on DOM ready
document.addEventListener("DOMContentLoaded", init);
