#%%

import numpy as np
import pandas as pd

# 1. Reproducible random data
rng = np.random.default_rng(42)
n = 10_000

# One observation every 10 seconds
timestamps = pd.date_range(
    start="2026-01-01",
    periods=n,
    freq="10s"
)

# 2. Simulate changing server workload
t = np.arange(n)

workload = (
    0.5
    + 0.22 * np.sin(2 * np.pi * t / 8640 - 1.2)
    + 0.08 * np.sin(2 * np.pi * t / 1440)
)
workload = np.clip(workload, 0.08, 0.95)

# 3. Generate normal server metrics
requests = np.clip(
    rng.normal(45 + 120 * workload, 12 + 18 * workload, n),
    5, None
)

cpu = np.clip(
    12 + 0.30 * requests + rng.normal(0, 7, n),
    3, 88
)

memory = np.clip(
    42 + 0.018 * np.arange(n) / n * 100
    + 0.04 * requests + rng.normal(0, 3, n),
    25, 85
)

latency = np.clip(
    75 + 1.2 * requests + 0.9 * cpu
    + rng.normal(0, 28, n),
    30, None
)

error_rate = np.clip(
    rng.beta(1.2, 80, n) * 100,
    0, 5
)

# 4. Ground-truth labels
is_anomaly = np.zeros(n, dtype=int)
incident_type = np.full(n, "normal", dtype=object)

# Each incident is a contiguous time window
incidents = [
    (900, 960, "cpu_spike"),
    (2100, 2160, "latency_spike"),
    (3500, 3570, "error_spike"),
    (5100, 5180, "traffic_surge"),
    (6800, 6860, "combined_incident"),
    (8300, 8370, "combined_incident"),
]

for start, end, kind in incidents:
    is_anomaly[start:end] = 1
    incident_type[start:end] = kind

    if kind == "cpu_spike":
        cpu[start:end] = np.clip(
            cpu[start:end] + rng.normal(40, 6, end-start),
            0, 100
        )

    elif kind == "latency_spike":
        latency[start:end] += rng.normal(
            1200, 180, end-start
        )

    elif kind == "error_spike":
        error_rate[start:end] = np.clip(
            rng.normal(28, 5, end-start),
            0, 100
        )

    elif kind == "traffic_surge":
        requests[start:end] *= rng.normal(
            2.4, 0.15, end-start
        )
        cpu[start:end] = np.clip(
            cpu[start:end] + rng.normal(20, 5, end-start),
            0, 100
        )
        latency[start:end] += rng.normal(
            250, 70, end-start
        )

    elif kind == "combined_incident":
        cpu[start:end] = np.clip(
            cpu[start:end] + rng.normal(45, 7, end-start),
            0, 100
        )
        latency[start:end] += rng.normal(
            1500, 250, end-start
        )
        error_rate[start:end] = np.clip(
            rng.normal(35, 6, end-start),
            0, 100
        )

# 5. Assemble the dataset
df = pd.DataFrame({
    "timestamp": timestamps,
    "request_duration_ms": np.round(latency, 2),
    "cpu_usage_pct": np.round(cpu, 2),
    "memory_usage_pct": np.round(memory, 2),
    "requests_per_second": np.round(requests, 2),
    "error_rate_pct": np.round(error_rate, 2),
    "is_anomaly": is_anomaly,
    "incident_type": incident_type,
})

# 6. Save it
df.to_csv("telemetry_data.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("\nLabel distribution:")
print(df["is_anomaly"].value_counts())
print("\nIncident types:")
print(df["incident_type"].value_counts())

df.head()
