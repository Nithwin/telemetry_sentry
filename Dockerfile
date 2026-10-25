# TelemetrySentry Production Container
FROM python:3.12-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, model artifacts, data, and frontend assets
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY frontend/ ./frontend/

# Expose API and frontend dashboard port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application serving both API and HTML5 Dashboard
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
