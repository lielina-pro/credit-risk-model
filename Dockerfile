# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="10Academy KAIM9 Week4"
LABEL description="Credit Risk Prediction API"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY src/ ./src/
COPY mlflow.db ./mlflow.db

# Create data directory (for any runtime artifacts)
RUN mkdir -p data/processed data/raw

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────────────────────────────────────
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV MODEL_NAME="CreditRiskProxyModel"
ENV MODEL_STAGE="Staging"
ENV RISK_THRESHOLD="0.5"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
