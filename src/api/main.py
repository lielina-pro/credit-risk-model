"""
Credit Risk Prediction API
FastAPI service that loads the best model from MLflow registry
and exposes prediction endpoints.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Or via Docker:
    docker-compose up
"""
import os
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
)
MODEL_NAME = os.getenv("MODEL_NAME", "CreditRiskProxyModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_model_version = "unknown"


def load_model() -> None:
    """Load the registered model from MLflow."""
    global _model, _model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from: %s", model_uri)
    try:
        _model = mlflow.sklearn.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        _model_version = versions[0].version if versions else "unknown"
        logger.info(
            "Model '%s' v%s loaded successfully.", MODEL_NAME, _model_version
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(f"Model load failed: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Predicts credit risk probability for customers using a "
        "machine-learning model trained on Xente transaction data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def features_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a Pydantic model instance to a single-row DataFrame."""
    return pd.DataFrame([customer.model_dump()])


def make_prediction(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference.
    Returns (probabilities, labels) arrays.
    """
    if _model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Try again shortly."
        )
    try:
        proba = _model.predict_proba(df)[:, 1]
        labels = (proba >= THRESHOLD).astype(int)
        return proba, labels
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=422, detail=f"Prediction failed: {exc}"
        ) from exc


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns service health and loaded model information."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_name=MODEL_NAME,
        model_version=_model_version,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict credit risk for a single customer.

    Returns the probability of being high-risk and a binary label.
    """
    df = features_to_df(customer)
    proba, labels = make_prediction(df)

    return PredictionResponse(
        risk_probability=float(proba[0]),
        risk_label="high_risk" if labels[0] == 1 else "low_risk",
        threshold_used=THRESHOLD,
        model_version=_model_version,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit risk for a batch of customers (max 1000).
    """
    df = pd.DataFrame([c.model_dump() for c in request.customers])
    proba, labels = make_prediction(df)

    predictions = [
        PredictionResponse(
            risk_probability=float(p),
            risk_label="high_risk" if l == 1 else "low_risk",
            threshold_used=THRESHOLD,
            model_version=_model_version,
        )
        for p, l in zip(proba, labels)
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=int(labels.sum()),
        low_risk_count=int((labels == 0).sum()),
    )


@app.get("/", tags=["Health"])
def root():
    """Redirect hint to docs."""
    return {
        "message": "Credit Risk API is running.",
        "docs": "/docs",
        "health": "/health",
    }
