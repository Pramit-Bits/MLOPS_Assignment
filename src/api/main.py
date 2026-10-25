"""
FastAPI service for heart disease model inference.

Run from the repository root with:

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

API_TITLE = "Heart Disease Prediction API"
API_VERSION = "1.0.0"


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("heart-disease-api")


# -------------------------------------------------------------------
# Prometheus metrics
# -------------------------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "heart_disease_prediction_requests_total",
    "Total number of prediction requests.",
)

PREDICTION_ERRORS = Counter(
    "heart_disease_prediction_errors_total",
    "Total number of failed prediction requests.",
)

PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_duration_seconds",
    "Time spent processing prediction requests.",
)

PREDICTED_CLASSES = Counter(
    "heart_disease_predicted_classes_total",
    "Number of predictions by predicted class.",
    labelnames=["predicted_class"],
)


# -------------------------------------------------------------------
# Request and response models
# -------------------------------------------------------------------

class HeartDiseaseInput(BaseModel):
    """
    Input schema for a single heart disease prediction.

    Categorical attributes use the numerical codes provided by the
    original UCI Heart Disease dataset.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }
    )

    age: Annotated[
        float,
        Field(
            ge=1,
            le=120,
            description="Age in years.",
        ),
    ]

    sex: Annotated[
        int,
        Field(
            ge=0,
            le=1,
            description="Sex code: 0 or 1.",
        ),
    ]

    cp: Annotated[
        int,
        Field(
            ge=1,
            le=4,
            description="Chest-pain type code from 1 to 4.",
        ),
    ]

    trestbps: Annotated[
        float,
        Field(
            ge=50,
            le=300,
            description="Resting blood pressure in mm Hg.",
        ),
    ]

    chol: Annotated[
        float,
        Field(
            ge=50,
            le=700,
            description="Serum cholesterol in mg/dl.",
        ),
    ]

    fbs: Annotated[
        int,
        Field(
            ge=0,
            le=1,
            description="Fasting blood sugar category: 0 or 1.",
        ),
    ]

    restecg: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            description="Resting electrocardiographic result code.",
        ),
    ]

    thalach: Annotated[
        float,
        Field(
            ge=40,
            le=250,
            description="Maximum heart rate achieved.",
        ),
    ]

    exang: Annotated[
        int,
        Field(
            ge=0,
            le=1,
            description="Exercise-induced angina category: 0 or 1.",
        ),
    ]

    oldpeak: Annotated[
        float,
        Field(
            ge=0,
            le=10,
            description="ST depression induced by exercise.",
        ),
    ]

    slope: Annotated[
        int,
        Field(
            ge=1,
            le=3,
            description="Slope category from 1 to 3.",
        ),
    ]

    ca: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            le=4,
            description=(
                "Number of major vessels. Null is allowed "
                "because the model pipeline includes imputation."
            ),
        ),
    ]

    thal: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Thalassemia-related category. Null is allowed "
                "because the model pipeline includes imputation."
            ),
        ),
    ]


class PredictionResponse(BaseModel):
    """Response schema for a successful prediction."""

    prediction: int
    prediction_label: str
    probability_no_heart_disease: float
    probability_heart_disease: float
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Response schema for application health checks."""

    status: str
    model_loaded: bool
    model_path: str
    api_version: str


class RootResponse(BaseModel):
    """Response schema for the root endpoint."""

    service: str
    version: str
    status: str
    documentation: str
    health_check: str
    metrics: str


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

model = None


def load_model():
    """Load and return the serialized model pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at {MODEL_PATH}. "
            "Run python src/models/train.py first."
        )

    loaded_model = joblib.load(MODEL_PATH)

    required_methods = [
        "predict",
        "predict_proba",
    ]

    for method_name in required_methods:
        if not hasattr(loaded_model, method_name):
            raise TypeError(
                f"Loaded model does not provide {method_name}()."
            )

    return loaded_model


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model when the API starts."""

    global model

    logger.info("Starting %s", API_TITLE)

    try:
        model = load_model()
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception:
        logger.exception("Model loading failed.")
        model = None

    yield

    logger.info("Stopping %s", API_TITLE)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Educational model-serving API for predicting the binary "
        "heart disease target from the UCI Heart Disease dataset. "
        "This service is not intended for clinical use."
    ),
    lifespan=lifespan,
)


# Mount Prometheus metrics as an ASGI endpoint.
metrics_application = make_asgi_app()
app.mount("/metrics", metrics_application)


# -------------------------------------------------------------------
# Request logging middleware
# -------------------------------------------------------------------

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    """Record request path, method, status code, and latency."""

    start_time = time.perf_counter()

    response = await call_next(request)

    duration_seconds = time.perf_counter() - start_time

    logger.info(
        "method=%s path=%s status=%s duration_seconds=%.4f",
        request.method,
        request.url.path,
        response.status_code,
        duration_seconds,
    )

    return response


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    tags=["Service"],
)
def root() -> RootResponse:
    """Return service information."""

    return RootResponse(
        service=API_TITLE,
        version=API_VERSION,
        status="running",
        documentation="/docs",
        health_check="/health",
        metrics="/metrics",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
)
def health() -> HealthResponse:
    """Return service and model health information."""

    model_is_loaded = model is not None

    return HealthResponse(
        status="healthy" if model_is_loaded else "unhealthy",
        model_loaded=model_is_loaded,
        model_path=str(MODEL_PATH),
        api_version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    input_data: HeartDiseaseInput,
) -> PredictionResponse:
    """
    Predict the binary heart disease target and return confidence.

    This output is for an educational machine-learning demonstration
    and must not be interpreted as medical advice.
    """

    if model is None:
        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=503,
            detail=(
                "The model is unavailable. "
                "Check the service logs and model artifact."
            ),
        )

    start_time = time.perf_counter()
    PREDICTION_REQUESTS.inc()

    try:
        input_dataframe = pd.DataFrame(
            [input_data.model_dump()]
        )

        prediction = int(
            model.predict(input_dataframe)[0]
        )

        probabilities = model.predict_proba(
            input_dataframe
        )[0]

        probability_no_heart_disease = float(
            probabilities[0]
        )

        probability_heart_disease = float(
            probabilities[1]
        )

        prediction_labels = {
            0: "No heart disease",
            1: "Heart disease present",
        }

        PREDICTED_CLASSES.labels(
            predicted_class=str(prediction)
        ).inc()

        logger.info(
            "prediction=%s probability_heart_disease=%.4f",
            prediction,
            probability_heart_disease,
        )

        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_labels[prediction],
            probability_no_heart_disease=round(
                probability_no_heart_disease,
                6,
            ),
            probability_heart_disease=round(
                probability_heart_disease,
                6,
            ),
            model_version=API_VERSION,
            disclaimer=(
                "Educational demonstration only. "
                "Not intended for clinical diagnosis."
            ),
        )

    except Exception as error:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from error

    finally:
        duration_seconds = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(duration_seconds)
