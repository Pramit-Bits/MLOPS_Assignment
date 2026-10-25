"""Tests for the heart disease prediction API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_INPUT = {
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


@pytest.fixture
def client():
    """Create a test client with application lifespan enabled."""

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    """The root endpoint should describe the service."""

    response = client.get("/")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["service"] == (
        "Heart Disease Prediction API"
    )
    assert response_data["status"] == "running"
    assert response_data["documentation"] == "/docs"
    assert response_data["health_check"] == "/health"
    assert response_data["metrics"] == "/metrics"


def test_health_endpoint(client: TestClient) -> None:
    """The health endpoint should confirm model availability."""

    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "healthy"
    assert response_data["model_loaded"] is True
    assert response_data["api_version"] == "1.0.0"


def test_prediction_endpoint(client: TestClient) -> None:
    """A valid request should return a binary prediction."""

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["prediction"] in [0, 1]

    assert response_data["prediction_label"] in [
        "No heart disease",
        "Heart disease present",
    ]

    assert (
        0.0
        <= response_data["probability_no_heart_disease"]
        <= 1.0
    )

    assert (
        0.0
        <= response_data["probability_heart_disease"]
        <= 1.0
    )

    probability_sum = (
        response_data["probability_no_heart_disease"]
        + response_data["probability_heart_disease"]
    )

    assert probability_sum == pytest.approx(
        1.0,
        abs=0.00001,
    )


def test_prediction_with_missing_optional_values(
    client: TestClient,
) -> None:
    """The fitted pipeline should impute missing ca and thal values."""

    request_data = VALID_INPUT.copy()
    request_data["ca"] = None
    request_data["thal"] = None

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_invalid_age_is_rejected(
    client: TestClient,
) -> None:
    """An impossible age should fail input validation."""

    request_data = VALID_INPUT.copy()
    request_data["age"] = 500

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected(
    client: TestClient,
) -> None:
    """Missing required predictors should fail validation."""

    request_data = VALID_INPUT.copy()
    request_data.pop("chol")

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422


def test_invalid_category_is_rejected(
    client: TestClient,
) -> None:
    """Values outside a categorical field range should be rejected."""

    request_data = VALID_INPUT.copy()
    request_data["sex"] = 5

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient) -> None:
    """Prometheus metrics should be exposed."""

    response = client.get("/metrics/")

    assert response.status_code == 200

    assert (
        "heart_disease_prediction_requests_total"
        in response.text
    )
