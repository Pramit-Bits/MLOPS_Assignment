"""Tests for the serialized machine-learning pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_cleaned.csv"
)

EXPECTED_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Load the processed dataset."""

    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def model():
    """Load the serialized model pipeline."""

    return joblib.load(MODEL_PATH)


def test_processed_dataset_exists() -> None:
    """The processed dataset should be available."""

    assert DATA_PATH.exists()


def test_model_artifact_exists() -> None:
    """The serialized model should be available."""

    assert MODEL_PATH.exists()


def test_dataset_contains_required_columns(
    dataset: pd.DataFrame,
) -> None:
    """The dataset should contain all predictors and the target."""

    required_columns = set(
        EXPECTED_FEATURES + ["target"]
    )

    assert required_columns.issubset(
        set(dataset.columns)
    )


def test_target_is_binary(
    dataset: pd.DataFrame,
) -> None:
    """The target should contain only binary values."""

    assert set(dataset["target"].unique()).issubset(
        {0, 1}
    )


def test_target_has_no_missing_values(
    dataset: pd.DataFrame,
) -> None:
    """The target should not contain missing values."""

    assert dataset["target"].notna().all()


def test_model_produces_binary_prediction(
    model,
    dataset: pd.DataFrame,
) -> None:
    """The model should produce a binary prediction."""

    sample = dataset[EXPECTED_FEATURES].iloc[[0]]

    prediction = int(model.predict(sample)[0])

    assert prediction in [0, 1]


def test_model_produces_valid_probabilities(
    model,
    dataset: pd.DataFrame,
) -> None:
    """The model should produce two probabilities summing to one."""

    sample = dataset[EXPECTED_FEATURES].iloc[[0]]

    probabilities = model.predict_proba(sample)[0]

    assert len(probabilities) == 2
    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities
    )
    assert probabilities.sum() == pytest.approx(
        1.0,
        abs=0.00001,
    )


def test_model_handles_expected_missing_values(
    model,
    dataset: pd.DataFrame,
) -> None:
    """The pipeline should impute missing ca and thal values."""

    sample = dataset[EXPECTED_FEATURES].iloc[[0]].copy()

    sample.loc[:, "ca"] = np.nan
    sample.loc[:, "thal"] = np.nan

    prediction = int(model.predict(sample)[0])

    assert prediction in [0, 1]
