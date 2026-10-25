"""
Train and evaluate heart disease classification models.

This script performs:
1. Data loading and validation
2. Stratified train-test splitting
3. Preprocessing using scikit-learn pipelines
4. Logistic Regression tuning
5. Random Forest tuning
6. Model evaluation
7. MLflow experiment tracking
8. Plot and report generation
9. Best-model selection
10. Model and metadata serialization

Run from the repository root:

    python src/models/train.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5
TARGET_COLUMN = "target"
EXPERIMENT_NAME = "heart-disease-classification"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_cleaned.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.csv"

MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
)

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

EXPECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def create_directories() -> None:
    """Create all output directories required by the training script."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def convert_to_python_value(value: Any) -> Any:
    """
    Convert NumPy values and other objects into JSON-compatible values.
    """

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def make_json_serializable(data: Any) -> Any:
    """Recursively convert objects into JSON-compatible values."""

    if isinstance(data, dict):
        return {
            str(key): make_json_serializable(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            make_json_serializable(value)
            for value in data
        ]

    if isinstance(data, tuple):
        return [
            make_json_serializable(value)
            for value in data
        ]

    return convert_to_python_value(data)


def load_and_validate_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed dataset and perform validation checks.

    Parameters
    ----------
    data_path:
        Path to the processed CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictor dataframe and target series.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data file was not found: {data_path}\n"
            "Run the EDA and preprocessing notebook first."
        )

    dataframe = pd.read_csv(data_path)

    required_columns = EXPECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The processed dataset is missing required columns: "
            f"{missing_columns}"
        )

    unexpected_columns = sorted(
        set(dataframe.columns) - set(required_columns)
    )

    if unexpected_columns:
        print(
            "Warning: Unexpected columns will not be used:",
            unexpected_columns,
        )

    if dataframe.empty:
        raise ValueError("The processed dataset is empty.")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    target_values = set(
        dataframe[TARGET_COLUMN].unique().tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "The target must contain only binary values 0 and 1. "
            f"Found: {sorted(target_values)}"
        )

    predictors = dataframe[EXPECTED_FEATURES].copy()
    target = dataframe[TARGET_COLUMN].astype(int).copy()

    print("\nData validation completed successfully.")
    print(f"Dataset shape: {dataframe.shape}")
    print(f"Predictor shape: {predictors.shape}")
    print("Target distribution:")
    print(target.value_counts().sort_index())

    print("\nMissing predictor values:")
    missing_values = predictors.isna().sum()
    print(missing_values[missing_values > 0])

    return predictors, target


# -------------------------------------------------------------------
# Preprocessing and model construction
# -------------------------------------------------------------------

def create_preprocessor() -> ColumnTransformer:
    """
    Create the reusable preprocessing transformer.

    Numerical preprocessing:
    - Median imputation
    - Standard scaling

    Categorical preprocessing:
    - Most-frequent imputation
    - One-hot encoding
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_configurations() -> dict[str, dict[str, Any]]:
    """
    Create the candidate model pipelines and small tuning grids.
    """

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )

    logistic_parameter_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__class_weight": [None, "balanced"],
    }

    return {
        "logistic_regression": {
            "pipeline": logistic_pipeline,
            "parameter_grid": logistic_parameter_grid,
        },
        "random_forest": {
            "pipeline": random_forest_pipeline,
            "parameter_grid": random_forest_parameter_grid,
        },
    }


# -------------------------------------------------------------------
# Evaluation and plotting functions
# -------------------------------------------------------------------

def calculate_metrics(
    y_true: pd.Series,
    y_predicted: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_predicted)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_predicted: np.ndarray,
    model_name: str,
) -> Path:
    """Generate and save a confusion-matrix plot."""

    matrix = confusion_matrix(y_true, y_predicted)

    output_path = (
        FIGURES_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "No heart disease",
            "Heart disease present",
        ],
        yticklabels=[
            "No heart disease",
            "Heart disease present",
        ],
    )

    plt.title(
        f"{DISPLAY_NAMES[model_name]} Confusion Matrix"
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_roc_curve(
    y_true: pd.Series,
    y_probability: np.ndarray,
    model_name: str,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Generate and save an individual ROC curve."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        y_probability,
    )

    area_under_curve = roc_auc_score(
        y_true,
        y_probability,
    )

    output_path = (
        FIGURES_DIR
        / f"{model_name}_roc_curve.png"
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"ROC-AUC = {area_under_curve:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    plt.title(
        f"{DISPLAY_NAMES[model_name]} ROC Curve"
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return (
        output_path,
        false_positive_rate,
        true_positive_rate,
    )


def save_classification_report(
    y_true: pd.Series,
    y_predicted: np.ndarray,
    model_name: str,
) -> Path:
    """Generate and save the text classification report."""

    report = classification_report(
        y_true,
        y_predicted,
        target_names=[
            "No heart disease",
            "Heart disease present",
        ],
        zero_division=0,
    )

    output_path = (
        REPORTS_DIR
        / f"{model_name}_classification_report.txt"
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    return output_path


def save_combined_roc_curve(
    model_results: dict[str, dict[str, Any]],
) -> Path:
    """Create a combined ROC curve for all candidate models."""

    output_path = FIGURES_DIR / "model_roc_curves.png"

    plt.figure(figsize=(8, 6))

    colors = {
        "logistic_regression": "#4C78A8",
        "random_forest": "#E45756",
    }

    for model_name, result in model_results.items():
        plt.plot(
            result["false_positive_rate"],
            result["true_positive_rate"],
            linewidth=2,
            color=colors.get(model_name),
            label=(
                f"{DISPLAY_NAMES[model_name]} "
                f"(ROC-AUC = {result['metrics']['roc_auc']:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    plt.title("Comparison of Model ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------

def train_and_evaluate_model(
    model_name: str,
    model_pipeline: Pipeline,
    parameter_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cross_validation: StratifiedKFold,
) -> dict[str, Any]:
    """
    Tune, train, evaluate, and log one candidate model.
    """

    display_name = DISPLAY_NAMES[model_name]

    print("\n" + "=" * 70)
    print(f"Training model: {display_name}")
    print("=" * 70)

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    with mlflow.start_run(run_name=model_name) as active_run:
        mlflow.log_param("model_name", display_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param(
            "cross_validation_folds",
            CROSS_VALIDATION_FOLDS,
        )
        mlflow.log_param(
            "numerical_imputation",
            "median",
        )
        mlflow.log_param(
            "categorical_imputation",
            "most_frequent",
        )
        mlflow.log_param(
            "categorical_encoding",
            "one_hot_encoding",
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_parameters = grid_search.best_params_
        best_cv_roc_auc = float(grid_search.best_score_)

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(
            y_true=y_test,
            y_predicted=predictions,
            y_probability=probabilities,
        )

        metrics["best_cv_roc_auc"] = best_cv_roc_auc

        cross_validation_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=-1,
        )

        metrics["cv_roc_auc_mean"] = float(
            cross_validation_scores.mean()
        )
        metrics["cv_roc_auc_std"] = float(
            cross_validation_scores.std()
        )

        confusion_matrix_path = save_confusion_matrix(
            y_true=y_test,
            y_predicted=predictions,
            model_name=model_name,
        )

        (
            roc_curve_path,
            false_positive_rate,
            true_positive_rate,
        ) = save_roc_curve(
            y_true=y_test,
            y_probability=probabilities,
            model_name=model_name,
        )

        classification_report_path = (
            save_classification_report(
                y_true=y_test,
                y_predicted=predictions,
                model_name=model_name,
            )
        )

        mlflow.log_params(
            {
                key: str(value)
                for key, value in best_parameters.items()
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(roc_curve_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(classification_report_path),
            artifact_path="evaluation",
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format="cloudpickle",
        )

        run_id = active_run.info.run_id

    print("\nBest parameters:")
    for parameter_name, parameter_value in best_parameters.items():
        print(f"  {parameter_name}: {parameter_value}")

    print(f"\nBest tuning ROC-AUC: {best_cv_roc_auc:.4f}")
    print(
        "Cross-validation ROC-AUC: "
        f"{metrics['cv_roc_auc_mean']:.4f} "
        f"+/- {metrics['cv_roc_auc_std']:.4f}"
    )

    print("\nTest-set metrics:")
    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        print(
            f"  {metric_name}: "
            f"{metrics[metric_name]:.4f}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No heart disease",
                "Heart disease present",
            ],
            zero_division=0,
        )
    )

    return {
        "model_name": model_name,
        "display_name": display_name,
        "model": best_model,
        "best_parameters": best_parameters,
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "mlflow_run_id": run_id,
        "confusion_matrix_path": confusion_matrix_path,
        "roc_curve_path": roc_curve_path,
        "classification_report_path": (
            classification_report_path
        ),
    }


# -------------------------------------------------------------------
# Output functions
# -------------------------------------------------------------------

def create_model_comparison(
    model_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Create and save the model-comparison table."""

    comparison_records = []

    for model_name, result in model_results.items():
        metrics = result["metrics"]

        comparison_records.append(
            {
                "model_key": model_name,
                "model_name": result["display_name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "cv_roc_auc_mean": (
                    metrics["cv_roc_auc_mean"]
                ),
                "cv_roc_auc_std": (
                    metrics["cv_roc_auc_std"]
                ),
                "mlflow_run_id": (
                    result["mlflow_run_id"]
                ),
            }
        )

    comparison_dataframe = pd.DataFrame(
        comparison_records
    )

    comparison_dataframe = comparison_dataframe.sort_values(
        by=[
            "roc_auc",
            "recall",
            "f1",
        ],
        ascending=False,
    ).reset_index(drop=True)

    comparison_dataframe.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    return comparison_dataframe


def select_best_model(
    model_results: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    Select the best model.

    Primary criterion:
    - Test ROC-AUC

    Tie-breakers:
    - Recall
    - F1-score
    """

    best_model_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["metrics"]["roc_auc"],
            model_results[name]["metrics"]["recall"],
            model_results[name]["metrics"]["f1"],
        ),
    )

    return (
        best_model_name,
        model_results[best_model_name],
    )


def save_final_model_and_metadata(
    best_model_name: str,
    best_result: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save the selected pipeline and its metadata."""

    joblib.dump(
        best_result["model"],
        MODEL_PATH,
    )

    metadata = {
        "project_name": (
            "Heart Disease Classification MLOps Pipeline"
        ),
        "model_name": (
            best_result["display_name"]
        ),
        "model_key": best_model_name,
        "model_file": MODEL_PATH.name,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cross_validation_folds": (
            CROSS_VALIDATION_FOLDS
        ),
        "target_column": TARGET_COLUMN,
        "target_definition": {
            "0": "No heart disease",
            "1": "Heart disease present",
        },
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": EXPECTED_FEATURES,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "training_target_distribution": (
            y_train.value_counts()
            .sort_index()
            .to_dict()
        ),
        "test_target_distribution": (
            y_test.value_counts()
            .sort_index()
            .to_dict()
        ),
        "selection_criteria": [
            "roc_auc",
            "recall",
            "f1",
        ],
        "best_parameters": (
            best_result["best_parameters"]
        ),
        "test_metrics": (
            best_result["metrics"]
        ),
        "mlflow_run_id": (
            best_result["mlflow_run_id"]
        ),
        "educational_use_notice": (
            "This model is an educational MLOps "
            "demonstration and is not intended for "
            "clinical diagnosis."
        ),
    }

    serializable_metadata = make_json_serializable(
        metadata
    )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            serializable_metadata,
            metadata_file,
            indent=4,
        )


def save_test_data(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save the held-out test records for reproducibility."""

    test_dataframe = X_test.copy()
    test_dataframe[TARGET_COLUMN] = y_test.to_numpy()

    test_dataframe.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def verify_saved_model(
    model_path: Path,
    X_test: pd.DataFrame,
) -> None:
    """Reload the saved model and test one prediction."""

    loaded_model = joblib.load(model_path)

    sample_input = X_test.iloc[[0]].copy()

    sample_prediction = int(
        loaded_model.predict(sample_input)[0]
    )

    sample_probability = float(
        loaded_model.predict_proba(sample_input)[0, 1]
    )

    print("\nSaved model verification:")
    print(f"Sample prediction: {sample_prediction}")
    print(
        "Probability of heart disease: "
        f"{sample_probability:.4f}"
    )

    if sample_prediction not in {0, 1}:
        raise ValueError(
            "Saved model produced an invalid prediction."
        )

    if not 0.0 <= sample_probability <= 1.0:
        raise ValueError(
            "Saved model produced an invalid probability."
        )

    print("Saved model verification passed.")


# -------------------------------------------------------------------
# Main training workflow
# -------------------------------------------------------------------

def main() -> None:
    """Run the full training and evaluation workflow."""

    print("=" * 70)
    print("Heart Disease Classification Training Pipeline")
    print("=" * 70)

    create_directories()

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input dataset: {DATA_PATH}")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    predictors, target = load_and_validate_data(
        DATA_PATH
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        predictors,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print("\nTrain-test split completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTest target distribution:")
    print(y_test.value_counts().sort_index())

    save_test_data(
        X_test=X_test,
        y_test=y_test,
    )

    cross_validation = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_configurations = (
        create_model_configurations()
    )

    model_results: dict[str, dict[str, Any]] = {}

    for model_name, configuration in (
        model_configurations.items()
    ):
        model_results[model_name] = (
            train_and_evaluate_model(
                model_name=model_name,
                model_pipeline=configuration["pipeline"],
                parameter_grid=configuration[
                    "parameter_grid"
                ],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cross_validation=cross_validation,
            )
        )

    combined_roc_path = save_combined_roc_curve(
        model_results
    )

    comparison_dataframe = create_model_comparison(
        model_results
    )

    (
        best_model_name,
        best_result,
    ) = select_best_model(model_results)

    save_final_model_and_metadata(
        best_model_name=best_model_name,
        best_result=best_result,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    verify_saved_model(
        model_path=MODEL_PATH,
        X_test=X_test,
    )

    print("\n" + "=" * 70)
    print("Model comparison")
    print("=" * 70)

    print(
        comparison_dataframe.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("Training completed successfully")
    print("=" * 70)

    print(
        f"Selected model: "
        f"{best_result['display_name']}"
    )
    print(
        f"Selected model ROC-AUC: "
        f"{best_result['metrics']['roc_auc']:.4f}"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(
        f"Model comparison saved to: "
        f"{COMPARISON_PATH}"
    )
    print(
        f"Combined ROC curve saved to: "
        f"{combined_roc_path}"
    )
    print(
        "\nTo view the MLflow runs, execute:\n"
        "mlflow ui --host 0.0.0.0 --port 5000"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nTraining pipeline failed: {error}",
            file=sys.stderr,
        )
        raise
