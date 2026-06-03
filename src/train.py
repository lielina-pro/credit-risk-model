"""
src/train.py — Task 5: Model Training, Tracking & Evaluation
=============================================================
Trains four models (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting) with hyperparameter tuning, logs everything to MLflow,
and registers the best model in the MLflow Model Registry.

Usage:
    python src/train.py
    python src/train.py --data data/processed/features_with_target.csv
    python src/train.py --quick   # fast run (small param grids) for CI/CD

MLflow UI:
    mlflow ui
    # then open http://localhost:5000
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-5s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "features_with_target.csv"
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "credit_risk_proxy_model"
REGISTERED_MODEL_NAME = "CreditRiskProxyModel"

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL = "is_high_risk"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER_SEARCH = 10  # RandomizedSearchCV iterations per model


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────


def load_and_split(data_path: Path, test_size: float = TEST_SIZE):
    """Load features_with_target.csv and return train/test splits."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            "Run src/proxy_target.py first to generate features_with_target.csv."
        )

    # Drop any non-numeric or ID-like columns that leaked through
    drop_cols = [c for c in df.columns if "CustomerId" in c or "TransactionId" in c]
    df = df.drop(columns=drop_cols, errors="ignore")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Keep only numeric columns (safety guard)
    X = X.select_dtypes(include=[np.number])

    # Fill any remaining NaN with column median
    X = X.fillna(X.median())

    logger.info(
        "Dataset: %d rows × %d features | High-risk rate: %.1f%%",
        len(df),
        X.shape[1],
        y.mean() * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITIONS & PARAM GRIDS
# ─────────────────────────────────────────────────────────────────────────────


def get_model_configs(quick: bool = False):
    """
    Returns a dict of {name: (estimator_pipeline, param_grid)}.
    quick=True uses tiny grids for fast CI/local testing.
    """
    configs = {}

    # ── Logistic Regression ──
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0] if quick else [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
    }
    configs["LogisticRegression"] = (lr_pipe, lr_grid)

    # ── Decision Tree ──
    dt_pipe = Pipeline(
        [
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )
    dt_grid = {
        "clf__max_depth": [3, 5, 10] if quick else [3, 5, 8, 10, 15, None],
        "clf__min_samples_split": [2, 10] if quick else [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 5] if quick else [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    }
    configs["DecisionTree"] = (dt_pipe, dt_grid)

    # ── Random Forest ──
    rf_pipe = Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [50, 100] if quick else [100, 200, 300],
        "clf__max_depth": [5, 10] if quick else [5, 10, 15, None],
        "clf__min_samples_split": [2, 10] if quick else [2, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    }
    configs["RandomForest"] = (rf_pipe, rf_grid)

    # ── Gradient Boosting ──
    gb_pipe = Pipeline(
        [
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    gb_grid = {
        "clf__n_estimators": [50, 100] if quick else [100, 200, 300],
        "clf__learning_rate": [0.1, 0.05] if quick else [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5] if quick else [3, 4, 5, 6],
        "clf__subsample": [0.8, 1.0],
        "clf__min_samples_split": [2, 10] if quick else [2, 5, 10],
    }
    configs["GradientBoosting"] = (gb_pipe, gb_grid)

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_model(model, X_test, y_test):
    """Return dict of all required evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    )
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN ONE MODEL WITH MLFLOW TRACKING
# ─────────────────────────────────────────────────────────────────────────────


def train_and_log(
    name: str,
    pipeline,
    param_grid: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names: list,
    n_iter: int = N_ITER_SEARCH,
    cv: int = CV_FOLDS,
):
    """Run RandomizedSearchCV, log to MLflow, return (run_id, best_estimator, metrics)."""
    logger.info("── Training %s ──────────────────────────────", name)

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_strategy,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Tags
        mlflow.set_tags(
            {
                "model_type": name,
                "task": "credit_risk_proxy",
                "dataset": "xente_ecommerce",
                "target": TARGET_COL,
            }
        )

        # Fit
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Log best hyperparameters
        best_params = search.best_params_
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("n_iter_search", n_iter)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log CV score
        mlflow.log_metric("cv_roc_auc_best", round(search.best_score_, 4))

        # Evaluate on hold-out test set
        metrics = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)

        logger.info(
            "  Best params : %s",
            {k: v for k, v in best_params.items()},
        )
        logger.info(
            "  CV ROC-AUC  : %.4f | Test ROC-AUC: %.4f | F1: %.4f",
            search.best_score_,
            metrics["roc_auc"],
            metrics["f1_score"],
        )

        # Log classification report as artifact
        report = classification_report(
            y_test,
            best_model.predict(X_test),
            target_names=["low_risk", "high_risk"],
        )
        report_path = f"/tmp/{name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model with input signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        logger.info("  Run ID: %s", run_id)

    return run_id, best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGISTER BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────


def register_best_model(results: list):
    """Pick the model with the highest test ROC-AUC and register it."""
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    model_uri = f"runs:/{best['run_id']}/model"

    logger.info(
        "Registering best model: %s (ROC-AUC=%.4f)",
        best["name"],
        best["metrics"]["roc_auc"],
    )

    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    # Add description
    client = mlflow.tracking.MlflowClient()
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            f"Best model: {best['name']} | "
            f"ROC-AUC: {best['metrics']['roc_auc']:.4f} | "
            f"F1: {best['metrics']['f1_score']:.4f} | "
            f"Target: {TARGET_COL}"
        ),
    )

    # Transition to Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered.version,
        stage="Staging",
    )

    logger.info(
        "Model registered as '%s' version %s — stage: Staging",
        REGISTERED_MODEL_NAME,
        registered.version,
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main(data_path: Path = DATA_PATH, quick: bool = False):
    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment          : %s", EXPERIMENT_NAME)

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_and_split(data_path)

    # Train all models
    model_configs = get_model_configs(quick=quick)
    n_iter = 3 if quick else N_ITER_SEARCH

    results = []
    for name, (pipeline, param_grid) in model_configs.items():
        run_id, best_model, metrics = train_and_log(
            name=name,
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            n_iter=n_iter,
        )
        results.append(
            {
                "name": name,
                "run_id": run_id,
                "model": best_model,
                "metrics": metrics,
            }
        )

    # Summary table
    logger.info("\n%s", "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info(
        "%-22s %8s %9s %8s %8s %8s",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    )
    logger.info("-" * 60)
    for r in sorted(results, key=lambda x: x["metrics"]["roc_auc"], reverse=True):
        m = r["metrics"]
        logger.info(
            "%-22s %8.4f %9.4f %8.4f %8.4f %8.4f",
            r["name"],
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1_score"],
            m["roc_auc"],
        )
    logger.info("=" * 60)

    # Register best model
    best = register_best_model(results)

    logger.info("\n✅ TASK 5 COMPLETE")
    logger.info("   Best model : %s", best["name"])
    logger.info("   ROC-AUC    : %.4f", best["metrics"]["roc_auc"])
    logger.info("   F1 Score   : %.4f", best["metrics"]["f1_score"])
    logger.info(
        "   Run MLflow UI: mlflow ui --backend-store-uri %s", MLFLOW_TRACKING_URI
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk proxy models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to features_with_target.csv",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small param grids for fast testing (CI/CD mode)",
    )
    args = parser.parse_args()
    main(data_path=args.data, quick=args.quick)

