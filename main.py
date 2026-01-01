import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# =========================
# Configuration
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
MODEL_PATH = "best_model.pkl"

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================
# Data Generation
# =========================
def create_dataset() -> pd.DataFrame:
    """
    Generates a synthetic binary classification dataset.
    """
    logging.info("Generating synthetic dataset")

    X, y = make_classification(
        n_samples=3000,
        n_features=25,
        n_informative=18,
        n_redundant=4,
        n_classes=2,
        class_sep=1.5,
        random_state=RANDOM_STATE
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


# =========================
# Train-Test Split
# =========================
def split_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset into stratified train and test sets.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# =========================
# Model Pipelines
# =========================
def get_pipelines() -> Dict[str, Pipeline]:
    """
    Returns ML pipelines for different models.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ]),

        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            ))
        ])
    }


# =========================
# Model Evaluation (CV)
# =========================
def evaluate_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, float]:
    """
    Performs cross-validation on all models.
    """
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = {}
    for name, pipeline in pipelines.items():
        logging.info(f"Evaluating {name} with cross-validation")

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1
        )

        scores[name] = cv_scores.mean()
        logging.info(f"{name} CV Accuracy: {scores[name]:.4f}")

    return scores


# =========================
# Hyperparameter Tuning
# =========================
def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Performs GridSearchCV for Random Forest.
    """
    logging.info("Starting Random Forest hyperparameter tuning")

    pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 15, 30],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=CV_SPLITS,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    logging.info(f"Best Parameters: {grid.best_params_}")
    logging.info(f"Best CV Accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_


# =========================
# Final Evaluation
# =========================
def evaluate_final_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Evaluates the trained model on test data.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        logging.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    logging.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# =========================
# Save Model
# =========================
def save_model(model: Pipeline) -> None:
    """
    Saves trained model to disk.
    """
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved at: {MODEL_PATH}")


# =========================
# Main Pipeline
# =========================
def main() -> None:
    logging.info("ML pipeline started")

    df = create_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    pipelines = get_pipelines()
    evaluate_models(pipelines, X_train, y_train)

    best_model = tune_random_forest(X_train, y_train)
    evaluate_final_model(best_model, X_test, y_test)

    save_model(best_model)

    logging.info("ML pipeline completed successfully")


if __name__ == "__main__":
    main()
