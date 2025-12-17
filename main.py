import logging
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
MODEL_PATH = "best_model.pkl"

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_dataset() -> pd.DataFrame:
    logging.info("Generating synthetic classification dataset")

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



def split_data(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def get_pipelines() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ]),

        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=200
            ))
        ])
    }



def evaluate_models(pipelines, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = {}
    for name, pipeline in pipelines.items():
        logging.info(f"Cross-validating {name}")
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = pipeline.predict(X_train.iloc[val_idx])
            cv_scores.append(accuracy_score(y_train.iloc[val_idx], preds))

        scores[name] = np.mean(cv_scores)
        logging.info(f"{name} CV Accuracy: {scores[name]:.4f}")

    return scores


def tune_random_forest(X_train, y_train):
    logging.info("Starting hyperparameter tuning for Random Forest")

    pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 15, 30],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        cv=CV_SPLITS,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    logging.info(f"Best Parameters: {grid.best_params_}")
    logging.info(f"Best CV Accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logging.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

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



def save_model(model):
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")


def main():
    logging.info("Pipeline started")

    df = create_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    pipelines = get_pipelines()
    evaluate_models(pipelines, X_train, y_train)

    best_model = tune_random_forest(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
