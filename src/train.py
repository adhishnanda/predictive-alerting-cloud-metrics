import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_utils import download_dataset, load_dataset
from features import make_incident_labels, make_supervised_dataset


def evaluate_threshold(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist()
    }


def main():
    os.makedirs("results", exist_ok=True)

    path = download_dataset()
    df = load_dataset(path)

    values = df["value"].values.astype(float)

    # time-based split first
    split_idx = int(0.7 * len(values))
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    # define threshold ONLY from training data to avoid leakage
    incident_threshold = np.quantile(train_values, 0.99)

    train_incidents = make_incident_labels(train_values, incident_threshold)
    test_incidents = make_incident_labels(test_values, incident_threshold)

    W = 20
    H = 5

    X_train, y_train = make_supervised_dataset(train_values, train_incidents, W, H)
    X_test, y_test = make_supervised_dataset(test_values, test_incidents, W, H)

    # Model 1: Logistic Regression
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.5, 0.7]

    results = {
        "dataset": "NAB cpu_utilization_asg_misconfiguration.csv",
        "window_size": W,
        "horizon": H,
        "incident_threshold": float(incident_threshold),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "logistic_regression": {
            "roc_auc": float(roc_auc_score(y_test, lr_probs)),
            "threshold_results": [evaluate_threshold(y_test, lr_probs, t) for t in thresholds],
        },
        "random_forest": {
            "roc_auc": float(roc_auc_score(y_test, rf_probs)),
            "threshold_results": [evaluate_threshold(y_test, rf_probs, t) for t in thresholds],
        },
    }

    import json
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({
        "y_true": y_test,
        "lr_prob": lr_probs,
        "rf_prob": rf_probs,
    }).to_csv("results/predictions.csv", index=False)

    print("Done. Results saved to results/metrics.json and results/predictions.csv")


if __name__ == "__main__":
    main()