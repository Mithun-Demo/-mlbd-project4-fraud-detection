"""Shared data loading, preprocessing and metrics for Project 4 (fraud / credit-risk classification)."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
SEED = 42
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

GERMAN_COLS = ["status", "duration", "credit_history", "purpose", "amount", "savings",
               "employment", "installment_rate", "personal_status", "other_debtors",
               "residence_since", "property", "age", "other_installments", "housing",
               "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
               "Class"]


def load_raw(name):
    if name == "credit":
        # drop exact duplicate rows (381 of them, all legitimate transactions)
        return pd.read_csv(os.path.join(DATA, "creditcard_sample.csv")).drop_duplicates().reset_index(drop=True)
    if name == "german":
        df = pd.read_csv(os.path.join(DATA, "german.csv"), header=None, names=GERMAN_COLS)
        df["Class"] = (df["Class"] == 2).astype(int)  # 1 = bad credit (positive class), 0 = good
        return df
    raise ValueError(name)


def get_split(name):
    """Return X_train, X_test, y_train, y_test (float32 numpy), identical for every framework."""
    df = load_raw(name)
    y = df["Class"].values.astype(int)
    X = df.drop(columns="Class")
    if name == "german":
        X = pd.get_dummies(X, columns=X.select_dtypes("object").columns.tolist(), dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)  # fit on train only -> no leakage
    return (scaler.transform(X_tr).astype("float32"), scaler.transform(X_te).astype("float32"),
            y_tr, y_te, list(X.columns))


def pos_weight(y):
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def evaluate(y_true, proba, threshold=0.5):
    """Classification metrics plus the RMSE / MAE / R2 the brief asks for (on predicted probabilities)."""
    proba = np.asarray(proba, dtype=float).ravel()
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "rmse": float(np.sqrt(mean_squared_error(y_true, proba))),
        "mae": mean_absolute_error(y_true, proba),
        "r2": r2_score(y_true, proba),
    }


def save_result(framework, dataset, metrics, fit_seconds, loc, notes=""):
    out = {"framework": framework, "dataset": dataset, "fit_seconds": round(fit_seconds, 2),
           "lines_of_code": loc, "notes": notes, **{k: round(float(v), 4) for k, v in metrics.items()}}
    with open(os.path.join(RESULTS, f"{framework}_{dataset}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t
