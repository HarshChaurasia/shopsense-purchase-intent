"""
ML Assignment 2 - Online Shoppers Purchasing Intention (UCI id 468)
Trains 5 classifiers on the same train/test split, evaluates them with
6 metrics and saves the fitted pipelines to model/.

Run:  python train_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 7
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "online_shoppers_intention.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Revenue"
NUMERIC_COLS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
# integer-coded columns that are really categories + true categoricals
CATEGORICAL_COLS = ["Month", "OperatingSystems", "Browser", "Region",
                    "TrafficType", "VisitorType", "Weekend"]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    df["Weekend"] = df["Weekend"].astype(str).str.upper()
    return df


def build_preprocessor(log_numeric=True):
    num_steps = []
    if log_numeric:
        # page counts / durations are heavily right-skewed -> log1p first
        from sklearn.preprocessing import FunctionTransformer
        num_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


def get_models():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        ]),
        "Decision Tree": GridSearchCV(
            Pipeline([("prep", build_preprocessor(log_numeric=False)),
                      ("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [4, 6, 8, 10], "clf__min_samples_leaf": [5, 20, 50]},
            scoring="f1", cv=cv, n_jobs=-1),
        "KNN": GridSearchCV(
            Pipeline([("prep", build_preprocessor()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
            scoring="f1", cv=cv, n_jobs=-1),
        "Naive Bayes": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", GaussianNB(var_smoothing=1e-2)),
        ]),
        "Support Vector Machine": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)),
        ]),
    }


FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    df = load_data()
    print(f"Data after de-duplication: {df.shape}, positive rate = {df[TARGET].mean():.3f}")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # held-out test file used by the Streamlit app (features + target)
    test_df = X_te.copy()
    test_df[TARGET] = y_te.map({1: "TRUE", 0: "FALSE"})
    test_df.to_csv(BASE / "test_data.csv", index=False)

    results = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        best = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(model, "best_params_"):
            print(f"{name} best params: {model.best_params_}")
        joblib.dump(best, MODEL_DIR / FILE_NAMES[name], compress=3)
        results[name] = evaluate(best, X_te, y_te)
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics_summary.csv")
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump({"features": FEATURES, "target": TARGET, "files": FILE_NAMES,
                   "seed": SEED, "train_rows": len(X_tr), "test_rows": len(X_te)}, f, indent=2)
    print("\n", table.to_markdown())


if __name__ == "__main__":
    main()
