# scripts/train.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.preprocessing.preprocessor import DataPreprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.25

TRAIN_PATH = Path("data/heart_train.csv")
MODEL_PATH = Path("models/model.cbm")
META_PATH = Path("models/meta.json")


def main():
    df = pd.read_csv(TRAIN_PATH)

    pre = DataPreprocessor()
    df_clean, _, dropped_ids, _ = pre.split_clean_and_fallback(df)
    df = df_clean

    # target variable
    target = "Heart Attack Risk (Binary)"

    # categorical features
    cat_features = ["Gender", "Diet"]

    # X / y
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    # Save feature order (excluding id)
    feature_columns = [c for c in X.columns if c != "id"]

    X_train, X_val, y_train, y_val = train_test_split(
        X[feature_columns],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=11,
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=200,
        auto_class_weights="Balanced",
        early_stopping_rounds=200,
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"Validation ROC-AUC: {auc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    meta = {
        "cat_features": cat_features,
        "feature_columns": feature_columns,
        "threshold": 0.36224489795918363,  # average threshold
        "output_mode": "class",
        "target": target,
        "model_type": "CatBoostClassifier",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta:  {META_PATH}")


if __name__ == "__main__":
    main()
