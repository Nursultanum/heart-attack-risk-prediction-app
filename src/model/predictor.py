import json
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier


class Predictor:
    def __init__(self, model_path="models/model.cbm", meta_path="models/meta.json"):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.feature_columns = meta["feature_columns"]  # без id и без target
        self.cat_features = meta["cat_features"]
        self.output_mode = meta.get("output_mode", "proba")
        self.threshold = float(meta.get("threshold", 0.5))

    def predict(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        if "id" not in df_clean.columns:
            raise ValueError("Input must contain 'id' column")

        X = df_clean.drop(columns=["id"], errors="ignore")

        # Проверяем, что все нужные колонки на месте
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = X[self.feature_columns]

        proba = self.model.predict_proba(X)[:, 1]

        if self.output_mode == "class":
            pred = (proba >= self.threshold).astype(int)
        else:
            pred = proba

        return pd.DataFrame({"id": df_clean["id"].astype(int), "prediction": pred})
