# scripts/predict.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.preprocessing.preprocessor import DataPreprocessor
from src.model.predictor import Predictor

TEST_PATH = Path("data/heart_test.csv")
OUT_PRED = Path("predictions.csv")
OUT_FALLBACK = Path("fallback_ids.json")


def main():
    df = pd.read_csv(TEST_PATH)

    pre = DataPreprocessor()
    df_clean, df_fallback, fallback_ids, clean_mask = pre.split_clean_and_fallback(df)

    predictor = Predictor()

    # 1) предсказания только для clean (в том же порядке, что df_clean)
    preds_clean = predictor.predict(df_clean)  # DF: id, prediction

    # 2) prediction для всех строк test.csv (fallback=0)
    pred_all = np.zeros(len(df), dtype=int)  

    # 3) вставляем предсказания модели на clean-позиции
    pred_all[clean_mask] = preds_clean["prediction"].astype(int).values

    # 4) собираем итог строго в исходном порядке test.csv
    preds_all = pd.DataFrame({
        "id": df["id"].astype(int).values,
        "prediction": pred_all
    })

    # 5) сохраняем
    preds_all.to_csv(OUT_PRED, index=False)

    OUT_FALLBACK.write_text(
        json.dumps({"fallback_ids": fallback_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # самопроверки
    assert list(preds_all.columns) == ["id", "prediction"]
    assert len(preds_all) == len(df)
    assert preds_all["prediction"].isna().sum() == 0

    print(f"Saved: {OUT_PRED} rows={len(preds_all)}")
    print(f"Fallback due to NaN: {len(fallback_ids)} (saved to {OUT_FALLBACK})")


if __name__ == "__main__":
    main()
