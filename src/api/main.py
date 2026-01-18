import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from src.preprocessing.preprocessor import DataPreprocessor
from src.model.predictor import Predictor

app = FastAPI(
    title="Heart Attack Risk Predictor",
    version="0.1.0",
    openapi_tags=[
        {
            "name": "Загрузка данных для предсказания",
            "description": "Приём CSV-файла",
        }
    ],
)

PRE = DataPreprocessor()
PREDICTOR = Predictor() 


class PredictRequest(BaseModel):
    csv_path: str = Field(example="data/heart_test.csv")


@app.post(
    "/predict",
    tags=["Загрузка данных для предсказания"],
    summary="Предсказание риска сердечного приступа",
    description=(
        "Принимает путь к CSV-файлу. "
        "Предсказания выполняются только для строк без пропусков. "
        "Для строк с NaN используется fallback-логика (prediction = 0). "
        "Возвращает результат предсказаний и список ID, для которых применён fallback."
    ),
)
def predict(req: PredictRequest):
    csv_path = req.csv_path

    path = Path(csv_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read csv: {e}")

    try:
        df_clean, df_fallback, fallback_ids, clean_mask = PRE.split_clean_and_fallback(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds_clean = PREDICTOR.predict(df_clean)  # DF: id, prediction (0/1)

    pred_all = np.zeros(len(df), dtype=int)
    pred_all[clean_mask] = preds_clean["prediction"].astype(int).values

    preds_all = pd.DataFrame({
        "id": df["id"].astype(int).values,
        "prediction": pred_all
    })

    return {
        "predictions": preds_all.to_dict(orient="records"),
        "fallback_ids": fallback_ids
    }
