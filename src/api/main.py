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
            "name": "Data upload for prediction",
            "description": "CSV file ingestion",
        }
    ],
)

PRE = DataPreprocessor()
PREDICTOR = Predictor() 


class PredictRequest(BaseModel):
    csv_path: str = Field(example="data/heart_test.csv")


@app.post(
    "/predict",
    tags=["Data upload for prediction"],
    summary="Heart attack risk prediction",
    description=(
        "Accepts a path to a CSV file. "
        "Predictions are performed only for rows without missing values. "
        "For rows containing NaN values, fallback logic is applied (prediction = 0). "
        "Returns prediction results and a list of IDs for which fallback was applied."
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
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

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
