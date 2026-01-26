# Heart Attack Risk Predictor

This project is an application for predicting the risk of a heart attack based on patients’ medical and behavioral data.
The solution includes exploratory data analysis, machine learning model training, and a **FastAPI** service for generating predictions on a test dataset.

A detailed data analysis (EDA), model selection, hyperparameter tuning, and methodological justification are presented in the Jupyter Notebook `EDA_and_modeling.ipynb`.

---

## 1. Project Structure

```
medical_project_repo/
│
├── data/
│   ├── heart_train.csv
│   └── heart_test.csv
│
├── models/
│   ├── model.cbm
│   └── meta.json
│
├── notebooks/
│   └── EDA_and_modeling.ipynb
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── model/
│   │   └── predictor.py
│   └── api/
│       └── main.py
│
├── predictions.csv
├── fallback_ids.json
└── README.md
```

---

## 2. Technology Stack

* Python
* Pandas, NumPy
* Scikit-learn
* CatBoost
* FastAPI

---

## 3. Environment and Dependencies

The project was tested in the following environment:

* Python 3.10 (Conda)

### Core Dependencies

* catboost==1.2.8
* pandas==2.3.3
* numpy==2.2.5
* scikit-learn==1.5.x
* scipy==1.15.3
* fastapi==0.128.0
* uvicorn==0.40.0
* pydantic==2.12.5

### Additional Dependencies

* matplotlib==3.10.8
* seaborn==0.13.2
* phik==0.12.5

---

## 4. Exploratory Data Analysis and Modeling (EDA)

Data exploration, feature distribution analysis, missing value handling, model selection, and hyperparameter tuning were performed in the Jupyter Notebook:

* `EDA_and_modeling.ipynb`

The notebook includes:

* analysis of data quality and structure;
* identification and interpretation of missing values;
* model selection;
* optimization of the classification threshold;
* conclusions regarding model performance.

All architectural and methodological decisions used in the application are based on the results of this analysis.

---

## 5. Model Training

Model training is performed on a cleaned training dataset without missing values.
Rows containing technical missing values are excluded from training.

To run training from the project root:

```bash
python -m scripts.train
```

As a result, the following files are saved:

* model: `models/model.cbm`
* metadata: `models/meta.json` (feature list, classification threshold, and output settings)

---

## 6. Local Prediction Generation (Without API)

Predictions are generated **strictly for all rows of the test file**, as required by the technical specification.

### Processing Logic

* rows **without missing values** are processed using the trained model;
* rows **with missing values** use fallback logic: `prediction = 0`;
* the final output file is created **in the original row order of the CSV file**;
* an additional list of `id` values is saved for which fallback logic was applied.

Execution:

```bash
python -m scripts.predict
```

Outputs:

* `predictions.csv` — file with columns `id, prediction` (strictly according to the specification);
* `fallback_ids.json` — list of `id` values for which fallback logic (`prediction = 0`) was applied due to missing data.

---

## 7. Running the FastAPI Service

Start the service from the project root:

```bash
uvicorn src.api.main:app --reload
```

Swagger documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

## 8. API Usage

### Endpoint

`POST /predict`

### Input

JSON containing the path to a CSV file on disk:

```json
{
  "csv_path": "data/heart_test.csv"
}
```

### Processing Logic

* predictions are generated **only for rows without NaN values**;
* rows with missing values receive `prediction = 0` (fallback logic);
* results are returned **in the original order of the input CSV file**;
* an additional list of `id` values is returned for which fallback logic was applied due to missing data.

### Response

```json
{
  "predictions": [
    {"id": 123, "prediction": 0},
    {"id": 124, "prediction": 1}
  ],
  "fallback_ids": [456, 789]
}
```

---

## 9. Compliance with the Technical Specification

The following guarantees are provided:

* `predictions.csv` contains **exactly two columns**: `id`, `prediction`;
* the number of rows matches the original test file;
* there are no missing values in `prediction`;
* `prediction` values belong to the set `{0, 1}`;
* row order strictly matches the original CSV file;
* an additional list of `id` values is generated for which fallback logic was applied.

---

## 10. Methodological Notes

* Missing values in the data are **systematic and technical**, therefore they are neither simulated nor imputed.
* A conservative approach is chosen for the medical task: rows with missing values receive `prediction = 0`.
* A separate list of `id` values is generated for which fallback logic was applied.
* The primary model evaluation metric is **ROC-AUC**; an additional metric is **F1-score** with an optimized classification threshold.
* The **CatBoostClassifier** model is used, as it is robust to nonlinearities and outliers.

---

## 11. Summary

The project implements a complete ML pipeline:

* data exploration and analysis;
* model training and evaluation;
* reproducible prediction generation;
* a service for model usage;
* strict validation of output format in accordance with the technical specification.
