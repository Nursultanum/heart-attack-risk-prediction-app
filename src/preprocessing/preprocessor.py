# src/preprocessing/preprocessor.py
import pandas as pd


class DataPreprocessor:
    def split_clean_and_fallback(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[bool]]:
        df = df.copy()

        if "id" not in df.columns:
            raise ValueError("Column 'id' is required")

        if df["id"].isna().any():
            raise ValueError("Column 'id' contains missing values")

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        clean_mask = (~df.isna().any(axis=1)).values 

        df_clean = df.loc[clean_mask].copy()
        df_fallback = df.loc[~clean_mask].copy()

        fallback_ids = df_fallback["id"].astype(int).tolist()

        return df_clean, df_fallback, fallback_ids, clean_mask
