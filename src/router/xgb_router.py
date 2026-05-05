"""Runtime loader for the XGBoost router head."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .utils_xgb import RouterFeatureSchema


class RouterHeadXGB:
    """Predict mixture weights for router experts using trained XGBoost heads."""

    EXPERTS: Sequence[str] = ("alpha", "beta", "gamma", "delta")

    def __init__(
        self,
        models: Dict[str, XGBRegressor],
        schema: RouterFeatureSchema,
        feature_columns: Sequence[str],
    ):
        self.models = models
        self.schema = schema
        self.feature_columns = list(feature_columns)

    @classmethod
    def load(cls, models_dir: str | Path) -> "RouterHeadXGB":
        models_dir = Path(models_dir)
        schema_path = models_dir / "scaler_and_schema.pkl"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        with open(schema_path, "rb") as f:
            payload = pickle.load(f)
        schema = payload["schema"]
        feature_columns = payload["feature_columns"]
        models: Dict[str, XGBRegressor] = {}
        for expert in cls.EXPERTS:
            model_path = models_dir / f"router_xgb_{expert}.json"
            booster = XGBRegressor()
            booster.load_model(model_path)
            models[expert] = booster
        return cls(models=models, schema=schema, feature_columns=feature_columns)

    @staticmethod
    def softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        max_val = np.max(arr, axis=axis, keepdims=True)
        exp = np.exp(arr - max_val)
        denom = np.sum(exp, axis=axis, keepdims=True) + 1e-9
        return exp / denom

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("RouterHeadXGB expects a pandas DataFrame.")
        frame = df.copy()
        has_design_cols = all(col in frame.columns for col in self.feature_columns)
        if not has_design_cols:
            frame = self.schema.transform(frame)
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = 0.0
        ordered = ["prompt_id"] + self.feature_columns if "prompt_id" in frame.columns else self.feature_columns
        return frame[ordered]

    def predict_many(self, prompts_df: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_features(prompts_df)
        X = frame[self.feature_columns].to_numpy(dtype=np.float32)
        logits = np.column_stack([self.models[e].predict(X) for e in self.EXPERTS])
        weights = self.softmax(logits, axis=1)
        cols = list(self.EXPERTS)
        out = pd.DataFrame(weights, columns=cols)
        if "prompt_id" in frame.columns:
            out.insert(0, "prompt_id", frame["prompt_id"].values)
        return out

    def predict_weights(self, prompt_df_or_dict: pd.DataFrame | Dict[str, Any]) -> Dict[str, float]:
        if isinstance(prompt_df_or_dict, dict):
            prompts_df = pd.DataFrame([prompt_df_or_dict])
        elif isinstance(prompt_df_or_dict, pd.DataFrame):
            prompts_df = prompt_df_or_dict
        else:
            raise TypeError("Input must be a dict or DataFrame.")
        preds = self.predict_many(prompts_df)
        first = preds.iloc[0]
        return {expert: float(first[expert]) for expert in self.EXPERTS}


__all__ = ["RouterHeadXGB"]
