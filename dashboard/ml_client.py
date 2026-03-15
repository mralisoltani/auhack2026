"""ML API client for training and inference."""
from typing import Any

import pandas as pd
import requests

from src.config import ML_API_URL
TIMEOUT = 300  # 5 min for training


def _url(path: str) -> str:
    return f"{ML_API_URL.rstrip('/')}{path}"


def is_available() -> bool:
    """Check if ML API is reachable."""
    try:
        r = requests.get(_url("/health"), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def train_model(
    zone: str,
    model_type: str,
    feat_cols: list[str],
    n_seeds: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str | None, dict | None, list[str] | None]:
    """
    Call POST /train. Returns (model_id, metrics, feat_cols) or (None, None, None) on error.
    """
    try:
        payload = {
            "zone": zone,
            "model_type": model_type,
            "feat_cols": feat_cols,
            "n_seeds": n_seeds,
            "X_train": X_train.to_dict(orient="records"),
            "y_train": y_train.tolist(),
            "X_val": X_val.to_dict(orient="records"),
            "y_val": y_val.tolist(),
            "X_test": X_test.to_dict(orient="records"),
            "y_test": y_test.tolist(),
        }
        r = requests.post(_url("/train"), json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["model_id"], data["metrics"], data["feat_cols"]
    except Exception:
        return None, None, None


def predict(model_id: str, X: pd.DataFrame, feat_cols: list[str]) -> list[float] | None:
    """Call POST /predict. Returns predictions list or None on error."""
    try:
        X_sub = X[feat_cols] if feat_cols else X
        payload = {"model_id": model_id, "features": X_sub.to_dict(orient="records")}
        r = requests.post(_url("/predict"), json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["predictions"]
    except Exception:
        return None


def list_models() -> list[dict[str, Any]]:
    """Call GET /models. Returns list of model info dicts."""
    try:
        r = requests.get(_url("/models"), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []
