"""Pydantic schemas for API request/response."""
from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    zone: str
    model_type: str
    feat_cols: list[str]
    n_seeds: int = 1
    X_train: list[dict[str, Any]]  # list of rows
    y_train: list[float]
    X_val: list[dict[str, Any]]
    y_val: list[float]
    X_test: list[dict[str, Any]]
    y_test: list[float]


class TrainResponse(BaseModel):
    model_id: str
    metrics: dict[str, Any]
    feat_cols: list[str]


class PredictRequest(BaseModel):
    model_id: str
    features: list[dict[str, Any]]  # list of rows


class PredictResponse(BaseModel):
    predictions: list[float]


class ModelInfo(BaseModel):
    model_id: str
    zone: str
    model_type: str
    created_at: str
