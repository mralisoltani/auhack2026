"""FastAPI ML service for training and inference."""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ModelInfo, PredictRequest, PredictResponse, TrainRequest, TrainResponse
from serializer import load_artifact, resolve_model_path, save_artifact
from trainer import predict_from_artifact, train_and_predict

from src.config import MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ML API", description="Training and inference for spot price prediction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _safe_model_id(zone: str, model_type: str) -> str:
    """Create filesystem-safe model_id."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_type = model_type.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{zone}_{safe_type}_{ts}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    X_train = pd.DataFrame(req.X_train)
    y_train = pd.Series(req.y_train)
    X_val = pd.DataFrame(req.X_val)
    y_val = pd.Series(req.y_val)
    X_test = pd.DataFrame(req.X_test)
    y_test = pd.Series(req.y_test)

    feat_cols = [c for c in req.feat_cols if c in X_train.columns]
    if not feat_cols:
        raise HTTPException(status_code=400, detail="No matching feature columns")
    X_train = X_train[feat_cols]
    X_val = X_val[feat_cols]
    X_test = X_test[feat_cols]

    seeds = [42, 123, 456, 19, 26][: req.n_seeds]
    try:
        y_pred, metrics, artifact = train_and_predict(
            req.model_type, X_train, y_train, X_val, y_val, X_test, y_test, seeds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if y_pred is None:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_type}' not available")

    model_id = _safe_model_id(req.zone, req.model_type)
    path = MODEL_DIR / model_id
    try:
        save_artifact(artifact, path, zone=req.zone, created_at=datetime.utcnow().isoformat())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {e}")

    return TrainResponse(model_id=model_id, metrics=metrics, feat_cols=feat_cols)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        path = resolve_model_path(req.model_id, MODEL_DIR)
        artifact = load_artifact(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X = pd.DataFrame(req.features)
    feat_cols = artifact["feat_cols"]
    missing = [c for c in feat_cols if c not in X.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    X = X[feat_cols].dropna(how="any")
    if X.empty:
        raise HTTPException(status_code=400, detail="No valid rows after dropna")

    try:
        pred = predict_from_artifact(artifact, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(predictions=pred.tolist())


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    import json
    infos = []
    seen = set()
    for f in MODEL_DIR.iterdir():
        if f.name.endswith("_meta.json"):
            continue
        base = f.stem
        if base in seen:
            continue
        meta_path = MODEL_DIR / (base + "_meta.json")
        if f.suffix in (".joblib", ".json", ".txt", ".cbm") and meta_path.exists():
            seen.add(base)
            try:
                with open(meta_path) as fp:
                    meta = json.load(fp)
                infos.append(ModelInfo(
                    model_id=base,
                    zone=meta.get("zone", "?"),
                    model_type=meta.get("model_type", "?"),
                    created_at=meta.get("created_at", datetime.fromtimestamp(meta_path.stat().st_mtime).isoformat()),
                ))
            except Exception:
                pass
    return infos


@app.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        path = resolve_model_path(model_id, MODEL_DIR)
        artifact = load_artifact(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"model_id": model_id, "model_type": artifact["model_type"], "feat_cols": artifact["feat_cols"]}
