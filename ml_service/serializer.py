"""Model serialization for persistence. Handles all model types."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None


def save_artifact(artifact: dict, path: Path, zone: str = "", created_at: str = "") -> None:
    """Save artifact to disk. Uses joblib for most, native format for XGB/LGB/CB when standalone."""
    from datetime import datetime
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = artifact["model_type"]
    created_at = created_at or datetime.utcnow().isoformat()
    meta_extra = {"zone": zone, "created_at": created_at}

    if model_type == "RandomForest":
        p = path.with_suffix(".joblib")
        joblib.dump(artifact, p)
        _save_meta(p, model_type, artifact["feat_cols"], meta_extra)

    elif model_type == "XGBoost":
        base = path.with_suffix("")
        artifact["model"].save_model(str(base) + ".json")
        meta = {"model_type": model_type, "feat_cols": artifact["feat_cols"], "format": "xgb_json", **meta_extra}
        with open(str(base) + "_meta.json", "w") as f:
            json.dump(meta, f)

    elif model_type == "LightGBM":
        base = path.with_suffix("")
        artifact["model"].booster_.save_model(str(base) + ".txt")
        meta = {"model_type": model_type, "feat_cols": artifact["feat_cols"], "format": "lgb_txt", **meta_extra}
        with open(str(base) + "_meta.json", "w") as f:
            json.dump(meta, f)

    elif model_type == "CatBoost":
        base = path.with_suffix("")
        artifact["model"].save_model(str(base) + ".cbm")
        meta = {"model_type": model_type, "feat_cols": artifact["feat_cols"], "format": "cb_cbm", **meta_extra}
        with open(str(base) + "_meta.json", "w") as f:
            json.dump(meta, f)

    else:
        # Ensemble, Stacking: use joblib for the whole bundle
        p = path.with_suffix(".joblib")
        joblib.dump(artifact, p)
        _save_meta(p, model_type, artifact["feat_cols"], meta_extra)


def _save_meta(joblib_path: Path, model_type: str, feat_cols: list, extra: dict) -> None:
    base = joblib_path.with_suffix("")
    meta = {"model_type": model_type, "feat_cols": feat_cols, **extra}
    with open(str(base) + "_meta.json", "w") as f:
        json.dump(meta, f)


def load_artifact(path: Path) -> dict:
    """Load artifact from disk."""
    path = Path(path)
    if path.suffix == ".joblib":
        return joblib.load(path)

    base = path.with_suffix("")
    meta_path = Path(str(base) + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    model_type = meta["model_type"]
    feat_cols = meta["feat_cols"]

    if model_type == "XGBoost" and xgb:
        bst = xgb.Booster()
        bst.load_model(str(base) + ".json")
        return {"model_type": model_type, "feat_cols": feat_cols, "model": bst}

    if model_type == "LightGBM" and lgb:
        booster = lgb.Booster(model_file=str(base) + ".txt")
        return {"model_type": model_type, "feat_cols": feat_cols, "model": booster}

    if model_type == "CatBoost" and cb:
        model = cb.CatBoostRegressor()
        model.load_model(str(base) + ".cbm")
        return {"model_type": model_type, "feat_cols": feat_cols, "model": model}

    raise ValueError(f"Cannot load model_type: {model_type}")


def resolve_model_path(model_id: str, model_dir: Path) -> Path:
    """Resolve model_id to file path. Supports both .joblib and legacy xgb/lgb/cb paths."""
    base = model_dir / model_id
    if (base.with_suffix(".joblib")).exists():
        return base.with_suffix(".joblib")
    if (Path(str(base) + ".json")).exists():
        return base
    if (Path(str(base) + ".txt")).exists():
        return base
    if (Path(str(base) + ".cbm")).exists():
        return base
    raise FileNotFoundError(f"Model not found: {model_id}")
