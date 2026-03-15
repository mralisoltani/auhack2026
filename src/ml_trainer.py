"""
Shared training logic for spot price prediction.
Used by both the dashboard (local training) and the ML API service.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None
try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    cb = None

SEED_POOL = [42, 123, 456, 19, 26]


def temporal_split(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.1, test_frac: float = 0.1
):
    """Split temporally: train (rest), val (before last test_frac), test (last test_frac)."""
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    if n_train < 10 or n_val < 5 or n_test < 5:
        return None, None, None, None, None, None
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train : n_train + n_val]
    y_val = y.iloc[n_train : n_train + n_val]
    X_test = X.iloc[-n_test:]
    y_test = y.iloc[-n_test:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_xgb(X_tr, y_tr, X_va, y_va, seed: int):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "tree_method": "hist", "max_depth": 6, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
    }
    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                    early_stopping_rounds=150, verbose_eval=False)
    return bst


def _train_lgb(X_tr, y_tr, X_va, y_va, seed: int):
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        num_leaves=63, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_estimators=2000, verbosity=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    return model


def _train_cb(X_tr, y_tr, X_va, y_va, seed: int):
    model = cb.CatBoostRegressor(
        loss_function="RMSE", iterations=2000, learning_rate=0.05,
        depth=6, subsample=0.8, rsm=0.8, random_seed=seed, verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
              use_best_model=True, early_stopping_rounds=150)
    return model


def _predict_xgb(model, X):
    return model.predict(xgb.DMatrix(X))


def _predict_lgb(model, X):
    return model.predict(X)


def _predict_cb(model, X):
    return model.predict(X)


def train_and_predict(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seeds: list[int],
) -> tuple[np.ndarray | None, dict, dict]:
    """
    Train model and return (y_pred_test, metrics_dict, artifact).
    artifact is a dict for serialization: {model_type, feat_cols, models, ...}
    """
    y_pred_test = None
    metrics = {}
    feat_cols = list(X_train.columns)
    artifact = {"model_type": model_type, "feat_cols": feat_cols}

    if model_type == "RandomForest":
        preds = []
        for seed in seeds:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
            model.fit(X_train, y_train)
            preds.append(model.predict(X_test))
        y_pred_test = np.mean(preds, axis=0)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seeds[0])
        model.fit(X_train, y_train)
        metrics["feature_importance"] = pd.Series(model.feature_importances_, index=X_train.columns).to_dict()
        artifact["model"] = model

    elif model_type == "XGBoost" and HAS_XGB:
        preds = []
        for seed in seeds:
            bst = _train_xgb(X_train.values, y_train.values, X_val.values, y_val.values, seed)
            preds.append(_predict_xgb(bst, X_test))
        y_pred_test = np.mean(preds, axis=0)
        bst = _train_xgb(X_train.values, y_train.values, X_val.values, y_val.values, seeds[0])
        score = bst.get_score(importance_type="gain")
        imp = pd.Series(0.0, index=X_train.columns)
        for k, v in score.items():
            i = int(k[1:]) if k.startswith("f") else int(k)
            if i < len(imp):
                imp.iloc[i] = v
        metrics["feature_importance"] = imp.to_dict()
        artifact["model"] = bst

    elif model_type == "LightGBM" and HAS_LGB:
        preds = []
        for seed in seeds:
            m = _train_lgb(X_train, y_train, X_val, y_val, seed)
            preds.append(_predict_lgb(m, X_test))
        y_pred_test = np.mean(preds, axis=0)
        m = _train_lgb(X_train, y_train, X_val, y_val, seeds[0])
        metrics["feature_importance"] = pd.Series(m.feature_importances_, index=X_train.columns).to_dict()
        artifact["model"] = m

    elif model_type == "CatBoost" and HAS_CB:
        preds = []
        for seed in seeds:
            m = _train_cb(X_train, y_train, X_val, y_val, seed)
            preds.append(_predict_cb(m, X_test))
        y_pred_test = np.mean(preds, axis=0)
        m = _train_cb(X_train, y_train, X_val, y_val, seeds[0])
        metrics["feature_importance"] = pd.Series(m.get_feature_importance(), index=X_train.columns).to_dict()
        artifact["model"] = m

    elif model_type == "Ensemble (avg)" and (HAS_XGB and HAS_LGB and HAS_CB):
        preds = []
        bst0 = lgb0 = cb0 = None
        for s in seeds:
            bst = _train_xgb(X_train.values, y_train.values, X_val.values, y_val.values, s)
            m_lgb = _train_lgb(X_train, y_train, X_val, y_val, s)
            m_cb = _train_cb(X_train, y_train, X_val, y_val, s)
            if bst0 is None:
                bst0, lgb0, cb0 = bst, m_lgb, m_cb
            preds.append((_predict_xgb(bst, X_test) + _predict_lgb(m_lgb, X_test) + _predict_cb(m_cb, X_test)) / 3)
        y_pred_test = np.mean(preds, axis=0)
        metrics["feature_importance"] = None
        artifact["xgb"] = bst0
        artifact["lgb"] = lgb0
        artifact["cb"] = cb0

    elif model_type == "Stacking (Ridge)" and (HAS_XGB and HAS_LGB and HAS_CB):
        p_xgb_va, p_lgb_va, p_cb_va = [], [], []
        p_xgb_te, p_lgb_te, p_cb_te = [], [], []
        bst0 = lgb0 = cb0 = None
        for s in seeds:
            bst = _train_xgb(X_train.values, y_train.values, X_val.values, y_val.values, s)
            p_xgb_va.append(_predict_xgb(bst, X_val))
            p_xgb_te.append(_predict_xgb(bst, X_test))
            m_lgb = _train_lgb(X_train, y_train, X_val, y_val, s)
            p_lgb_va.append(_predict_lgb(m_lgb, X_val))
            p_lgb_te.append(_predict_lgb(m_lgb, X_test))
            m_cb = _train_cb(X_train, y_train, X_val, y_val, s)
            p_cb_va.append(_predict_cb(m_cb, X_val))
            p_cb_te.append(_predict_cb(m_cb, X_test))
            if bst0 is None:
                bst0, lgb0, cb0 = bst, m_lgb, m_cb
        P_va = np.column_stack([np.mean(p_xgb_va, 0), np.mean(p_lgb_va, 0), np.mean(p_cb_va, 0)])
        P_te = np.column_stack([np.mean(p_xgb_te, 0), np.mean(p_lgb_te, 0), np.mean(p_cb_te, 0)])
        ridge = Ridge(alpha=1.0).fit(P_va, y_val.values)
        y_pred_test = ridge.predict(P_te)
        metrics["feature_importance"] = None
        artifact["xgb"] = bst0
        artifact["lgb"] = lgb0
        artifact["cb"] = cb0
        artifact["meta"] = ridge

    elif model_type == "Stacking (MLP)" and (HAS_XGB and HAS_LGB and HAS_CB):
        p_xgb_va, p_lgb_va, p_cb_va = [], [], []
        p_xgb_te, p_lgb_te, p_cb_te = [], [], []
        bst0 = lgb0 = cb0 = None
        for s in seeds:
            bst = _train_xgb(X_train.values, y_train.values, X_val.values, y_val.values, s)
            p_xgb_va.append(_predict_xgb(bst, X_val))
            p_xgb_te.append(_predict_xgb(bst, X_test))
            m_lgb = _train_lgb(X_train, y_train, X_val, y_val, s)
            p_lgb_va.append(_predict_lgb(m_lgb, X_val))
            p_lgb_te.append(_predict_lgb(m_lgb, X_test))
            m_cb = _train_cb(X_train, y_train, X_val, y_val, s)
            p_cb_va.append(_predict_cb(m_cb, X_val))
            p_cb_te.append(_predict_cb(m_cb, X_test))
            if bst0 is None:
                bst0, lgb0, cb0 = bst, m_lgb, m_cb
        P_va = np.column_stack([np.mean(p_xgb_va, 0), np.mean(p_lgb_va, 0), np.mean(p_cb_va, 0)])
        P_te = np.column_stack([np.mean(p_xgb_te, 0), np.mean(p_lgb_te, 0), np.mean(p_cb_te, 0)])
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=500, early_stopping=True, random_state=42)
        mlp.fit(P_va, y_val.values)
        y_pred_test = mlp.predict(P_te)
        metrics["feature_importance"] = None
        artifact["xgb"] = bst0
        artifact["lgb"] = lgb0
        artifact["cb"] = cb0
        artifact["meta"] = mlp

    if y_pred_test is not None:
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred_test))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        metrics["r2"] = float(r2_score(y_test, y_pred_test))
        metrics["n_test"] = len(y_test)

    return y_pred_test, metrics, artifact


def predict_from_artifact(artifact: dict, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Run prediction using a loaded artifact. X can be DataFrame or numpy array."""
    model_type = artifact["model_type"]
    feat_cols = artifact["feat_cols"]
    if isinstance(X, pd.DataFrame):
        X_arr = X[feat_cols].values if feat_cols else X.values
    else:
        X_arr = X

    if model_type == "RandomForest":
        return artifact["model"].predict(X_arr)

    elif model_type == "XGBoost":
        return artifact["model"].predict(xgb.DMatrix(X_arr))

    elif model_type == "LightGBM":
        return artifact["model"].predict(X_arr if isinstance(X, np.ndarray) else X[feat_cols])

    elif model_type == "CatBoost":
        return artifact["model"].predict(X_arr if isinstance(X, np.ndarray) else X[feat_cols])

    elif model_type == "Ensemble (avg)":
        p1 = _predict_xgb(artifact["xgb"], X_arr)
        p2 = _predict_lgb(artifact["lgb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols])
        p3 = _predict_cb(artifact["cb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols])
        return (p1 + p2 + p3) / 3

    elif model_type == "Stacking (Ridge)":
        P = np.column_stack([
            _predict_xgb(artifact["xgb"], X_arr),
            _predict_lgb(artifact["lgb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols]),
            _predict_cb(artifact["cb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols]),
        ])
        return artifact["meta"].predict(P)

    elif model_type == "Stacking (MLP)":
        P = np.column_stack([
            _predict_xgb(artifact["xgb"], X_arr),
            _predict_lgb(artifact["lgb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols]),
            _predict_cb(artifact["cb"], X_arr if isinstance(X, np.ndarray) else X[feat_cols]),
        ])
        return artifact["meta"].predict(P)

    raise ValueError(f"Unknown model_type: {model_type}")
