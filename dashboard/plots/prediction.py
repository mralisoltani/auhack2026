"""Prediction model — Spot price from load, generation, weather, flows.
Supports RandomForest, XGBoost, LightGBM, CatBoost, Ensemble, and Stacking (Ridge/MLP).
"""
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dashboard.utils import to_15min, naive_index, format_date_axis, tight_layout
from src.data_loader import (
    load_spot_price,
    load_total_load,
    load_generation_pivot,
    load_weather,
    load_flows_into,
)
from src.zone_registry import get_spot_zones

# Optional gradient boosting libs
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

SEED_POOL = [42, 123, 456, 19, 26]  # Up to 5 seeds for multi-seed averaging
VAL_FRAC = 0.1  # 10% for early stopping
TEST_FRAC = 0.1  # last 10% for prediction display


def build_features(zone: str) -> pd.DataFrame | None:
    """Build 15-min feature matrix for prediction. Returns None if data missing."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
        weather = to_15min(load_weather(zone))
        flows = load_flows_into(zone)
    except FileNotFoundError:
        return None

    pivot_flows = flows.pivot(index="time", columns="zone", values="value (MW)")
    pivot_flows.index = pd.to_datetime(pivot_flows.index, utc=True)
    pivot_flows = to_15min(pivot_flows)

    ren = [
        "SOLAR", "WIND-ONSHORE", "WIND-OFFSHORE",
        "HYDRO-ROR", "HYDRO-WATER-RESERVOIR", "BIOMASS",
    ]
    ren_cols = [c for c in ren if c in gen.columns]
    total_gen = gen.sum(axis=1)
    renewable = gen[ren_cols].sum(axis=1) if ren_cols else pd.Series(0, index=gen.index)
    gen = gen.copy()
    gen["renewable_share"] = renewable / total_gen.replace(0, np.nan)

    net_import = (
        pivot_flows.sum(axis=1).to_frame("net_import")
        if not pivot_flows.empty
        else pd.DataFrame()
    )

    idx = load.index
    features = pd.DataFrame(index=idx)
    features["hour"] = idx.hour
    features["dayofweek"] = idx.dayofweek
    features["month"] = idx.month

    for name, df in [
        ("load", load),
        ("renewable_share", gen[["renewable_share"]]),
        ("net_import", net_import),
    ]:
        if not df.empty:
            features = features.join(df.reindex(idx).ffill(), how="left")
    features = features.join(weather.reindex(idx).ffill(), how="left")
    features = features.dropna(how="all", axis=1)
    return features


@st.cache_data
def get_features_df(zone: str, _zone_list: tuple[str, ...] = ()) -> pd.DataFrame | None:
    """Cached feature matrix for zone. _zone_list invalidates cache when zones change."""
    return build_features(zone)


def _temporal_split(X: pd.DataFrame, y: pd.Series, val_frac: float, test_frac: float):
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


def _train_and_predict(
    model_type: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    seeds: list[int],
) -> tuple[np.ndarray, dict, callable]:
    """Train model and return (y_pred_test, metrics_dict, predict_fn). predict_fn(X) returns predictions."""
    y_pred_test = None
    metrics = {}
    predict_fn = None
    feat_cols = list(X_train.columns)

    if model_type == "RandomForest":
        preds = []
        for seed in seeds:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
            model.fit(X_train, y_train)
            preds.append(model.predict(X_test))
        y_pred_test = np.mean(preds, axis=0)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seeds[0])
        model.fit(X_train, y_train)
        metrics["feature_importance"] = pd.Series(model.feature_importances_, index=X_train.columns)
        predict_fn = lambda X, m=model, fc=feat_cols: m.predict(X[fc] if isinstance(X, pd.DataFrame) else X)

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
        metrics["feature_importance"] = imp
        predict_fn = lambda X, b=bst, fc=feat_cols: b.predict(xgb.DMatrix(X[fc].values if isinstance(X, pd.DataFrame) else X))

    elif model_type == "LightGBM" and HAS_LGB:
        preds = []
        for seed in seeds:
            m = _train_lgb(X_train, y_train, X_val, y_val, seed)
            preds.append(_predict_lgb(m, X_test))
        y_pred_test = np.mean(preds, axis=0)
        m = _train_lgb(X_train, y_train, X_val, y_val, seeds[0])
        metrics["feature_importance"] = pd.Series(m.feature_importances_, index=X_train.columns)
        predict_fn = lambda X, mod=m, fc=feat_cols: mod.predict(X[fc] if isinstance(X, pd.DataFrame) else X)

    elif model_type == "CatBoost" and HAS_CB:
        preds = []
        for seed in seeds:
            m = _train_cb(X_train, y_train, X_val, y_val, seed)
            preds.append(_predict_cb(m, X_test))
        y_pred_test = np.mean(preds, axis=0)
        m = _train_cb(X_train, y_train, X_val, y_val, seeds[0])
        metrics["feature_importance"] = pd.Series(m.get_feature_importance(), index=X_train.columns)
        predict_fn = lambda X, mod=m, fc=feat_cols: mod.predict(X[fc] if isinstance(X, pd.DataFrame) else X)

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
        def _ens_pred(X, b=bst0, l=lgb0, c=cb0, fc=feat_cols):
            X_ = X[fc] if isinstance(X, pd.DataFrame) else X
            return (_predict_xgb(b, X_) + _predict_lgb(l, X_) + _predict_cb(c, X_)) / 3
        predict_fn = _ens_pred

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
        def _stack_pred(X, b=bst0, l=lgb0, c=cb0, r=ridge, fc=feat_cols):
            X_ = X[fc] if isinstance(X, pd.DataFrame) else X
            P = np.column_stack([_predict_xgb(b, X_), _predict_lgb(l, X_), _predict_cb(c, X_)])
            return r.predict(P)
        predict_fn = _stack_pred

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
        def _stack_pred(X, b=bst0, l=lgb0, c=cb0, m=mlp, fc=feat_cols):
            X_ = X[fc] if isinstance(X, pd.DataFrame) else X
            P = np.column_stack([_predict_xgb(b, X_), _predict_lgb(l, X_), _predict_cb(c, X_)])
            return m.predict(P)
        predict_fn = _stack_pred

    if y_pred_test is not None:
        metrics["mae"] = mean_absolute_error(y_test, y_pred_test)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        metrics["r2"] = r2_score(y_test, y_pred_test)
        metrics["n_test"] = len(y_test)

    return y_pred_test, metrics, predict_fn


# Display names for dropdown only — edit here to add labels like (lightest), (best)
MODEL_DISPLAY_NAMES = {
    "RandomForest": "RandomForest (lightest)",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "Ensemble (avg)": "Stacking (avg)",
    "Stacking (Ridge)": "Stacking (Ridge) - (best)",
    "Stacking (MLP)": "Stacking (MLP)",
}


def _get_model_options() -> list[str]:
    opts = ["RandomForest"]
    if HAS_XGB:
        opts.append("XGBoost")
    if HAS_LGB:
        opts.append("LightGBM")
    if HAS_CB:
        opts.append("CatBoost")
    if HAS_XGB and HAS_LGB and HAS_CB:
        opts.extend(["Ensemble (avg)", "Stacking (Ridge)", "Stacking (MLP)"])
    return opts


def _display_to_internal(display: str) -> str:
    """Map dropdown display name back to internal model name."""
    rev = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
    return rev.get(display, display)


def _render_upload_inference(res: dict, zone: str) -> None:
    """Section to predict on uploaded CSV with same features as training. Shows RMSE/MAE, plot, download."""
    predict_fn = res.get("predict_fn")
    feat_cols = res.get("feat_cols")
    if predict_fn is None or not feat_cols:
        return
    with st.expander("Predict on uploaded CSV", expanded=False):
        st.caption(
            f"Upload a CSV with columns: **time** + {', '.join(feat_cols)}. "
            "Optional **price** or **value (EUR/MWh)** for RMSE/MAE evaluation."
        )
        uploaded = st.file_uploader("CSV file", type=["csv"], key="pred_upload_csv")
        if uploaded is not None:
            try:
                df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                return
            # Find time column
            time_col = None
            for c in ["time", "datetime", "date", "timestamp"]:
                if c in df.columns:
                    time_col = c
                    break
            if time_col is None:
                st.error("CSV must have a 'time' (or datetime/date) column.")
                return
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index(time_col).sort_index()
            # Check required features
            missing = [f for f in feat_cols if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}. Required: {', '.join(feat_cols)}")
                return
            # Optional actual price
            price_col = None
            for c in ["price", "value (EUR/MWh)"]:
                if c in df.columns:
                    price_col = c
                    break
            X_up = df[feat_cols].dropna(how="any")
            if X_up.empty:
                st.error("No rows with all features non-null.")
                return
            try:
                pred = predict_fn(X_up)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return
            pred_series = pd.Series(pred, index=X_up.index)
            has_actual = price_col is not None
            if has_actual:
                y_actual = df.loc[X_up.index, price_col]
                valid = y_actual.notna()
                if valid.any():
                    y_act = y_actual[valid]
                    y_pr = pred_series[valid]
                    mae = mean_absolute_error(y_act, y_pr)
                    rmse = float(np.sqrt(mean_squared_error(y_act, y_pr)))
                    r2 = r2_score(y_act, y_pr)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("MAE (unseen)", f"{mae:.2f} EUR/MWh")
                    with c2:
                        st.metric("RMSE (unseen)", f"{rmse:.2f} EUR/MWh")
                    with c3:
                        st.metric("R² (unseen)", f"{r2:.3f}")
            # Plot
            fig, ax = plt.subplots(figsize=(12, 4))
            pred_naive = naive_index(pred_series.to_frame("predicted"))
            pred_naive["predicted"].plot(ax=ax, label="Predicted", color="C1", alpha=0.9)
            if has_actual:
                y_plot = df.loc[X_up.index, price_col]
                y_plot_naive = naive_index(y_plot.to_frame("actual"))
                y_plot_naive["actual"].plot(ax=ax, label="Actual", color="C0", alpha=0.9)
            ax.set_ylabel("EUR/MWh")
            ax.set_title(f"{zone}: Predictions on uploaded data ({len(pred_series)} points)")
            ax.legend()
            format_date_axis(ax)
            tight_layout(fig)
            st.pyplot(fig)
            plt.close()
            # Download CSV
            out = pd.DataFrame({"time": pred_series.index, "predicted_price": pred_series.values})
            if has_actual:
                out["actual_price"] = df.loc[pred_series.index, price_col].values
            csv = out.to_csv(index=False)
            st.download_button(
                "Save predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                key="pred_download_csv",
            )


def plot_prediction(zone: str, start: str, end: str) -> None:
    """Model selector, Start train button, metrics, feature importance, actual vs predicted (last 10%)."""
    if zone not in get_spot_zones():
        st.warning(f"No prediction model for {zone}. Requires spot price, load, generation, weather, and flow data.")
        return

    model_options = _get_model_options()
    if len(model_options) == 1:
        st.info("Install xgboost, lightgbm, catboost for more model options.")

    options_display = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]

    # Load available features for this zone (pass zone list to invalidate cache when zones change)
    X_full = get_features_df(zone, tuple(get_spot_zones()))
    available_features = list(X_full.columns) if X_full is not None and not X_full.empty else []
    if not available_features:
        st.warning("Could not load features for this zone. Check data availability.")
        return

    with st.expander("Select features for training", expanded=True):
        selected_features = st.multiselect(
            "Features",
            options=available_features,
            default=available_features,
            key="pred_features",
            help="Choose which features to use. Use insights from other tabs to experiment.",
        )
    if not selected_features:
        st.warning("Select at least one feature.")
        return

    col_sel, col_seeds, col_btn = st.columns([2, 1, 1])
    with col_sel:
        model_type_display = st.selectbox("Model", options_display, key="pred_model_type")
        model_type = _display_to_internal(model_type_display)
    with col_seeds:
        n_seeds = st.selectbox(
            "Seeds",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"{x} seed{'s' if x > 1 else ''}",
            index=0,
            key="pred_n_seeds",
        )
        seeds = SEED_POOL[:n_seeds]
    with col_btn:
        st.markdown('<div style="height: 1.6rem;"></div>', unsafe_allow_html=True)
        train_clicked = st.button("Start train", type="primary", key="pred_train_btn")

    if train_clicked:
        with st.spinner("Training model..."):
            try:
                X = get_features_df(zone, tuple(get_spot_zones()))
                sp = load_spot_price(zone)
            except FileNotFoundError:
                st.error(f"Could not load data for {zone}.")
                return
            if X is None or X.empty:
                st.error("Could not build features.")
                return

            y = to_15min(sp)["price"].reindex(X.index).ffill()
            valid = X.notna().all(axis=1) & y.notna()
            X_clean = X[valid].dropna()
            y_clean = y[valid].loc[X_clean.index].squeeze()

            # Filter to selected date range
            X_range = X_clean.loc[start:end]
            y_range = y_clean.loc[X_range.index]
            if len(X_range) < 100:
                st.error("Insufficient data in selected range. Need at least 100 samples.")
                return

            split = _temporal_split(X_range, y_range, VAL_FRAC, TEST_FRAC)
            if split[0] is None:
                st.error("Could not create train/val/test split. Try a longer date range.")
                return
            X_train, y_train, X_val, y_val, X_test, y_test = split

            # Filter to selected features
            feat_ok = [f for f in selected_features if f in X_train.columns]
            if not feat_ok:
                st.error("Selected features not found in data.")
                return
            X_train = X_train[feat_ok]
            X_val = X_val[feat_ok]
            X_test = X_test[feat_ok]

            y_pred_test, metrics, predict_fn = _train_and_predict(
                model_type, X_train, y_train, X_val, y_val, X_test, y_test, seeds
            )

            if y_pred_test is None:
                st.error(f"Model '{model_type}' not available. Install xgboost, lightgbm, catboost.")
                return

            test_start = y_test.index[0]
            test_end = y_test.index[-1]
            st.session_state["prediction_result"] = {
                "zone": zone,
                "start": start,
                "end": end,
                "model_type": model_type,
                "model_type_display": model_type_display,
                "n_seeds": n_seeds,
                "selected_features": tuple(sorted(selected_features)),
                "y_range": y_range,
                "y_test": y_test,
                "y_pred": y_pred_test,
                "test_start": test_start,
                "test_end": test_end,
                "metrics": metrics,
                "predict_fn": predict_fn,
                "feat_cols": feat_ok,
                "X_full": X_clean,
                "y_full": y_clean,
            }
        st.rerun()

    res = st.session_state.get("prediction_result")
    selected_tuple = tuple(sorted(selected_features))
    if res is None or res["zone"] != zone or res["start"] != start or res["end"] != end or res["model_type"] != model_type or res.get("n_seeds") != n_seeds or res.get("selected_features") != selected_tuple:
        st.caption("Select model and click **Start train** to train on the first 90% of the date range and predict on the last 10%.")
        return

    metrics = res["metrics"]
    y_range = res["y_range"]
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    test_start = res["test_start"]
    test_end = res["test_end"]
    feat_imp = res["metrics"].get("feature_importance")

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("MAE (last 10%)", f"{metrics['mae']:.2f} EUR/MWh")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.2f} EUR/MWh")
    with col3:
        st.metric("R²", f"{metrics['r2']:.3f}")
    with col4:
        st.metric("Test samples", metrics["n_test"])
    with col5:
        st.metric("Seeds", res.get("n_seeds", 1))

    # Feature importance (if available)
    if feat_imp is not None and len(feat_imp) > 0:
        imp = feat_imp.sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        imp.head(12).plot(kind="barh", ax=ax_imp, color="steelblue", alpha=0.8)
        ax_imp.set_xlabel("Importance")
        seed_label = f", {res.get('n_seeds', 1)} seed{'s' if res.get('n_seeds', 1) > 1 else ''}" if res.get("n_seeds") else ""
        model_label = res.get("model_type_display", res["model_type"])
        ax_imp.set_title(f"{zone}: Feature importance ({model_label}{seed_label})")
        tight_layout(fig_imp)
        st.pyplot(fig_imp)
        plt.close()

    # Live prediction state
    if "live_prediction_active" not in st.session_state:
        st.session_state["live_prediction_active"] = False
    if "live_prediction_data" not in st.session_state:
        st.session_state["live_prediction_data"] = None

    predict_fn = res.get("predict_fn")
    feat_cols = res.get("feat_cols")
    X_full = res.get("X_full")
    y_full = res.get("y_full")

    # Live prediction controls (only when model supports it)
    live_controls = predict_fn is not None and feat_cols is not None and X_full is not None and y_full is not None
    if live_controls:
        live_mode = st.radio(
            "Live prediction mode",
            options=["Inference only", "With retraining"],
            index=0,
            key="live_mode",
            horizontal=True,
            help="Inference only: use the trained model as-is. With retraining: retrain on expanding data before each prediction.",
        )
        use_retraining = live_mode == "With retraining"
        points_per_update = st.number_input(
            "Points per update",
            min_value=1,
            max_value=96,
            value=4,
            step=1,
            key="live_points",
            help="Number of 15-min intervals to predict each update (4 = 1 hour).",
        )
        update_interval = st.number_input(
            "Update interval (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="live_interval",
            help="How often to add predictions.",
        )
        col_start, col_stop, _ = st.columns([1, 1, 2])
        with col_start:
            start_live = st.button("Start live", key="live_start", disabled=st.session_state["live_prediction_active"])
        with col_stop:
            stop_live = st.button("Stop", key="live_stop", disabled=not st.session_state["live_prediction_active"])

        if stop_live:
            st.session_state["live_prediction_active"] = False
            st.rerun()

        if start_live:
            st.session_state["live_prediction_active"] = True
            st.session_state["live_prediction_data"] = {"ts": [], "pred": []}
            st.rerun()

        # Run live prediction step if active
        if st.session_state["live_prediction_active"]:
            after = X_full.loc[X_full.index > test_end]
            if after.empty:
                st.warning("No feature data available after the selected end date. Live prediction stopped.")
                st.session_state["live_prediction_active"] = False
            else:
                live_data = st.session_state["live_prediction_data"]
                last_ts = live_data["ts"][-1] if live_data["ts"] else None
                chunk_size = int(points_per_update)
                if last_ts is None:
                    chunk = after.iloc[:chunk_size]
                else:
                    chunk = after.loc[after.index > last_ts].iloc[:chunk_size]
                if not chunk.empty:
                    X_chunk = chunk[feat_cols].dropna()
                    if not X_chunk.empty:
                        if use_retraining:
                            # Retrain on expanding window (data before chunk), then predict chunk
                            before_end = X_chunk.index[0] - pd.Timedelta(minutes=15)
                            X_before = X_full.loc[X_full.index <= before_end][feat_cols].dropna()
                            y_before = y_full.loc[X_before.index].squeeze()
                            if len(X_before) >= 100:
                                split = _temporal_split(X_before, y_before, VAL_FRAC, TEST_FRAC)
                                if split[0] is not None:
                                    X_tr, y_tr, X_va, y_va, _, _ = split
                                    y_chunk = y_full.loc[X_chunk.index].squeeze()
                                    with st.spinner("Retraining model..."):
                                        _, _, pred_fn = _train_and_predict(
                                            res["model_type"], X_tr, y_tr, X_va, y_va,
                                            X_chunk, y_chunk, SEED_POOL[: res.get("n_seeds", 1)],
                                        )
                                    pred = pred_fn(X_chunk) if pred_fn is not None else predict_fn(X_chunk)
                                else:
                                    pred = predict_fn(X_chunk)
                            else:
                                pred = predict_fn(X_chunk)
                        else:
                            pred = predict_fn(X_chunk)
                        live_data["ts"].extend(X_chunk.index.tolist())
                        live_data["pred"].extend(pred.tolist())
                        # RMSE of this chunk (pred vs actual)
                        y_actual_chunk = y_full.reindex(X_chunk.index)
                        valid = y_actual_chunk.notna()
                        if valid.any():
                            pred_arr = np.array(pred)[valid.to_numpy()]
                            actual_arr = np.array(y_actual_chunk[valid])
                            live_data["last_chunk_rmse"] = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
                else:
                    st.session_state["live_prediction_active"] = False

    # Single continuous plot: actual + predicted (test) + live predicted + ground truth in live period
    fig, ax = plt.subplots(figsize=(14, 5))
    live_data = st.session_state.get("live_prediction_data") or {"ts": [], "pred": []}
    has_live = live_controls and live_data["ts"]

    # Determine x range: from start through end of live (or full range)
    plot_end = test_end
    if has_live:
        plot_end = max(plot_end, pd.Timestamp(live_data["ts"][-1]))
    plot_start = y_range.index[0]

    # Actual (ground truth) — full range including live period
    y_plot = y_full.loc[plot_start:plot_end] if y_full is not None else y_range
    if y_plot is not None and not y_plot.empty:
        y_plot_naive = naive_index(y_plot.to_frame("actual"))
        y_plot_naive["actual"].plot(ax=ax, label="Actual (ground truth)", alpha=0.9, color="C0", zorder=3)

    # Predicted (test period)
    pred_series = pd.Series(y_pred, index=y_test.index)
    pred_naive = naive_index(pred_series.to_frame("predicted"))
    pred_naive["predicted"].plot(ax=ax, label="Predicted (test)", alpha=0.9, color="C1", zorder=3)

    # Live predicted — continues after test_end
    if has_live:
        live_pred_series = pd.Series(live_data["pred"], index=pd.DatetimeIndex(live_data["ts"]))
        live_pred_naive = naive_index(live_pred_series.to_frame("live_pred"))
        live_pred_naive["live_pred"].plot(ax=ax, label="Live predicted", alpha=0.9, color="C2", zorder=3)

    # Shading: test period (blue), live period (green)
    ts_test_start = naive_index(pd.DataFrame(index=[test_start])).index[0]
    ts_test_end = naive_index(pd.DataFrame(index=[test_end])).index[0]
    ax.axvspan(ts_test_start, ts_test_end, alpha=0.12, color="steelblue", zorder=0)
    ax.axvline(x=ts_test_start, color="steelblue", linestyle="--", linewidth=1.2, zorder=2)
    if has_live:
        ts_live_start = naive_index(pd.DataFrame(index=[pd.Timestamp(live_data["ts"][0])])).index[0]
        ts_live_end = naive_index(pd.DataFrame(index=[pd.Timestamp(live_data["ts"][-1])])).index[0]
        ax.axvspan(ts_live_start, ts_live_end, alpha=0.12, color="green", zorder=0)
        ax.axvline(x=ts_live_start, color="green", linestyle="--", linewidth=1.2, zorder=2)
        ymin_ax, ymax_ax = ax.get_ylim()
        ax.text(ts_live_start, ymax_ax - (ymax_ax - ymin_ax) * 0.04, " Live ", fontsize=9, color="green", fontweight="bold", zorder=2)
    ymin_ax, ymax_ax = ax.get_ylim()
    ax.text(ts_test_start, ymax_ax - (ymax_ax - ymin_ax) * 0.04, " Test ", fontsize=9, color="steelblue", fontweight="bold", zorder=2)

    ax.set_ylabel("EUR/MWh")
    seed_label = f" — {res.get('n_seeds', 1)} seed{'s' if res.get('n_seeds', 1) > 1 else ''}" if res.get("n_seeds", 1) > 1 else ""
    title = f"{zone}: Spot price — Actual vs Predicted (continuous)"
    if has_live:
        title += f" — Live: {len(live_data['ts'])} points"
    ax.set_title(title + seed_label)
    ax.legend()
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()

    # Live RMSE (last chunk) and error plot
    if has_live and live_controls:
        last_rmse = live_data.get("last_chunk_rmse")
        if last_rmse is not None:
            st.metric("Live RMSE (last chunk)", f"{last_rmse:.2f} EUR/MWh")
        # Plot: prediction - actual (error) over live period
        live_ts = pd.DatetimeIndex(live_data["ts"])
        live_pred_arr = np.array(live_data["pred"])
        live_actual = y_full.reindex(live_ts)
        err_series = pd.Series(live_pred_arr - np.array(live_actual), index=live_ts).dropna()
        if not err_series.empty:
            fig_err, ax_err = plt.subplots(figsize=(14, 3))
            err_naive = naive_index(err_series.to_frame("error"))
            err_naive["error"].plot(ax=ax_err, color="C3", alpha=0.9, label="Predicted − Actual")
            ax_err.axhline(y=0, color="gray", linestyle="--", linewidth=1)
            ax_err.set_ylabel("EUR/MWh")
            ax_err.set_title(f"{zone}: Live prediction error (Predicted − Actual)")
            ax_err.legend()
            format_date_axis(ax_err)
            tight_layout(fig_err)
            st.pyplot(fig_err)
            plt.close()

    # Predict on uploaded CSV
    _render_upload_inference(res, zone)

    if live_controls:
        st.caption("**Start live** to extend predictions. Ground truth (actual) is shown so you can compare accuracy. Click **Stop** to halt.")
        if st.session_state["live_prediction_active"]:
            time.sleep(update_interval)
            st.rerun()
