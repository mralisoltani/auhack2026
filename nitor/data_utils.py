"""
Shared data loading and feature setup for model training scripts.
Extracted from models.ipynb for reuse across tune_*.py scripts.
"""
import numpy as np
import pandas as pd


EXCLUDED_COLS = [
    "Unnamed: 1", "id", "delivery_start", "delivery_end", 
    "target", "is_test", "is_val", "market", "market.1"
]
TARGET = "target"


def transform_target(y, method: str = "log1p", y_min: float = None):
    """Transform target for skewed distributions. method: 'log1p' (non-negative) | 'log_shifted' (handles negatives)."""
    y = np.asarray(y, dtype=float)
    if method == "none" or method is None:
        return y
    if method == "log1p":
        return np.log1p(np.maximum(y, 0))
    if method == "log_shifted":
        shift = y_min if y_min is not None else y.min()
        # Clamp to avoid log(0) or log(negative) when val/test has values below train min
        return np.log(np.maximum(y - shift + 1, 1e-10))
    raise ValueError(f"Unknown method: {method}")


def inverse_transform_target(y_pred, method: str = "log1p", y_min: float = None):
    """Inverse of transform_target to get predictions in original scale."""
    y_pred = np.asarray(y_pred, dtype=float)
    if method == "none" or method is None:
        return y_pred
    if method == "log1p":
        return np.expm1(y_pred)
    if method == "log_shifted":
        shift = y_min if y_min is not None else 0.0
        return np.exp(y_pred) + shift - 1
    raise ValueError(f"Unknown method: {method}")


def load_data(data_dir: str = "dataset/processed", return_market: bool = False, verbose: bool = True):
    """Load train, val, test splits and extract feature columns.

    If return_market=True, also returns market_train and market_val (for per-market splits).
    If verbose=True, prints row counts to verify consistency.
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    if verbose:
        print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)} "
              f"(total={len(train) + len(val) + len(test)})")

    feature_cols = [col for col in train.columns if col not in EXCLUDED_COLS]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_val = val[feature_cols]
    y_val = val[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]

    if return_market:
        return X_train, y_train, X_val, y_val, X_test, y_test, train["market"].values, val["market"].values, test["market"].values
    return X_train, y_train, X_val, y_val, X_test, y_test




def load_real_test(data_dir: str = "dataset/processed", real_test_file: str = "test_pred.csv", return_market: bool = False):
    """
    Load the real test set (no labels) for submission predictions.
    Returns (X_real_test, ids) with same feature columns as train/val/test,
    or (None, None) if the file does not exist.
    If return_market=True, also returns real_market (test_pred["market"]) when column exists, else None.
    """
    from pathlib import Path
    path = Path(data_dir) / real_test_file
    if not path.exists():
        return (None, None, None) if return_market else (None, None)
    train = pd.read_csv(Path(data_dir) / "train.csv")
    feature_cols = [col for col in train.columns if col not in EXCLUDED_COLS]
    test_pred = pd.read_csv(path)
    missing = [c for c in feature_cols if c not in test_pred.columns]
    if missing:
        raise ValueError(
            f"test_pred.csv missing feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    X_real = test_pred[feature_cols].copy()
    ids = test_pred["id"]
    if return_market:
        real_market = test_pred["market"].values if "market" in test_pred.columns else None
        return X_real, ids, real_market
    return X_real, ids


def get_trainval(X_train, y_train, X_val, y_val):
    """Concatenate train and val for tuning/full refit."""
    X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_all = np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_val, dtype=float)])
    return X_all, y_all


def get_trainval_with_market(X_train, y_train, X_val, y_val, market_train, market_val):
    """Concatenate train and val, and market labels (for per-market splits)."""
    X_all, y_all = get_trainval(X_train, y_train, X_val, y_val)
    market_all = np.concatenate([np.asarray(market_train), np.asarray(market_val)])
    return X_all, y_all, market_all


def sample_df_tail_by_market(
    df: pd.DataFrame,
    n_samples: int,
    market_col: str = "market",
    time_col: str = "delivery_start",
) -> pd.DataFrame:
    """
    Subsample df to at most n_samples rows, balanced across markets, taking the
    chronological tail per market (most recent rows). No shuffling; preserves
    time order within each market.
    """
    if market_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"DataFrame must have '{market_col}' and '{time_col}' columns")
    df_sorted = df.sort_values([market_col, time_col]).reset_index(drop=True)
    market = df_sorted[market_col].values
    n_markets = len(np.unique(market))
    n_per_market = max(1, n_samples // n_markets)
    indices = []
    for m in np.unique(market):
        mask = market == m
        idx = np.where(mask)[0]
        take = min(n_per_market, len(idx))
        indices.extend(idx[-take:].tolist())
    return df_sorted.iloc[indices].copy()


def sample_trainval_tail_by_market(X_trainval, y_trainval, market_trainval, n_samples: int):
    """
    Sample the last n_samples rows balanced across markets. For each market, takes
    the last n_per_market rows (chronological tail). Avoids bias when data is
    ordered by (market, time) and .tail() would give mostly one market.
    Returns (X_sampled, y_sampled) with at most n_samples rows.
    """
    market = np.asarray(market_trainval)
    indices = []
    n_markets = len(np.unique(market))
    n_per_market = max(1, n_samples // n_markets)
    for m in np.unique(market):
        mask = market == m
        idx = np.where(mask)[0]
        take = min(n_per_market, len(idx))
        indices.extend(idx[-take:].tolist())
    indices = np.array(indices)
    return X_trainval.iloc[indices], y_trainval.iloc[indices]


def tail_train_val_split(X, y, val_frac: float = 0.10, market=None):
    """Chronological tail split for train/val. Use last val_frac as validation.

    If market is provided, does per-market split: for each market, takes the last
    val_frac of that market's rows. This produces a validation set balanced across
    all markets (avoids bias when data is ordered by market then time).

    If market is None, uses simple row-wise tail split (original behavior).

    Returns X_tr, y_tr, X_va, y_va.
    """
    n = len(X)
    market = np.asarray(market) if market is not None else None

    if market is None or len(np.unique(market)) <= 1:
        cut = int(np.floor((1 - val_frac) * n))
        cut = max(cut, 1)
        X_tr = X.iloc[:cut]
        X_va = X.iloc[cut:]
        y_tr = y.iloc[:cut] if hasattr(y, "iloc") else y[:cut]
        y_va = y.iloc[cut:] if hasattr(y, "iloc") else y[cut:]
        return X_tr, y_tr, X_va, y_va

    val_indices = []
    for m in np.unique(market):
        mask = market == m
        indices = np.where(mask)[0]
        n_m = len(indices)
        cut = int(np.floor((1 - val_frac) * n_m))
        cut = max(cut, 0)
        val_indices.extend(indices[cut:].tolist())
    val_indices = np.array(val_indices)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_indices] = False
    train_indices = np.where(train_mask)[0]

    X_tr = X.iloc[train_indices]
    X_va = X.iloc[val_indices]
    y_tr = y.iloc[train_indices] if hasattr(y, "iloc") else y[train_indices]
    y_va = y.iloc[val_indices] if hasattr(y, "iloc") else y[val_indices]
    return X_tr, y_tr, X_va, y_va


def split_train_val_tail(tr_idx, val_frac: float = 0.10, market=None):
    """Split training indices into fit/val for CV. Last val_frac used as validation.

    If market is provided (full array aligned with data), does per-market split
    within tr_idx so validation is balanced across markets.
    Returns (tr_fit_indices, tr_val_indices).
    """
    tr_idx = np.asarray(tr_idx)
    if market is None:
        cut = int(np.floor((1 - val_frac) * len(tr_idx)))
        cut = max(cut, 1)
        return tr_idx[:cut], tr_idx[cut:]

    market = np.asarray(market)
    markets_in_fold = market[tr_idx]
    val_indices = []
    for m in np.unique(markets_in_fold):
        mask = markets_in_fold == m
        m_indices = tr_idx[mask]
        n_m = len(m_indices)
        cut = int(np.floor((1 - val_frac) * n_m))
        cut = max(cut, 0)
        val_indices.extend(m_indices[cut:].tolist())
    val_set = set(val_indices)
    tr_fit = np.array([i for i in tr_idx if i not in val_set])
    return tr_fit, np.array(val_indices)


def get_trainval_cal_test_split(
    X_train, y_train, X_val, y_val, X_test, y_test, cal_frac: float = 0.15
):
    """
    Temporal split for conformal prediction: train -> calibration -> test.
    From train+val, use first (1 - cal_frac) for training, last cal_frac for calibration.
    Test stays separate.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test : DataFrames/Series
        Raw train, val, test splits.
    cal_frac : float, default 0.15
        Fraction of train+val to use as calibration (last rows).

    Returns
    -------
    X_train_split, y_train_split : Training data (first part of train+val)
    X_cal, y_cal : Calibration data (last cal_frac of train+val)
    X_test, y_test : Test data (unchanged)
    """
    X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    n = len(X_all)
    cut = max(1, int((1 - cal_frac) * n))
    X_train_split = X_all.iloc[:cut]
    y_train_split = y_all.iloc[:cut]
    X_cal = X_all.iloc[cut:]
    y_cal = y_all.iloc[cut:]
    return X_train_split, y_train_split, X_cal, y_cal, X_test, y_test
