"""Shared utilities for data processing."""
import pandas as pd


def to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Expand hourly/mixed data to 15-min with forward fill."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
    return df.reindex(full_idx).ffill()
