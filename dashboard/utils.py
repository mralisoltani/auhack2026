"""
Shared utilities for the dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import load_flows_into, load_all_spot_prices


def format_date_axis(ax, fmt: str = "%d %b %Y") -> None:
    """Apply consistent date formatting to x-axis. Handles both matplotlib days and pandas minutes."""
    from matplotlib.ticker import FuncFormatter

    def _format(x, _pos):
        x = float(x)
        # Matplotlib plot(): days since epoch (~2e4 for 2025)
        # Pandas .plot(): minutes since epoch (~3e7 for 2025)
        if 1000 < x < 100000:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=x)
        elif 1e6 < x < 1e10:
            dt = datetime.fromtimestamp(x * 60, tz=timezone.utc)
        else:
            return ""
        return dt.strftime(fmt)

    ax.xaxis.set_major_formatter(FuncFormatter(_format))
    ax.tick_params(axis="x", rotation=45)


def tight_layout(fig, pad: float = 1.2) -> None:
    """Apply tight layout with extra bottom space for rotated date labels."""
    fig.tight_layout(pad=pad, h_pad=1.5, rect=[0, 0.06, 1, 1])


def to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Expand hourly/mixed data to 15-min with ffill. Re-exported from src.utils."""
    from src.utils import to_15min as _to_15min
    return _to_15min(df)


def naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert index to timezone-naive for matplotlib compatibility."""
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


@st.cache_data
def get_flows_pivot_15min(zone: str, _zone_list: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load flows into zone, pivot, resample to 15-min. _zone_list invalidates cache when zones change."""
    flows = load_flows_into(zone)
    pivot = flows.pivot(index="time", columns="zone", values="value (MW)")
    pivot.index = pd.to_datetime(pivot.index, utc=True)
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="15min", tz="UTC")
    pivot = pivot.reindex(full_idx).ffill()
    pivot["net_import"] = pivot.sum(axis=1)
    return pivot


@st.cache_data
def get_prices_15min(_zone_list: tuple[str, ...]) -> pd.DataFrame:
    """Load all spot prices at 15-min. _zone_list is for cache invalidation when zones change."""
    prices = load_all_spot_prices(list(_zone_list))
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq="15min", tz="UTC")
    return prices.reindex(full_idx).ffill()
