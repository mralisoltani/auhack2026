"""Data Overview — Samples, time range, gaps, stats, and distributions per data type."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dashboard.utils import to_15min, tight_layout
from src.data_loader import (
    load_spot_price,
    load_total_load,
    load_generation_pivot,
    load_weather,
    load_flows_into,
)


def _check_gaps(df: pd.DataFrame, freq: str = "15min") -> pd.DatetimeIndex:
    """Return timestamps missing from expected full range."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DatetimeIndex([])
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    return full.difference(df.index)


def _infer_freq(df: pd.DataFrame) -> str:
    """Infer dominant frequency from index (15min or 1h)."""
    if df.empty or len(df) < 2:
        return "15min"
    diffs = df.index.to_series().diff().dropna()
    median_min = diffs.median().total_seconds() / 60
    return "1h" if median_min >= 30 else "15min"


def _stats_table(series: pd.Series) -> pd.DataFrame:
    """Build stats table for a numeric series."""
    df = pd.DataFrame({
        "mean": [series.mean()],
        "median": [series.median()],
        "min": [series.min()],
        "max": [series.max()],
        "std": [series.std()],
        "count": [int(series.count())],
        "nulls": [int(series.isna().sum())],
    }).T.rename(columns={0: "value"})
    # Round numeric columns for display (except count/null)
    for idx in ["mean", "median", "min", "max", "std"]:
        if idx in df.index and pd.notna(df.loc[idx, "value"]):
            df.loc[idx, "value"] = round(float(df.loc[idx, "value"]), 2)
    return df


def _fmt_time_range(ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> str:
    """Compact time range for metric display."""
    return f"{ts_min.strftime('%d %b %y')} – {ts_max.strftime('%d %b %y')}"


def _plot_distribution(series: pd.Series, title: str, ax=None) -> None:
    """Plot histogram of a numeric series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    clean = series.dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.hist(clean, bins=min(50, max(10, len(clean) // 100)), alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("Count")
    ax.set_title(title)


def _section_spot_price(zone: str, start: str, end: str) -> None:
    """Spot price data overview."""
    try:
        df = load_spot_price(zone)
    except FileNotFoundError:
        st.warning("No spot price data for this zone.")
        return
    sample = df.loc[start:end]
    if sample.empty:
        st.warning("No data in selected date range.")
        return
    freq = _infer_freq(df)
    gaps = _check_gaps(df, freq)
    st.markdown("**Spot price** — Day-ahead market price (EUR/MWh).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", f"{len(sample):,}")
    with c2:
        st.metric("Time range", _fmt_time_range(sample.index.min(), sample.index.max()))
    with c3:
        st.metric("Dominant freq", freq)
    with c4:
        st.metric("Missing timestamps", len(gaps))
    if len(gaps) > 0 and len(gaps) <= 10:
        st.caption(f"Gaps: {gaps.tolist()}")
    elif len(gaps) > 10:
        st.caption(f"First 5 gaps: {gaps[:5].tolist()} … ({len(gaps)} total)")
    col_stats, col_plot = st.columns([1, 1])
    with col_stats:
        stats = _stats_table(sample["price"])
        st.dataframe(stats, use_container_width=True)
    with col_plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        _plot_distribution(sample["price"], f"{zone}: Spot price distribution", ax)
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()


def _section_total_load(zone: str, start: str, end: str) -> None:
    """Total load data overview."""
    try:
        df = load_total_load(zone)
    except FileNotFoundError:
        st.warning("No total load data for this zone.")
        return
    sample = df.loc[start:end]
    if sample.empty:
        st.warning("No data in selected date range.")
        return
    freq = _infer_freq(df)
    gaps = _check_gaps(df, freq)
    st.markdown("**Total load** — Electricity consumption (MW).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", f"{len(sample):,}")
    with c2:
        st.metric("Time range", _fmt_time_range(sample.index.min(), sample.index.max()))
    with c3:
        st.metric("Dominant freq", freq)
    with c4:
        st.metric("Missing timestamps", len(gaps))
    col_stats, col_plot = st.columns([1, 1])
    with col_stats:
        stats = _stats_table(sample["load"])
        st.dataframe(stats, use_container_width=True)
    with col_plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        _plot_distribution(sample["load"], f"{zone}: Load distribution (MW)", ax)
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()


def _section_generation(zone: str, start: str, end: str) -> None:
    """Generation data overview."""
    try:
        df = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning("No generation data for this zone.")
        return
    sample = df.loc[start:end]
    if sample.empty:
        st.warning("No data in selected date range.")
        return
    freq = _infer_freq(df)
    gaps = _check_gaps(df, freq)
    st.markdown("**Generation** — Production by type (MW).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", f"{len(sample):,}")
    with c2:
        st.metric("Time range", _fmt_time_range(sample.index.min(), sample.index.max()))
    with c3:
        st.metric("Types", len(df.columns))
    with c4:
        st.metric("Missing timestamps", len(gaps))
    # Summary stats for all types (describe uses "50%" not "median")
    desc = sample.describe()
    rows = [r for r in ["count", "mean", "std", "min", "50%", "max"] if r in desc.index]
    summary = desc.loc[rows].T.rename(columns={"50%": "median"})
    summary["nulls"] = sample.isna().sum()
    st.dataframe(summary, use_container_width=True)
    # Distribution plots for all generation types
    all_cols = sample.columns.tolist()
    if all_cols:
        n_plots = len(all_cols)
        n_cols = 3
        n_rows = max(1, (n_plots + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
        axes = np.atleast_2d(axes)
        for i, col in enumerate(all_cols):
            r, c = i // n_cols, i % n_cols
            _plot_distribution(sample[col], col, axes[r, c])
        for i in range(n_plots, n_rows * n_cols):
            r, c = i // n_cols, i % n_cols
            axes[r, c].axis("off")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()


def _section_flows(zone: str, start: str, end: str) -> None:
    """Flows data overview."""
    try:
        df = load_flows_into(zone)
    except FileNotFoundError:
        st.warning("No flow data for this zone.")
        return
    pivot = df.pivot(index="time", columns="zone", values="value (MW)")
    pivot.index = pd.to_datetime(pivot.index, utc=True)
    sample = pivot.loc[start:end]
    if sample.empty:
        st.warning("No data in selected date range.")
        return
    freq = _infer_freq(pivot)
    gaps = _check_gaps(pivot, freq)
    st.markdown("**Physical flows into zone** — Cross-border imports (MW).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", f"{len(sample):,}")
    with c2:
        st.metric("Time range", _fmt_time_range(sample.index.min(), sample.index.max()))
    with c3:
        st.metric("Neighbor links", len(pivot.columns))
    with c4:
        st.metric("Missing timestamps", len(gaps))
    net = sample.sum(axis=1)
    col_stats, col_plot = st.columns([1, 1])
    with col_stats:
        stats = _stats_table(net)
        st.caption("Net import = sum of all flows into zone")
        st.dataframe(stats, use_container_width=True)
    with col_plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        _plot_distribution(net, f"{zone}: Net import distribution (MW)", ax)
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    # Per-link summary
    with st.expander("Per-link statistics"):
        desc = sample.describe()
        rows = [r for r in ["mean", "50%", "min", "max"] if r in desc.index]
        link_summary = desc.loc[rows].T.rename(columns={"50%": "median"})
        st.dataframe(link_summary, use_container_width=True)


def _section_weather(zone: str, start: str, end: str) -> None:
    """Weather data overview."""
    try:
        df = load_weather(zone)
    except FileNotFoundError:
        st.warning("No weather data for this zone.")
        return
    weather_15 = to_15min(df)
    sample = weather_15.loc[start:end]
    if sample.empty:
        st.warning("No data in selected date range.")
        return
    freq = "15min"
    gaps = _check_gaps(weather_15, freq)
    st.markdown("**Weather** — Open-Meteo (resampled to 15-min).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", f"{len(sample):,}")
    with c2:
        st.metric("Time range", _fmt_time_range(sample.index.min(), sample.index.max()))
    with c3:
        st.metric("Variables", len(sample.columns))
    with c4:
        st.metric("Missing timestamps", len(gaps))
    desc = sample.describe()
    rows = [r for r in ["count", "mean", "std", "min", "50%", "max"] if r in desc.index]
    summary = desc.loc[rows].T.rename(columns={"50%": "median"})
    summary["nulls"] = sample.isna().sum()
    st.dataframe(summary, use_container_width=True)
    # Distribution plots for all numeric variables
    plot_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    if plot_cols:
        n_plots = len(plot_cols)
        n_cols = 3
        n_rows = max(1, (n_plots + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
        axes = np.atleast_2d(axes)
        for i, col in enumerate(plot_cols):
            r, c = i // n_cols, i % n_cols
            _plot_distribution(sample[col], col, axes[r, c])
        for i in range(n_plots, n_rows * n_cols):
            r, c = i // n_cols, i % n_cols
            axes[r, c].axis("off")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()


def plot_data_overview(zone: str, start: str, end: str) -> None:
    """Data Overview: samples, time range, gaps, stats, and distributions for each data type."""
    st.subheader(f"{zone}: Data Overview")
    st.caption("Samples, time range, missing values, and feature distributions per data source.")
    sections = [
        ("Spot price", _section_spot_price),
        ("Total load", _section_total_load),
        ("Generation", _section_generation),
        ("Flows into zone", _section_flows),
        ("Weather", _section_weather),
    ]
    for title, fn in sections:
        with st.expander(title, expanded=(title == "Spot price")):
            fn(zone, start, end)
