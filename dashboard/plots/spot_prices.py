"""Spot price plots — Notebooks 03, 08."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import (
    get_prices_15min,
    to_15min,
    naive_index,
    format_date_axis,
    tight_layout,
)
from src.data_loader import load_spot_price, load_total_load, load_generation_pivot, load_weather
from src.zone_registry import get_spot_zones


def plot_spot_prices(start: str, end: str) -> None:
    """Spot prices across zones. User selects which zones to show via checkboxes."""
    prices = get_prices_15min(tuple(get_spot_zones()))
    sample = naive_index(prices.loc[start:end])
    if sample.empty:
        st.warning("No data for selected range.")
        return

    available_zones = list(sample.columns)
    st.caption("Select zones to show:")
    cols = st.columns(min(5, len(available_zones)))
    selected_zones = []
    for i, z in enumerate(available_zones):
        with cols[i % len(cols)]:
            if st.checkbox(z, value=True, key=f"spot_zone_{z}"):
                selected_zones.append(z)
    if not selected_zones:
        st.warning("Select at least one zone.")
        return

    sample_sel = sample[selected_zones]
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_sel.plot(ax=ax, alpha=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Spot prices across zones")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_zone_spot_price(zone: str, start: str, end: str) -> None:
    """Spot price for selected zone over date range."""
    try:
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sp_15 = to_15min(sp)
    sample = naive_index(sp_15.loc[start:end])
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    sample["price"].plot(ax=ax, color="C0", alpha=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{zone}: Spot price")
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_price_by_hour(zone: str) -> None:
    """Average spot price by hour (all-time)."""
    try:
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sp_ = sp.copy()
    sp_["hour"] = sp_.index.hour
    by_hour = sp_.groupby("hour")["price"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour.index, by_hour["mean"], yerr=by_hour["std"], capsize=2, alpha=0.8)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Mean spot price (EUR/MWh)")
    ax.set_title(f"{zone}: Average spot price by hour")
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_negative_prices(zone: str, start: str, end: str) -> None:
    """Negative price analysis: frequency, duration, and conditions."""
    try:
        sp = to_15min(load_spot_price(zone))
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sample = sp.loc[start:end]
    if sample.empty:
        return

    neg = sample[sample["price"] < 0]
    total_hrs = len(sample) / 4
    neg_hrs = len(neg) / 4
    pct = (neg_hrs / total_hrs * 100) if total_hrs > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Negative price hours", f"{neg_hrs:.0f} h")
    with c2:
        st.metric("% of period", f"{pct:.1f}%")
    with c3:
        st.metric("Min price", f"{sample['price'].min():.1f} EUR/MWh")
    with c4:
        avg_neg = neg["price"].mean() if not neg.empty else 0
        st.metric("Avg negative price", f"{avg_neg:.1f} EUR/MWh")

    # Duration of consecutive negative episodes
    is_neg = (sample["price"] < 0).astype(int)
    blocks = is_neg.diff().ne(0).cumsum()
    episodes = is_neg.groupby(blocks).agg(["sum", "first"])
    episodes = episodes[episodes["first"] == 1]
    durations_h = episodes["sum"] / 4
    if not durations_h.empty:
        st.caption(f"Negative episodes: {len(durations_h)} | avg duration: {durations_h.mean():.1f}h | max: {durations_h.max():.1f}h")

    # Price histogram highlighting negatives
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    prices = sample["price"].dropna()
    axes[0].hist(prices[prices >= 0], bins=50, alpha=0.7, color="steelblue", label="Positive")
    if not neg.empty:
        axes[0].hist(prices[prices < 0], bins=max(5, len(neg) // 20), alpha=0.8, color="#ff4b4b", label="Negative")
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_xlabel("EUR/MWh")
    axes[0].set_ylabel("Count (15-min intervals)")
    axes[0].set_title(f"{zone}: Price distribution")
    axes[0].legend()

    # Conditions during negative prices: wind, load
    try:
        gen = load_generation_pivot(zone)
        load_df = load_total_load(zone)
        wind_cols = [c for c in gen.columns if "WIND" in c]
        gen["wind_total"] = gen[wind_cols].sum(axis=1)
        joined = sample.join(gen[["wind_total"]], how="inner").join(load_df, how="inner").dropna()
        if not joined.empty:
            neg_j = joined[joined["price"] < 0]
            pos_j = joined[joined["price"] >= 0]
            labels = ["Wind gen (MW)", "Load (MW)"]
            neg_vals = [neg_j["wind_total"].mean(), neg_j["load"].mean()] if not neg_j.empty else [0, 0]
            pos_vals = [pos_j["wind_total"].mean(), pos_j["load"].mean()]
            x = np.arange(len(labels))
            w = 0.35
            axes[1].bar(x - w/2, neg_vals, w, label="Negative price hrs", color="#ff4b4b", alpha=0.8)
            axes[1].bar(x + w/2, pos_vals, w, label="Positive price hrs", color="steelblue", alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels)
            axes[1].set_ylabel("MW")
            axes[1].set_title("Conditions: negative vs positive prices")
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No generation/load data", ha="center", va="center", transform=axes[1].transAxes)
    except (FileNotFoundError, KeyError):
        axes[1].text(0.5, 0.5, "No generation/load data", ha="center", va="center", transform=axes[1].transAxes)

    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_price_volatility(zone: str, start: str, end: str) -> None:
    """Rolling price volatility (24h and 7d std dev)."""
    try:
        sp = to_15min(load_spot_price(zone))
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sample = sp.loc[start:end]
    if sample.empty or len(sample) < 96:
        st.warning("Need at least 1 day of data for volatility analysis.")
        return

    vol_24h = sample["price"].rolling(96, min_periods=48).std()   # 96 x 15min = 24h
    vol_7d = sample["price"].rolling(672, min_periods=336).std()   # 672 x 15min = 7d

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg 24h volatility", f"{vol_24h.mean():.2f} EUR/MWh")
    with c2:
        st.metric("Max 24h volatility", f"{vol_24h.max():.2f} EUR/MWh")
    with c3:
        st.metric("Avg 7d volatility", f"{vol_7d.mean():.2f} EUR/MWh" if vol_7d.notna().any() else "N/A")

    sample_n = naive_index(sample)
    vol_24h_n = naive_index(vol_24h.to_frame("vol_24h"))
    vol_7d_n = naive_index(vol_7d.to_frame("vol_7d"))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1], sharex=True)
    axes[0].plot(sample_n.index, sample_n["price"], alpha=0.7, lw=0.8, color="C0", label="Spot price")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title(f"{zone}: Spot price with volatility regimes")
    axes[0].legend(loc="upper left")

    # Shade high-vol periods (above 75th percentile of 24h vol)
    thresh = vol_24h.quantile(0.75)
    high_vol = vol_24h > thresh
    high_vol_n = naive_index(high_vol.to_frame("hv"))
    for i in range(1, len(high_vol_n)):
        if high_vol_n["hv"].iloc[i]:
            axes[0].axvspan(high_vol_n.index[i-1], high_vol_n.index[i], alpha=0.15, color="red")

    axes[1].plot(vol_24h_n.index, vol_24h_n["vol_24h"], label="24h rolling σ", color="#ff4b4b", alpha=0.9)
    if vol_7d.notna().any():
        axes[1].plot(vol_7d_n.index, vol_7d_n["vol_7d"], label="7d rolling σ", color="#1f6feb", alpha=0.9, lw=2)
    axes[1].axhline(thresh, color="gray", ls="--", lw=1, alpha=0.7)
    axes[1].set_ylabel("Volatility (σ EUR/MWh)")
    axes[1].set_title("Rolling volatility")
    axes[1].legend()
    format_date_axis(axes[1])
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_peak_offpeak(zone: str, start: str, end: str) -> None:
    """Peak (8-20 weekdays) vs off-peak price spread over time."""
    try:
        sp = to_15min(load_spot_price(zone))
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sample = sp.loc[start:end].copy()
    if sample.empty:
        return

    sample["hour"] = sample.index.hour
    sample["dow"] = sample.index.dayofweek
    sample["is_peak"] = (sample["hour"] >= 8) & (sample["hour"] < 20) & (sample["dow"] < 5)

    peak_avg = sample.loc[sample["is_peak"], "price"].mean()
    offpeak_avg = sample.loc[~sample["is_peak"], "price"].mean()
    spread = peak_avg - offpeak_avg

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Peak avg (8-20 weekday)", f"{peak_avg:.2f} EUR/MWh")
    with c2:
        st.metric("Off-peak avg", f"{offpeak_avg:.2f} EUR/MWh")
    with c3:
        st.metric("Peak-Offpeak spread", f"{spread:.2f} EUR/MWh")

    # Daily spread over time
    daily = sample.groupby(sample.index.date).apply(
        lambda g: pd.Series({
            "peak": g.loc[g["is_peak"], "price"].mean(),
            "offpeak": g.loc[~g["is_peak"], "price"].mean(),
        })
    )
    daily["spread"] = daily["peak"] - daily["offpeak"]
    daily.index = pd.to_datetime(daily.index)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[1, 1])
    axes[0].plot(daily.index, daily["peak"], label="Peak", color="#ff4b4b", alpha=0.9)
    axes[0].plot(daily.index, daily["offpeak"], label="Off-peak", color="#1f6feb", alpha=0.9)
    axes[0].fill_between(daily.index, daily["peak"], daily["offpeak"], alpha=0.15, color="orange")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title(f"{zone}: Peak (08-20 weekday) vs Off-peak daily average")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["#238636" if v >= 0 else "#ff4b4b" for v in daily["spread"]]
    axes[1].bar(daily.index, daily["spread"], color=colors, alpha=0.8, width=0.8)
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("EUR/MWh")
    axes[1].set_title("Daily peak-offpeak spread")
    axes[1].grid(True, alpha=0.3)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
