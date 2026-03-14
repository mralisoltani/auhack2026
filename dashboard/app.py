"""
European Power Market Dashboard — AU Hack 2026
Interactive dashboard consolidating analyses from notebooks 01-08.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import (
    SPOT_ZONES,
    FLOW_ZONES,
    load_spot_price,
    load_total_load,
    load_generation_pivot,
    load_weather,
    load_flows_into,
    load_flows_out_of,
    load_all_spot_prices,
)


def _to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Expand hourly/mixed data to 15-min with ffill."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
    return df.reindex(full_idx).ffill()


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert index to timezone-naive for matplotlib compatibility."""
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


@st.cache_data
def get_flows_pivot_15min(zone: str):
    """Load flows into zone, pivot, resample to 15-min."""
    flows = load_flows_into(zone)
    pivot = flows.pivot(index="time", columns="zone", values="value (MW)")
    pivot.index = pd.to_datetime(pivot.index, utc=True)
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="15min", tz="UTC")
    pivot = pivot.reindex(full_idx).ffill()
    pivot["net_import"] = pivot.sum(axis=1)
    return pivot


@st.cache_data
def get_prices_15min():
    """Load all spot prices at 15-min."""
    prices = load_all_spot_prices()
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq="15min", tz="UTC")
    return prices.reindex(full_idx).ffill()


def plot_spot_prices(start, end):
    """Notebook 03/08: Spot prices across zones."""
    prices = get_prices_15min()
    sample = _naive_index(prices.loc[start:end])
    if sample.empty:
        st.warning("No data for selected range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    sample.plot(ax=ax, alpha=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Spot prices across zones")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_supply_demand(zone, start, end):
    """Notebook 02: Load vs Generation."""
    load = load_total_load(zone)
    gen = load_generation_pivot(zone)
    gen["total_gen"] = gen.sum(axis=1)
    l_sample = _naive_index(load.loc[start:end])
    g_sample = _naive_index(gen.loc[start:end])
    if l_sample.empty or g_sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    l_sample["load"].plot(ax=ax, label="Load", color="C0")
    g_sample["total_gen"].plot(ax=ax, label="Generation", color="C1", alpha=0.8)
    ax.set_ylabel("MW")
    ax.set_title(f"{zone}: Load vs Generation")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_flows_in(zone, start, end):
    """Notebook 04: Flows into zone."""
    try:
        pivot = get_flows_pivot_15min(zone)
    except (FileNotFoundError, KeyError) as e:
        st.warning(f"No flow data for {zone}: {e}")
        return
    zone_cols = [c for c in pivot.columns if c != "net_import"]
    sample = _naive_index(pivot.loc[start:end])
    if sample.empty:
        st.warning("No flow data for selected zone/range.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    sample[zone_cols].plot.area(ax=axes[0], stacked=True, alpha=0.8)
    axes[0].set_ylabel("MW")
    axes[0].set_title(f"Flows into {zone}")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    sample["net_import"].plot(ax=axes[1], color="C0")
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_ylabel("MW")
    axes[1].set_title("Total inflow")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_flows_out(zone, start, end):
    """Notebook 04: Flows out of zone."""
    try:
        flows_out = load_flows_out_of(zone)
    except Exception as e:
        st.warning(f"Could not load outflow data: {e}")
        return
    if flows_out.empty:
        st.warning("No outflow data for this zone.")
        return
    pivot = flows_out.pivot(index="time", columns="zone", values="value (MW)")
    pivot.index = pd.to_datetime(pivot.index, utc=True)
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="15min", tz="UTC")
    pivot = pivot.reindex(full_idx).ffill()
    pivot["total_export"] = pivot.sum(axis=1)
    sample = _naive_index(pivot.loc[start:end])
    if sample.empty:
        st.warning("No data for selected range.")
        return
    cols = [c for c in sample.columns if c != "total_export"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    sample[cols].plot.area(ax=axes[0], stacked=True, alpha=0.8)
    axes[0].set_ylabel("MW")
    axes[0].set_title(f"Flows out of {zone}")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    sample["total_export"].plot(ax=axes[1], color="C1")
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_ylabel("MW")
    axes[1].set_title("Total outflow")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_net_import_vs_price(zone, start, end):
    """Notebook 04: Net import vs spot price (normalized)."""
    pivot = get_flows_pivot_15min(zone)
    sp = load_spot_price(zone)
    sp_15 = _to_15min(sp)
    net = pivot["net_import"].reindex(sp_15.index).ffill()
    join = sp_15.join(net, how="inner").dropna()
    join = join.loc[start:end]
    if join.empty or len(join) < 10:
        st.warning("Insufficient data for correlation plot.")
        return
    corr = join["net_import"].corr(join["price"])
    st.metric("Correlation: net import vs spot price", f"{corr:.3f}")
    net_norm = (join["net_import"] - join["net_import"].mean()) / join["net_import"].std()
    price_norm = (join["price"] - join["price"].mean()) / join["price"].std()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(net_norm, price_norm, alpha=0.3, s=5)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Net import (z-score)")
    ax.set_ylabel("Spot price (z-score)")
    ax.set_title(f"{zone}: Net import vs Spot price")
    st.pyplot(fig)
    plt.close()


def plot_weather_load(zone, start, end):
    """Notebook 05: Temperature vs load."""
    try:
        weather = _to_15min(load_weather(zone))
        load = load_total_load(zone)
        join = weather[["temperature"]].join(load, how="inner").dropna()
        join = join.loc[start:end]
        if join.empty or len(join) < 10:
            st.warning("Insufficient weather/load data.")
            return
        corr = join["temperature"].corr(join["load"])
        st.metric("Correlation: temperature vs load", f"{corr:.3f}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(join["temperature"], join["load"], alpha=0.2, s=5)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Load (MW)")
        ax.set_title(f"{zone}: Temperature vs Consumption")
        st.pyplot(fig)
        plt.close()
    except FileNotFoundError:
        st.warning(f"No weather data for {zone}.")


def plot_wind_generation(zone, start, end):
    """Notebook 05: Wind speed vs wind generation."""
    try:
        weather = _to_15min(load_weather(zone))
        gen = load_generation_pivot(zone)
        wind_cols = [c for c in gen.columns if "WIND" in c]
        gen["wind_total"] = gen[wind_cols].sum(axis=1)
        join = weather[["wind_speed_10m"]].join(gen[["wind_total"]], how="inner").dropna()
        join = join.loc[start:end]
        if join.empty or len(join) < 10:
            st.warning("Insufficient data.")
            return
        corr = join["wind_speed_10m"].corr(join["wind_total"])
        st.metric("Correlation: wind speed vs wind generation", f"{corr:.3f}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(join["wind_speed_10m"], join["wind_total"], alpha=0.2, s=5)
        ax.set_xlabel("Wind speed 10m (km/h)")
        ax.set_ylabel("Wind generation (MW)")
        ax.set_title(f"{zone}: Wind vs Generation")
        st.pyplot(fig)
        plt.close()
    except FileNotFoundError:
        st.warning(f"No weather data for {zone}.")


def plot_price_by_hour(zone):
    """Notebook 03: Average spot price by hour."""
    sp = load_spot_price(zone)
    sp["hour"] = sp.index.hour
    by_hour = sp.groupby("hour")["price"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour.index, by_hour["mean"], yerr=by_hour["std"], capsize=2, alpha=0.8)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Mean spot price (EUR/MWh)")
    ax.set_title(f"{zone}: Average spot price by hour")
    st.pyplot(fig)
    plt.close()


def plot_correlation_matrix():
    """Notebook 06: Spot price correlation matrix."""
    prices = get_prices_15min()
    corr = prices.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, label="Correlation")
    plt.title("Spot price correlation across zones")
    st.pyplot(fig)
    plt.close()


def plot_price_spreads(start, end):
    """Notebook 06: DE vs neighbor price spreads."""
    prices = get_prices_15min()
    if "DE" not in prices.columns:
        st.warning("DE not in price data.")
        return
    neighbors = [z for z in prices.columns if z != "DE"]
    spreads = pd.DataFrame({f"DE-{z}": prices["DE"] - prices[z] for z in neighbors})
    sample = spreads.loc[start:end]
    if sample.empty:
        st.warning("No data for selected range.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    sample_naive = sample.copy()
    if sample_naive.index.tz is not None:
        sample_naive.index = sample_naive.index.tz_localize(None)
    for col in sample_naive.columns:
        axes[0].plot(sample_naive.index, sample_naive[col], label=col, alpha=0.9, linewidth=1)
    axes[0].axhline(0, color="gray", ls="--")
    axes[0].set_ylim(-80, 80)
    axes[0].set_ylabel("Price spread (EUR/MWh)")
    axes[0].set_title("DE minus neighbor price")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    spreads.abs().mean().sort_values(ascending=False).plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_ylabel("Mean |spread| (EUR/MWh)")
    axes[1].set_title("Average absolute price spread")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def main():
    st.set_page_config(page_title="European Power Market", page_icon="⚡", layout="wide")
    st.title("⚡ European Power Market Dashboard")
    st.caption("AU Hack 2026 — InCommodities case: Decode the power market")

    with st.sidebar:
        st.header("Controls")
        zone = st.selectbox("Zone", SPOT_ZONES, index=SPOT_ZONES.index("DE") if "DE" in SPOT_ZONES else 0)
        start = st.date_input("Start date", pd.Timestamp("2024-07-01"))
        end = st.date_input("End date", pd.Timestamp("2024-07-14"))
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Spot Prices", "Supply & Demand", "Flows", "Weather", "Market Coupling"
    ])

    with tab1:
        st.subheader("Spot prices across zones")
        plot_spot_prices(start_str, end_str)
        st.subheader("Price by hour (all-time)")
        plot_price_by_hour(zone)

    with tab2:
        st.subheader("Load vs Generation")
        plot_supply_demand(zone, start_str, end_str)

    with tab3:
        st.subheader("Flows into zone")
        plot_flows_in(zone, start_str, end_str)
        st.subheader("Flows out of zone")
        plot_flows_out(zone, start_str, end_str)
        st.subheader("Net import vs Spot price")
        plot_net_import_vs_price(zone, start_str, end_str)

    with tab4:
        st.subheader("Temperature vs Load")
        plot_weather_load(zone, start_str, end_str)
        st.subheader("Wind vs Wind Generation")
        plot_wind_generation(zone, start_str, end_str)

    with tab5:
        st.subheader("Price correlation matrix")
        plot_correlation_matrix()
        st.subheader("DE vs neighbor price spreads")
        plot_price_spreads(start_str, end_str)


if __name__ == "__main__":
    main()
