"""Weather plots — Notebook 05."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import to_15min, naive_index, format_date_axis, tight_layout
from src.data_loader import load_weather, load_total_load, load_generation_pivot, load_spot_price


def plot_weather_load(zone: str, start: str, end: str) -> None:
    """Temperature vs load."""
    try:
        weather = to_15min(load_weather(zone))
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
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather data for {zone}.")


def plot_wind_generation(zone: str, start: str, end: str) -> None:
    """Wind speed vs wind generation."""
    try:
        weather = to_15min(load_weather(zone))
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
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather data for {zone}.")


def plot_cloud_solar(zone: str, start: str, end: str) -> None:
    """Cloud cover vs solar generation."""
    try:
        weather = to_15min(load_weather(zone))
        gen = load_generation_pivot(zone)
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather or generation data for {zone}.")
        return

    solar_cols = [c for c in gen.columns if "SOLAR" in c]
    if not solar_cols:
        st.info(f"No solar generation data for {zone}.")
        return
    if "cloud_cover" not in weather.columns:
        st.info("No cloud cover data available.")
        return

    gen["solar_total"] = gen[solar_cols].sum(axis=1)
    join = weather[["cloud_cover"]].join(gen[["solar_total"]], how="inner").dropna()
    join = join.loc[start:end]
    join = join[join.index.hour.isin(range(6, 21))]
    if join.empty or len(join) < 10:
        st.warning("Insufficient data.")
        return

    corr = join["cloud_cover"].corr(join["solar_total"])
    st.metric("Correlation: cloud cover vs solar generation (daytime)", f"{corr:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(join["cloud_cover"], join["solar_total"], alpha=0.2, s=5, color="#ff7f0e")
    axes[0].set_xlabel("Cloud cover (%)")
    axes[0].set_ylabel("Solar generation (MW)")
    axes[0].set_title(f"{zone}: Cloud cover vs Solar (daytime)")
    axes[0].grid(True, alpha=0.3)

    bins = pd.cut(join["cloud_cover"], bins=10)
    binned = join.groupby(bins, observed=True)["solar_total"].mean()
    binned.plot(kind="bar", ax=axes[1], color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("Cloud cover range (%)")
    axes[1].set_ylabel("Avg solar generation (MW)")
    axes[1].set_title("Solar output by cloud cover band")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_weather_price_impact(zone: str, start: str, end: str) -> None:
    """Temperature extremes vs load spikes and price impact."""
    try:
        weather = to_15min(load_weather(zone))
        load_df = load_total_load(zone)
        sp = to_15min(load_spot_price(zone))
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather, load, or price data for {zone}.")
        return

    if "temperature" not in weather.columns:
        st.warning("No temperature data available.")
        return

    join = weather[["temperature"]].join(load_df, how="inner").join(sp, how="inner").dropna()
    join = join.loc[start:end]
    if join.empty or len(join) < 50:
        st.warning("Insufficient data.")
        return

    q10 = join["temperature"].quantile(0.1)
    q90 = join["temperature"].quantile(0.9)
    join["regime"] = "Normal"
    join.loc[join["temperature"] <= q10, "regime"] = "Cold extreme"
    join.loc[join["temperature"] >= q90, "regime"] = "Hot extreme"

    regime_stats = join.groupby("regime").agg(
        avg_load=("load", "mean"),
        avg_price=("price", "mean"),
        count=("price", "count"),
    )

    c1, c2, c3 = st.columns(3)
    for i, (regime, row) in enumerate(regime_stats.iterrows()):
        col = [c1, c2, c3][i % 3]
        with col:
            st.metric(f"{regime}", f"Load: {row['avg_load']:.0f} MW | Price: {row['avg_price']:.1f} EUR")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Cold extreme": "#1f77b4", "Normal": "#7f7f7f", "Hot extreme": "#d62728"}
    for regime, color in colors.items():
        mask = join["regime"] == regime
        if mask.any():
            axes[0].scatter(join.loc[mask, "temperature"], join.loc[mask, "load"],
                          alpha=0.3, s=5, color=color, label=regime)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_title(f"{zone}: Temperature extremes vs Load")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for regime, color in colors.items():
        mask = join["regime"] == regime
        if mask.any():
            axes[1].scatter(join.loc[mask, "temperature"], join.loc[mask, "price"],
                          alpha=0.3, s=5, color=color, label=regime)
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Spot price (EUR/MWh)")
    axes[1].set_title(f"{zone}: Temperature extremes vs Price")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
    st.caption("Cold/hot extremes = bottom/top 10% of temperature. These drive load spikes (heating/cooling) and price surges.")
