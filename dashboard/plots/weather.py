"""Weather plots — Notebook 05."""
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import to_15min, tight_layout
from src.data_loader import load_weather, load_total_load, load_generation_pivot


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
