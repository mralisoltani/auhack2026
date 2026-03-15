"""Weather plots — Notebook 05."""
import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
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


def plot_cloud_cover_solar_generation(zone: str, start: str, end: str) -> None:
    """Cloud cover vs solar generation."""
    try:
        weather = to_15min(load_weather(zone))
        gen = load_generation_pivot(zone)
        solar_cols = [c for c in gen.columns if "SOLAR" in c]
        if not solar_cols:
            st.warning(f"No solar generation data for {zone}.")
            return
        gen["solar_total"] = gen[solar_cols].sum(axis=1)
        join = weather[["cloud_cover"]].join(gen[["solar_total"]], how="inner").dropna()
        join = join.loc[start:end]
        if join.empty or len(join) < 10:
            st.warning("Insufficient data for cloud cover vs solar generation.")
            return
        corr = join["cloud_cover"].corr(join["solar_total"])
        st.metric("Correlation: cloud cover vs solar generation", f"{corr:.3f}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(join["cloud_cover"], join["solar_total"], alpha=0.2, s=5)
        ax.set_xlabel("Cloud Cover (%)")
        ax.set_ylabel("Solar Generation (MW)")
        ax.set_title(f"{zone}: Cloud Cover vs Solar Generation")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather or solar generation data for {zone}.")


def plot_humidity_load(zone: str, start: str, end: str) -> None:
    """Humidity vs load."""
    try:
        weather = to_15min(load_weather(zone))
        load = load_total_load(zone)
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather data for {zone}.")
        return
    # 1. Look for the humidity column dynamically
    # This finds the first column that contains 'humidity' (case-insensitive)
    hum_matches = [c for c in weather.columns if "humidity" in c.lower()]
    
    if not hum_matches:
        st.warning(f"No humidity column found for {zone}.")
        return
        
    h_col = hum_matches[0]
    
    # 2. Join data and clean
    df = weather[[h_col]].join(load, how="inner").dropna()
    
    if df.empty:
        st.warning("No overlapping data for plot.")
        return

    # 3. Calculate actual correlation coefficient
    corr_val = df[h_col].corr(df["load"])
    
    # 4. Visualization
    st.metric(f"Correlation: {h_col} vs Load", f"{corr_val:.3f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Using scatter with low alpha to see density
    ax.scatter(df[h_col], df["load"], alpha=0.3, s=10, color="#2ecc71")
    
    # Add a regression line to visualize the relationship
    m, b = np.polyfit(df[h_col], df["load"], 1)
    ax.plot(df[h_col], m*df[h_col] + b, color="#e74c3c", linestyle="--", label="Trend")
    
    ax.set_xlabel(h_col)
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"{zone}: Humidity vs Consumption")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    


def plot_apparent_temp_load(zone: str, start: str, end: str) -> None:
    """Calculates Steadman's Apparent Temperature and analyzes its correlation with load."""
    try:
        weather = to_15min(load_weather(zone))
        load = load_total_load(zone)
        
        # 1. Dynamic Column Lookup (The "Fuzzy" Match)
        t_col = next((c for c in weather.columns if "temp" in c.lower()), None)
        rh_col = next((c for c in weather.columns if "humidity" in c.lower()), None)
        ws_col = next((c for c in weather.columns if "wind_speed_10m" in c.lower()), None)

        if not all([t_col, rh_col, ws_col]):
            st.warning(f"Missing required weather columns for {zone} (Temp, Humidity, or Wind).")
            return

        # 2. Extract variables for the formula
        Ta = weather[t_col]
        RH = weather[rh_col]
        ws_ms = weather[ws_col] / 3.6  # Convert km/h to m/s

        # 3. Steadman Formula Implementation
        # e = Vapor Pressure (hPa)
        e = (RH / 100) * 6.105 * np.exp((17.27 * Ta) / (237.7 + Ta))
        weather["apparent_temp"] = Ta + 0.33 * e - 0.70 * ws_ms - 4.00

        # 4. Data Join and Slice
        join = weather[["apparent_temp"]].join(load, how="inner").dropna()
        join = join.loc[start:end]

        if join.empty or len(join) < 10:
            st.warning("Insufficient data for apparent temperature vs load in this time range.")
            return

        # 5. Statistical Analysis
        corr = join["apparent_temp"].corr(join["load"])
        st.metric("Correlation: Apparent Temp vs Load", f"{corr:.3f}")

        # 6. Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # We use a scatter plot with a regression line
        ax.scatter(join["apparent_temp"], join["load"], alpha=0.15, s=8, color="#3498db", label="Data Points")
        
        # Linear regression to show the "Sensitivity" of the load to temp
        m, b = np.polyfit(join["apparent_temp"], join["load"], 1)
        ax.plot(join["apparent_temp"], m*join["apparent_temp"] + b, color="#e67e22", linewidth=2, label="Trend Line")
        
        ax.set_xlabel("Apparent Temperature (°C)")
        ax.set_ylabel("Load (MW)")
        ax.set_title(f"{zone}: Apparent Temperature vs Consumption Trend")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error calculating apparent temperature for {zone}: {e}")
    


def plot_wind_generation(zone: str, start: str, end: str) -> None:
    """Wind speed vs wind generation."""
    try:
        weather = to_15min(load_weather(zone))
        gen = load_generation_pivot(zone)
    except (FileNotFoundError, KeyError):
        st.warning(f"No weather data for {zone}.")
        return
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
    
