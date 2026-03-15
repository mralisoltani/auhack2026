"""Supply & demand plots — Notebook 02."""
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import naive_index, format_date_axis, tight_layout, to_15min
from src.data_loader import load_total_load, load_generation_pivot, load_spot_price


def plot_renewable_penetration(zone: str, start: str, end: str) -> None:
    """Renewable Penetration (Wind + Solar / Total Generation)."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning(f"No load/generation data for {zone}.")
        return

    # Identify wind and solar columns
    wind_cols = [c for c in gen.columns if "WIND" in c]
    solar_cols = [c for c in gen.columns if "SOLAR" in c]
    renewable_cols = [c for c in gen.columns if "RENEWABLE" in c]
    gen["renewables"] = gen[wind_cols].sum(axis=1) + gen[solar_cols].sum(axis=1) + gen[renewable_cols].sum(axis=1)
    total_gen = gen.sum(axis=1)
    # Combine load and generation for alignment and calculate penetration percentage
    combined = load.join(gen[['renewables']], how='inner').dropna()
    combined['penetration'] = (combined['renewables'] / total_gen) * 100

    sample = naive_index(combined.loc[start:end])
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return
        
    fig, ax = plt.subplots(figsize=(12, 4))
    sample["penetration"].plot(ax=ax, label="Renewable Share", color="C3")
    ax.set_ylabel("Penetration (%)")
    ax.set_title(f"{zone}: Renewable Share (Wind + Solar / Total Generation)")
    ax.legend()
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_penetration_vs_price(zone: str, start: str, end: str, figsize: tuple = (8, 5)) -> None:
    """Renewable penetration vs spot price (scatter)."""
    try:
        gen = load_generation_pivot(zone)
        sp = load_spot_price(zone)
        
        # Calculate penetration
        wind_cols = [c for c in gen.columns if "WIND" in c]
        solar_cols = [c for c in gen.columns if "SOLAR" in c]
        renewable_cols = [c for c in gen.columns if "RENEWABLE" in c]
        
        total_gen = gen.sum(axis=1)
        renewables = gen[wind_cols].sum(axis=1) + gen[solar_cols].sum(axis=1) + gen[renewable_cols].sum(axis=1)
        gen['penetration'] = (renewables / total_gen) * 100
        
        # Align 15-minute generation data with hourly/spot prices
        sp_15 = to_15min(sp)
        join = gen[['penetration']].join(sp_15, how="inner").dropna()
        join = join.loc[start:end]
        
        if join.empty or len(join) < 10:
            st.warning("Insufficient data for penetration vs price correlation.")
            return
            
        corr = join["penetration"].corr(join["price"])
        st.metric("Correlation: penetration vs spot price", f"{corr:.3f}")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(join["penetration"], join["price"], alpha=0.3, s=5)
        ax.set_xlabel("Renewable Penetration (%)")
        ax.set_ylabel("Spot Price (EUR/MWh)")
        ax.set_title(f"{zone}: Renewable Share vs Spot Price")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not plot penetration vs price: {e}")


def plot_fossil_ratio_vs_price(zone: str, start: str, end: str, figsize: tuple = (8, 5)) -> None:
    """Fossil fuel ratio vs spot price (scatter)."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
        sp = load_spot_price(zone)
        
        # Calculate fossil generation
        fossil_keywords = ["COAL", "LIGNITE", "GAS", "OIL", "FOSSIL"]
        fossil_cols = [c for c in gen.columns if any(k in str(c).upper() for k in fossil_keywords)]
        gen["fossil_total"] = gen[fossil_cols].sum(axis=1)
        total_gen = gen.sum(axis=1)
        combined = load.join(gen[['fossil_total']], how='inner').dropna()
        combined['fossil_ratio'] = (combined['fossil_total'] / total_gen) * 100
        
        # Align 15-minute generation data with hourly/spot prices
        sp_15 = to_15min(sp)
        join = combined[['fossil_ratio']].join(sp_15, how="inner").dropna()
        join = join.loc[start:end]
        
        if join.empty or len(join) < 10:
            st.warning("Insufficient data for fossil ratio vs price correlation.")
            return
            
        corr = join["fossil_ratio"].corr(join["price"])
        st.metric("Correlation: fossil ratio vs spot price", f"{corr:.3f}")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(join["fossil_ratio"], join["price"], alpha=0.3, s=5, color="#8c564b")
        ax.set_xlabel("Fossil Fuel Ratio (%)")
        ax.set_ylabel("Spot Price (EUR/MWh)")
        ax.set_title(f"{zone}: Fossil Fuel Ratio vs Spot Price")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not plot fossil ratio vs price: {e}")


def plot_supply_demand(zone: str, start: str, end: str) -> None:
    """Load vs Generation."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning(f"No load/generation data for {zone}.")
        return
    gen["total_gen"] = gen.sum(axis=1)
    l_sample = naive_index(load.loc[start:end])
    g_sample = naive_index(gen.loc[start:end])
    if l_sample.empty or g_sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    l_sample["load"].plot(ax=ax, label="Load", color="C0")
    g_sample["total_gen"].plot(ax=ax, label="Generation", color="C1", alpha=0.8)
    ax.set_ylabel("MW")
    ax.set_title(f"{zone}: Load vs Generation")
    ax.legend()
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()

def plot_residual_load(zone: str, start: str, end: str) -> None:
    """Load vs Generation and Residual Load."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning(f"No load/generation data for {zone}.")
        return
    gen["total_gen"] = gen.sum(axis=1)
    l_sample = naive_index(load.loc[start:end])
    g_sample = naive_index(gen.loc[start:end])
    if l_sample.empty or g_sample.empty:
        st.warning("No data for selected zone/range.")
        return

    # Combine load and generation for alignment and calculate residual
    combined = load.join(gen[['total_gen']], how='inner').dropna()
    combined['residual_load'] = combined['load'] - combined['total_gen']
    sample = naive_index(combined.loc[start:end])
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))      
    sample["residual_load"].plot(ax=ax, label="Residual Load", color="C2", alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel("MW")    
    ax.set_title(f"{zone}: Residual Load")
    ax.legend()
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()

def plot_generation_mix(zone: str, start: str, end: str) -> None:
    """Stacked area chart of generation mix."""
    try:
        gen = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning(f"No generation data for {zone}.")
        return   
    
    sample = naive_index(gen.loc[start:end])
  
    
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return

    # Clean data: Replace NaNs with 0 and remove negative values so stacking works correctly
    sample = sample.fillna(0).clip(lower=0)

    # Order columns by marginal cost (cheapest to most expensive)
    def get_sort_key(label):
        lbl = str(label).upper()
        if any(x in lbl for x in ["WIND", "SOLAR", "HYDRO", "RENEWABLE"]): return 1
        if any(x in lbl for x in ["NUCLEAR", "BIOMASS", "WASTE", "GEOTHERMAL", "OTHER"]): return 2
        if any(x in lbl for x in ["LIGNITE", "COAL", "IGNITE"]): return 3
        if any(x in lbl for x in ["GAS", "OIL", "FOSSIL"]): return 5
        return 4

    sorted_cols = sorted(sample.columns, key=get_sort_key)
    sample = sample[sorted_cols]

    # Generalize label mapping (order matters: more specific first)
    color_mapping = {
        "OTHER-RENEWABLE": "#c7e9c0",
        "STORAGE": "#286c9c",
        "WIND": "#7bddfeff",
        "SOLAR": "#eeff5aaa",
        "HYDRO": "#3569a985",
        "NUCLEAR": "#9467bd",
        "BIOMASS": "#c5b0d5",
        "WASTE": "#7f7f7f",
        "LIGNITE": "#8c564b",
        "IGNITE": "#8c564b",
        "COAL": "#543005",
        "GAS": "#ff7f0e",
        "OIL": "#d62728",
        "OTHER": "#bcbddc",
    }

    def get_color(label):
        for k, v in color_mapping.items():
            if k in str(label).upper():
                return v
        return f"#{random.randint(0, 0xFFFFFF):06x}"

    colors = [get_color(c) for c in sample.columns]

    fig, ax = plt.subplots(figsize=(12, 6))      
    sample.plot.area(ax=ax, stacked=True, alpha=0.8, color=colors)
    
   
    ax.set_ylabel("Generation (MW)")    
    ax.set_title(f"{zone}: Energy Generation Distribution")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
 
    # 2. Reverse Legend to match Stack
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.02, 1), loc="upper left")

    # 3. Add a "Zero Line" emphasis
    ax.axhline(0, color='black', linewidth=0.8)

    # 4. Use a more professional font and grid
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_duck_curve(zone: str, start: str, end: str) -> None:
    """Solar duck curve: intraday residual load shape on high-solar vs low-solar days."""
    try:
        load_df = load_total_load(zone)
        gen = load_generation_pivot(zone)
    except FileNotFoundError:
        st.warning(f"No load/generation data for {zone}.")
        return

    solar_cols = [c for c in gen.columns if "SOLAR" in c]
    if not solar_cols:
        st.info(f"No solar generation data for {zone}. Duck curve requires solar.")
        return

    gen["solar_total"] = gen[solar_cols].sum(axis=1)
    gen["total_gen"] = gen.sum(axis=1)
    combined = load_df.join(gen[["solar_total", "total_gen"]], how="inner").dropna()
    combined["residual_load"] = combined["load"] - combined["total_gen"]
    sample = combined.loc[start:end]
    if sample.empty or len(sample) < 96:
        st.warning("Need at least 1 day of data for duck curve analysis.")
        return

    # Daily solar production
    sample["date"] = sample.index.date
    sample["hour"] = sample.index.hour + sample.index.minute / 60
    daily_solar = sample.groupby("date")["solar_total"].sum()

    # Split into high vs low solar days (above/below median)
    median_solar = daily_solar.median()
    high_solar_days = set(daily_solar[daily_solar >= daily_solar.quantile(0.75)].index)
    low_solar_days = set(daily_solar[daily_solar <= daily_solar.quantile(0.25)].index)

    high = sample[sample["date"].isin(high_solar_days)]
    low = sample[sample["date"].isin(low_solar_days)]

    if high.empty or low.empty:
        st.warning("Not enough variation in solar production for duck curve.")
        return

    high_profile = high.groupby("hour")["residual_load"].mean()
    low_profile = low.groupby("hour")["residual_load"].mean()
    high_load = high.groupby("hour")["load"].mean()
    low_load = low.groupby("hour")["load"].mean()

    c1, c2 = st.columns(2)
    with c1:
        belly_depth = high_profile.min() - low_profile.min()
        st.metric("Duck belly depth", f"{belly_depth:.0f} MW")
    with c2:
        ramp = high_profile.iloc[-1] - high_profile.min() if len(high_profile) > 1 else 0
        st.metric("Evening ramp (high solar)", f"{ramp:.0f} MW")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(high_profile.index, high_profile.values, color="#ff7f0e", lw=2.5, label="High solar days (Q4)")
    axes[0].plot(low_profile.index, low_profile.values, color="#1f77b4", lw=2.5, label="Low solar days (Q1)")
    axes[0].fill_between(high_profile.index, high_profile.values, low_profile.values, alpha=0.15, color="orange")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual load (MW)")
    axes[0].set_title(f"{zone}: Duck curve — Residual load by hour")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(high_load.index, high_load.values, color="#ff7f0e", lw=2, ls="--", label="Load (high solar)")
    axes[1].plot(low_load.index, low_load.values, color="#1f77b4", lw=2, ls="--", label="Load (low solar)")
    high_solar_profile = high.groupby("hour")["solar_total"].mean()
    axes[1].fill_between(high_solar_profile.index, 0, high_solar_profile.values, alpha=0.3, color="#eeff5a", label="Solar gen (high days)")
    axes[1].set_xlabel("Hour of day (UTC)")
    axes[1].set_ylabel("MW")
    axes[1].set_title(f"{zone}: Load profile & solar generation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
    st.caption("The 'duck curve' shows how solar production creates a midday dip in residual load, followed by a steep evening ramp when solar drops off.")
