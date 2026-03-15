"""Supply & demand plots — Notebook 02."""
import random
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from dashboard.utils import get_flows_pivot_15min, naive_index, format_date_axis, tight_layout, to_15min
from src.data_loader import load_total_load, load_generation_pivot, load_spot_price, load_flows_out_of
from src.data_loader import load_spot_price, load_flows_out_of


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


def plot_penetration_vs_price(zone: str, start: str, end: str) -> None:
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
            st.warning("Insufficient data for renewable share vs price correlation.")
            return
            
        corr = join["penetration"].corr(join["price"])
        st.metric("Correlation: renewable share vs spot price", f"{corr:.3f}")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(join["penetration"], join["price"], alpha=0.3, s=5)
        ax.set_xlabel("Renewable Share (%)")
        ax.set_ylabel("Spot Price (EUR/MWh)")
        ax.set_title(f"{zone}: Renewable Share vs Spot Price")
        tight_layout(fig)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not plot renewable share vs price: {e}")


def plot_fossil_ratio_vs_price(zone: str, start: str, end: str) -> None:
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
        
        fig, ax = plt.subplots(figsize=(8, 5))
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
    """Residual Load and Net Flow."""
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
    
    # Calculate net flow (Imports - Exports)
    try:
        pivot_in = get_flows_pivot_15min(zone)
        total_import = pivot_in["net_import"]
    except Exception:
        total_import = pd.Series(0, index=combined.index)
        
    try:
        flows_out = load_flows_out_of(zone)
        if not flows_out.empty:
            pivot_out = flows_out.pivot(index="time", columns="zone", values="value (MW)")
            pivot_out.index = pd.to_datetime(pivot_out.index, utc=True)
            pivot_out = to_15min(pivot_out)
            total_export = pivot_out.sum(axis=1)
        else:
            total_export = pd.Series(0, index=combined.index)
    except Exception:
        total_export = pd.Series(0, index=combined.index)
        
    combined = combined.join(total_import.rename("total_import"), how="left")
    combined = combined.join(total_export.rename("total_export"), how="left")
    combined.fillna(0, inplace=True)
    combined['net_flow'] = combined['total_import'] - combined['total_export']

    sample = naive_index(combined.loc[start:end])
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))      
    sample["residual_load"].plot(ax=ax, label="Residual Load", color="C2", alpha=0.8)
    sample["net_flow"].plot(ax=ax, label="Net Flow (Imports - Exports)", color="C4", alpha=0.8, linestyle="--")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel("MW")    
    ax.set_title(f"{zone}: Residual Load & Net Flow")
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
