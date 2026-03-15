"""Flow plots — Notebook 04."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dashboard.utils import get_flows_pivot_15min, to_15min, naive_index, format_date_axis, tight_layout
from src.data_loader import load_spot_price, load_flows_out_of
from src.zone_registry import get_flow_zones


def plot_flows_in(zone: str, start: str, end: str) -> None:
    """Flows into zone."""
    try:
        pivot = get_flows_pivot_15min(zone, tuple(get_flow_zones()))
    except (FileNotFoundError, KeyError) as e:
        st.warning(f"No flow data for {zone}: {e}")
        return
    zone_cols = [c for c in pivot.columns if c != "net_import"]
    sample = naive_index(pivot.loc[start:end])
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
    for ax in axes:
        format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_flows_out(zone: str, start: str, end: str) -> None:
    """Flows out of zone."""
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
    sample = naive_index(pivot.loc[start:end])
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
    for ax in axes:
        format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_net_import_vs_price(zone: str, start: str, end: str) -> None:
    """Net import vs spot price (normalized scatter)."""
    try:
        pivot = get_flows_pivot_15min(zone, tuple(get_flow_zones()))
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No flow or spot price data for {zone}.")
        return
    sp_15 = to_15min(sp)
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
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()




def plot_net_import_vs_price_2(zone: str, start: str, end: str) -> None:
    """Action-oriented dashboard for energy traders."""
    try:
        pivot = get_flows_pivot_15min(zone, tuple(get_flow_zones()))
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No flow or spot price data for {zone}.")
        return
    join = to_15min(sp).join(pivot["net_import"], how="inner").dropna().loc[start:end]
    net_z = (join["net_import"] - join["net_import"].mean()) / join["net_import"].std()
    price_z = (join["price"] - join["price"].mean()) / join["price"].std()
    def get_action(pz, nz):
        if pz > 1 and nz > 1: return ('SHORTAGE', '#ff4b4b')  
        if pz < -1 and nz < -1: return ('SURPLUS', '#238636') 
        if pz > 1 and nz < -1: return ('EXPORTER GOLD', '#d4a017') 
        if pz < -1 and nz > 1: return ('ARBITRAGE', '#1f6feb') 
        return ('NEUTRAL', '#8b949e') # Gray: Hold
    actions = [get_action(p, n) for p, n in zip(price_z, net_z)]
    fig, (ax1, ax_strat) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
    ax1.plot(join.index, join['price'], color='#ff4b4b', label='Spot Price (€)', lw=2)
    ax2 = ax1.twinx()
    ax2.fill_between(join.index, join['net_import'], color='#1f6feb', alpha=0.2, label='Net Import (MW)')
    ax1.set_ylabel("Price (€/MWh)", color='#ff4b4b', fontweight='bold')
    ax2.set_ylabel("Net Import (MW)", color='#1f6feb', fontweight='bold')
    ax1.set_title(f"COMMAND CENTER: {zone}", fontsize=16, fontweight='bold')

    for i in range(len(join)-1):
        ax_strat.axvspan(join.index[i], join.index[i+1], color=actions[i][1], alpha=0.9)
    ax_strat.set_yticks([])
    ax_strat.set_xlabel("Time")
    ax_strat.set_ylabel("ACTION", fontweight='bold')
    labels = [mpatches.Patch(color=c, label=l) for l, c in 
              dict(zip([a[0] for a in actions], [a[1] for a in actions])).items()]
    ax_strat.legend(handles=labels, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
