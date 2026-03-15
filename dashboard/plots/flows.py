"""Flow plots — Notebook 04."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dashboard.utils import get_flows_pivot_15min, get_prices_15min, to_15min, naive_index, format_date_axis, tight_layout
from src.data_loader import load_spot_price, load_flows_out_of, load_total_load, load_generation_pivot
from src.zone_registry import get_flow_zones, get_spot_zones


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


def plot_congestion(zone: str, start: str, end: str) -> None:
    """Detect congestion: flow flatlines at capacity limits → prices decouple."""
    try:
        pivot = get_flows_pivot_15min(zone, tuple(get_flow_zones()))
    except (FileNotFoundError, KeyError):
        st.warning(f"No flow data for {zone}.")
        return
    link_cols = [c for c in pivot.columns if c != "net_import"]
    sample = pivot.loc[start:end]
    if sample.empty or not link_cols:
        st.warning("No flow data for selected range.")
        return

    # Detect congestion: rolling std near zero means flatline at capacity
    window = 16  # 4 hours of 15-min data
    congestion_data = {}
    for link in link_cols:
        series = sample[link].dropna()
        if len(series) < window:
            continue
        rolling_std = series.rolling(window, min_periods=window).std()
        # Congested when std is very low relative to overall std AND flow is near min/max
        overall_std = series.std()
        if overall_std == 0:
            continue
        threshold = overall_std * 0.02
        is_congested = rolling_std < threshold
        pct = is_congested.mean() * 100
        max_flow = series.max()
        min_flow = series.min()
        congestion_data[link] = {
            "congested_pct": pct,
            "max_flow": max_flow,
            "min_flow": min_flow,
            "is_congested": is_congested,
            "series": series,
        }

    if not congestion_data:
        st.info("Insufficient data for congestion detection.")
        return

    # Summary table
    summary = pd.DataFrame({
        link: {"Congestion %": f"{d['congested_pct']:.1f}%", "Max flow (MW)": f"{d['max_flow']:.0f}",
               "Min flow (MW)": f"{d['min_flow']:.0f}"}
        for link, d in congestion_data.items()
    }).T
    st.dataframe(summary, use_container_width=True)

    # Plot top congested links
    sorted_links = sorted(congestion_data.keys(), key=lambda k: congestion_data[k]["congested_pct"], reverse=True)
    top = sorted_links[:min(4, len(sorted_links))]
    n_plots = len(top)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for ax, link in zip(axes, top):
        d = congestion_data[link]
        s = naive_index(d["series"].to_frame("flow"))
        cong = naive_index(d["is_congested"].to_frame("cong"))
        ax.plot(s.index, s["flow"], lw=0.8, color="C0", alpha=0.8)
        for i in range(1, len(cong)):
            if cong["cong"].iloc[i]:
                ax.axvspan(cong.index[i-1], cong.index[i], alpha=0.3, color="red")
        ax.set_ylabel("MW")
        ax.set_title(f"{link} → {zone} (congestion: {d['congested_pct']:.1f}% of period)")
    format_date_axis(axes[-1])
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
    st.caption("Red bands = potential congestion (flow flatlines near capacity). When congested, prices between zones decouple.")


def plot_flow_drivers(zone: str, start: str, end: str) -> None:
    """Decompose what drives net flow: price spread, wind, demand differential."""
    try:
        pivot = get_flows_pivot_15min(zone, tuple(get_flow_zones()))
        sp = to_15min(load_spot_price(zone))
    except FileNotFoundError:
        st.warning(f"No flow or price data for {zone}.")
        return

    net = pivot["net_import"].reindex(sp.index).ffill()
    joined = sp.join(net, how="inner").dropna().loc[start:end]
    if joined.empty or len(joined) < 20:
        st.warning("Insufficient data for flow driver analysis.")
        return

    # Build driver features
    drivers = pd.DataFrame(index=joined.index)
    drivers["net_import"] = joined["net_import"]

    # Price spread vs neighbors
    all_prices = get_prices_15min(tuple(get_spot_zones()))
    neighbors = [c for c in all_prices.columns if c != zone]
    if neighbors:
        neighbor_avg = all_prices[neighbors].mean(axis=1).reindex(drivers.index).ffill()
        drivers["price_spread"] = joined["price"] - neighbor_avg

    # Wind generation
    try:
        gen = load_generation_pivot(zone)
        wind_cols = [c for c in gen.columns if "WIND" in c]
        if wind_cols:
            drivers["wind_gen"] = gen[wind_cols].sum(axis=1).reindex(drivers.index, method="ffill")
    except FileNotFoundError:
        pass

    # Load
    try:
        load_df = load_total_load(zone)
        drivers["load"] = load_df["load"].reindex(drivers.index, method="ffill")
    except FileNotFoundError:
        pass

    # Renewable share
    try:
        gen = load_generation_pivot(zone)
        ren_kw = ["SOLAR", "WIND", "HYDRO", "BIOMASS", "RENEWABLE"]
        ren_cols = [c for c in gen.columns if any(k in c for k in ren_kw)]
        total_gen = gen.sum(axis=1)
        ren_gen = gen[ren_cols].sum(axis=1) if ren_cols else pd.Series(0, index=gen.index)
        drivers["renewable_share"] = (ren_gen / total_gen.replace(0, np.nan)).reindex(drivers.index, method="ffill")
    except FileNotFoundError:
        pass

    drivers = drivers.dropna(how="all", axis=1).dropna()
    feat_cols = [c for c in drivers.columns if c != "net_import"]
    if not feat_cols:
        st.warning("No driver features available.")
        return

    # Correlations
    corrs = drivers[feat_cols].corrwith(drivers["net_import"]).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#238636" if v >= 0 else "#ff4b4b" for v in corrs]
    axes[0].barh(corrs.index, corrs.values, color=colors, alpha=0.8)
    axes[0].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Correlation with net import")
    axes[0].set_title(f"{zone}: What drives net flow?")
    axes[0].grid(True, alpha=0.3)

    # Scatter: top driver vs net import
    top_driver = corrs.abs().idxmax()
    axes[1].scatter(drivers[top_driver], drivers["net_import"], alpha=0.2, s=5, color="C0")
    axes[1].set_xlabel(top_driver)
    axes[1].set_ylabel("Net import (MW)")
    axes[1].set_title(f"Top driver: {top_driver} (r={corrs[top_driver]:.3f})")
    axes[1].grid(True, alpha=0.3)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
