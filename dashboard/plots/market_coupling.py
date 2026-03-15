"""Market coupling plots — Notebook 06."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import get_prices_15min, naive_index, format_date_axis, tight_layout
from src.zone_registry import get_spot_zones


def plot_correlation_matrix(zone: str) -> None:
    """Bar chart of spot price correlations between selected zone and all others, sorted by strength."""
    prices = get_prices_15min(tuple(get_spot_zones()))
    if zone not in prices.columns:
        st.warning(f"{zone} not in price data.")
        return
    corr = prices.corr()
    # Correlations of selected zone with all others (exclude self)
    zone_corr = corr[zone].drop(zone, errors="ignore").sort_values(ascending=False)
    if zone_corr.empty:
        st.warning("No other zones to correlate with.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(zone_corr) * 0.4)))
    colors = ["#238636" if v >= 0 else "#ff4b4b" for v in zone_corr.values]
    bars = ax.barh(zone_corr.index, zone_corr.values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Correlation with spot price")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"{zone}: Price correlation with other zones")
    ax.grid(True, axis="x", alpha=0.3)
    for i, (idx, val) in enumerate(zone_corr.items()):
        ax.text(val + (0.02 if val >= 0 else -0.02), i, f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9, fontweight="medium")
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_price_spreads(zone: str, start: str, end: str) -> None:
    """Zone vs neighbor price spreads."""
    prices = get_prices_15min(tuple(get_spot_zones()))
    if zone not in prices.columns:
        st.warning(f"{zone} not in price data.")
        return
    neighbors = [z for z in prices.columns if z != zone]
    spreads = pd.DataFrame({f"{zone}-{z}": prices[zone] - prices[z] for z in neighbors})
    sample = spreads.loc[start:end]
    if sample.empty:
        st.warning("No data for selected range.")
        return
    sample_naive = naive_index(sample)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    for col in sample_naive.columns:
        axes[0].plot(sample_naive.index, sample_naive[col], label=col, alpha=0.9, linewidth=1)
    axes[0].axhline(0, color="gray", ls="--")
    # axes[0].set_ylim(-80, 80)
    axes[0].set_ylabel("Price spread (EUR/MWh)")
    axes[0].set_title(f"{zone} minus neighbor price")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    format_date_axis(axes[0])
    spreads.abs().mean().sort_values(ascending=False).plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_ylabel("Mean |spread| (EUR/MWh)")
    axes[1].set_title("Average absolute price spread")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_coupling_events(zone: str, start: str, end: str) -> None:
    """Detect when prices converge (coupled) vs diverge (decoupled) between zone pairs."""
    prices = get_prices_15min(tuple(get_spot_zones()))
    if zone not in prices.columns:
        st.warning(f"{zone} not in price data.")
        return
    neighbors = [z for z in prices.columns if z != zone]
    if not neighbors:
        return
    sample = prices.loc[start:end]
    if sample.empty:
        return

    coupling_threshold = st.slider(
        "Coupling threshold (EUR/MWh)", 0.5, 20.0, 2.0, 0.5,
        help="Price spread below this = coupled (prices converge). Above = decoupled (congestion).",
        key="coupling_thresh",
    )

    # Compute coupling % for each neighbor
    coupling_stats = {}
    for nb in neighbors:
        spread = (sample[zone] - sample[nb]).abs()
        coupled_pct = (spread <= coupling_threshold).mean() * 100
        avg_spread = spread.mean()
        coupling_stats[nb] = {"coupled_pct": coupled_pct, "avg_spread": avg_spread}

    stats_df = pd.DataFrame(coupling_stats).T.sort_values("coupled_pct", ascending=False)
    stats_df.columns = ["Coupled %", "Avg |spread| (EUR/MWh)"]
    stats_df["Coupled %"] = stats_df["Coupled %"].map(lambda x: f"{x:.1f}%")
    stats_df["Avg |spread| (EUR/MWh)"] = stats_df["Avg |spread| (EUR/MWh)"].map(lambda x: f"{x:.2f}")
    st.dataframe(stats_df, use_container_width=True)

    # Top 3 most coupled + least coupled neighbors
    sorted_nbs = sorted(coupling_stats.keys(), key=lambda k: coupling_stats[k]["coupled_pct"], reverse=True)
    show_nbs = sorted_nbs[:min(4, len(sorted_nbs))]

    fig, axes = plt.subplots(len(show_nbs), 1, figsize=(14, 3 * len(show_nbs)), sharex=True)
    if len(show_nbs) == 1:
        axes = [axes]

    for ax, nb in zip(axes, show_nbs):
        spread = sample[zone] - sample[nb]
        spread_n = naive_index(spread.to_frame("spread"))
        coupled = spread.abs() <= coupling_threshold
        coupled_n = naive_index(coupled.to_frame("c"))

        ax.plot(spread_n.index, spread_n["spread"], lw=0.6, color="C0", alpha=0.7)
        ax.axhline(coupling_threshold, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(-coupling_threshold, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", ls="-", lw=0.5)
        # Shade coupled periods
        for i in range(1, len(coupled_n)):
            if coupled_n["c"].iloc[i]:
                ax.axvspan(coupled_n.index[i-1], coupled_n.index[i], alpha=0.2, color="#238636")
        cpct = coupling_stats[nb]["coupled_pct"]
        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"{zone}–{nb}: price spread (coupled {cpct:.0f}% of time)")

    format_date_axis(axes[-1])
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
    st.caption("Green bands = coupled (spread within threshold). White = decoupled (congestion, price divergence).")

    # Rolling coupling ratio over time
    st.subheader("Rolling coupling ratio (7-day window)")
    window = 672  # 7d x 96 intervals
    if len(sample) >= window:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        for nb in show_nbs:
            spread = (sample[zone] - sample[nb]).abs()
            rolling_coupled = (spread <= coupling_threshold).rolling(window, min_periods=window//2).mean() * 100
            rc_n = naive_index(rolling_coupled.to_frame(nb))
            ax2.plot(rc_n.index, rc_n[nb], label=f"{zone}–{nb}", alpha=0.8)
        ax2.axhline(50, color="gray", ls="--", lw=1, alpha=0.5)
        ax2.set_ylabel("Coupled %")
        ax2.set_title("7-day rolling coupling ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        format_date_axis(ax2)
        tight_layout(fig2)
        st.pyplot(fig2)
        plt.close()


def plot_arbitrage(start: str, end: str) -> None:
    """Cross-zone arbitrage: max price spread across all zone pairs per 15-min interval."""
    prices = get_prices_15min(tuple(get_spot_zones()))
    sample = prices.loc[start:end].dropna(axis=1, how="all")
    if sample.empty or len(sample.columns) < 2:
        st.warning("Need at least 2 zones with price data.")
        return

    # Max spread per interval
    max_spread = sample.max(axis=1) - sample.min(axis=1)
    cheapest = sample.idxmin(axis=1)
    most_expensive = sample.idxmax(axis=1)

    threshold = st.slider(
        "Arbitrage threshold (EUR/MWh)", 1.0, 50.0, 10.0, 1.0,
        help="Show intervals where max cross-zone spread exceeds this.",
        key="arb_thresh",
    )

    arb_hours = (max_spread > threshold).sum() / 4
    total_hours = len(max_spread) / 4
    arb_pct = arb_hours / total_hours * 100 if total_hours > 0 else 0
    avg_spread = max_spread.mean()
    max_val = max_spread.max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg max spread", f"{avg_spread:.2f} EUR/MWh")
    with c2:
        st.metric("Peak spread", f"{max_val:.1f} EUR/MWh")
    with c3:
        st.metric(f"Hours > {threshold:.0f} EUR", f"{arb_hours:.0f} h")
    with c4:
        st.metric("% of period", f"{arb_pct:.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1])

    # Spread time series
    ms_n = naive_index(max_spread.to_frame("spread"))
    axes[0].plot(ms_n.index, ms_n["spread"], lw=0.8, color="C0", alpha=0.8)
    axes[0].axhline(threshold, color="#ff4b4b", ls="--", lw=1.5, label=f"Threshold ({threshold:.0f} EUR)")
    # Shade above threshold
    above = max_spread > threshold
    above_n = naive_index(above.to_frame("a"))
    for i in range(1, len(above_n)):
        if above_n["a"].iloc[i]:
            axes[0].axvspan(above_n.index[i-1], above_n.index[i], alpha=0.2, color="#ff4b4b")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title("Max cross-zone price spread (arbitrage opportunity)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    format_date_axis(axes[0])

    # Most frequent cheapest/expensive pairs
    pair_counts = pd.Series(
        [f"{c}→{e}" for c, e in zip(cheapest, most_expensive)]
    ).value_counts().head(10)
    pair_counts.plot(kind="barh", ax=axes[1], color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Frequency (15-min intervals)")
    axes[1].set_title("Most common arbitrage direction (buy cheap → sell expensive)")

    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
