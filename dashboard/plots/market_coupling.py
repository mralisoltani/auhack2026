"""Market coupling plots — Notebook 06."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import get_prices_15min, naive_index, format_date_axis, tight_layout
from src.zone_registry import get_spot_zones


def plot_correlation_matrix() -> None:
    """Spot price correlation matrix across zones."""
    prices = get_prices_15min(tuple(get_spot_zones()))
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
