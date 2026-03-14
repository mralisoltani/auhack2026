"""Spot price plots — Notebooks 03, 08."""
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import (
    get_prices_15min,
    to_15min,
    naive_index,
    format_date_axis,
    tight_layout,
)
from src.data_loader import load_spot_price


def plot_spot_prices(start: str, end: str) -> None:
    """Spot prices across zones."""
    prices = get_prices_15min()
    sample = naive_index(prices.loc[start:end])
    if sample.empty:
        st.warning("No data for selected range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    sample.plot(ax=ax, alpha=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Spot prices across zones")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_zone_spot_price(zone: str, start: str, end: str) -> None:
    """Spot price for selected zone over date range."""
    try:
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sp_15 = to_15min(sp)
    sample = naive_index(sp_15.loc[start:end])
    if sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    sample["price"].plot(ax=ax, color="C0", alpha=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{zone}: Spot price")
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()


def plot_price_by_hour(zone: str) -> None:
    """Average spot price by hour (all-time)."""
    try:
        sp = load_spot_price(zone)
    except FileNotFoundError:
        st.warning(f"No spot price data for {zone}.")
        return
    sp_ = sp.copy()
    sp_["hour"] = sp_.index.hour
    by_hour = sp_.groupby("hour")["price"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour.index, by_hour["mean"], yerr=by_hour["std"], capsize=2, alpha=0.8)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Mean spot price (EUR/MWh)")
    ax.set_title(f"{zone}: Average spot price by hour")
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
