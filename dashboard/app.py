"""
European Power Market Dashboard — AU Hack 2026
Layout and design only; plot logic lives in dashboard/plots/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.data_loader import SPOT_ZONES
from dashboard.plots import (
    plot_spot_prices,
    plot_price_by_hour,
    plot_supply_demand,
    plot_flows_in,
    plot_flows_out,
    plot_net_import_vs_price,
    plot_weather_load,
    plot_wind_generation,
    plot_correlation_matrix,
    plot_price_spreads,
)


def main():
    st.set_page_config(page_title="European Power Market", page_icon="⚡", layout="wide")
    st.title("⚡ European Power Market Dashboard")
    st.caption("AU Hack 2026 — InCommodities case: Decode the power market")

    with st.sidebar:
        st.header("Controls")
        zone = st.selectbox(
            "Zone",
            SPOT_ZONES,
            index=SPOT_ZONES.index("DE") if "DE" in SPOT_ZONES else 0,
        )
        start = st.date_input("Start date", pd.Timestamp("2024-07-01"))
        end = st.date_input("End date", pd.Timestamp("2024-07-14"))
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Spot Prices",
        "Supply & Demand",
        "Flows",
        "Weather",
        "Market Coupling",
    ])

    with tab1:
        st.subheader("Spot prices across zones")
        plot_spot_prices(start_str, end_str)
        st.subheader("Price by hour (all-time)")
        plot_price_by_hour(zone)

    with tab2:
        st.subheader("Load vs Generation")
        plot_supply_demand(zone, start_str, end_str)

    with tab3:
        st.subheader("Flows into zone")
        plot_flows_in(zone, start_str, end_str)
        st.subheader("Flows out of zone")
        plot_flows_out(zone, start_str, end_str)
        st.subheader("Net import vs Spot price")
        plot_net_import_vs_price(zone, start_str, end_str)

    with tab4:
        st.subheader("Temperature vs Load")
        plot_weather_load(zone, start_str, end_str)
        st.subheader("Wind vs Wind Generation")
        plot_wind_generation(zone, start_str, end_str)

    with tab5:
        st.subheader("Price correlation matrix")
        plot_correlation_matrix()
        st.subheader("DE vs neighbor price spreads")
        plot_price_spreads(start_str, end_str)


if __name__ == "__main__":
    main()
