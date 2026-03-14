"""
European Power Market Dashboard — AU Hack 2026
Layout and design only; plot logic lives in dashboard/plots/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.data_loader import SPOT_ZONES, FLOW_ZONES
from dashboard.zone_info import get_zone_info
from dashboard.zone_map import render_zone_map
from dashboard.plots import (
    plot_spot_prices,
    plot_zone_spot_price,
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

        # Sync zone between selectbox and map (use selected_zone; map must run before selectbox)
        if "selected_zone" not in st.session_state:
            st.session_state.selected_zone = "DE"

        zone = st.session_state.selected_zone

        # Map first so we can update selected_zone from clicks before the selectbox is created
        st.caption("Or click a country on the map")
        clicked_zone = render_zone_map(zone, key="zone_map")
        if clicked_zone and clicked_zone != zone and clicked_zone in FLOW_ZONES:
            st.session_state.selected_zone = clicked_zone
            st.rerun()

        # Ensure selected zone is valid (has flow data)
        if st.session_state.selected_zone not in FLOW_ZONES:
            st.session_state.selected_zone = "DE"

        zone = st.session_state.selected_zone
        st.selectbox(
            "Zone",
            FLOW_ZONES,
            index=FLOW_ZONES.index(zone),
            key="selected_zone",
        )
        zone = st.session_state.selected_zone
        info = get_zone_info(zone)
        st.markdown(f"**{info['full']}**")
        st.caption(f"Capital: {info['capital']}")
        st.write(info["description"])

        st.divider()
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
        st.subheader(f"{zone} spot price (selected range)")
        plot_zone_spot_price(zone, start_str, end_str)
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
        st.subheader(f"{zone} vs neighbor price spreads")
        plot_price_spreads(zone, start_str, end_str)


if __name__ == "__main__":
    main()
