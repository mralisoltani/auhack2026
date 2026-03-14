"""
European Power Market Dashboard — AU Hack 2026
Layout and design only; plot logic lives in dashboard/plots/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from src.zone_registry import get_spot_zones, get_flow_zones
import numpy as np
from src.data_loader import SPOT_ZONES, FLOW_ZONES
from dashboard.zone_info import get_zone_info
from dashboard.zone_map import render_zone_map
from dashboard.plots import (
    plot_spot_prices,
    plot_zone_spot_price,
    plot_price_by_hour,
    plot_renewable_penetration,
    plot_supply_demand,
    plot_residual_load,
    plot_generation_mix,
    plot_penetration_vs_price,
    plot_fossil_ratio_vs_price,
    plot_flows_in,
    plot_flows_out,
    plot_net_import_vs_price,
    plot_net_import_vs_price_2,
    plot_weather_load,
    plot_wind_generation,
    plot_correlation_matrix,
    plot_price_spreads,
    plot_prediction,
    plot_data_overview,
    evaluate_renewable_business_case,
)


def main():
    st.set_page_config(page_title="European Power Market", page_icon="⚡", layout="wide")
    st.title("European Power Market Dashboard")
    st.caption("AU Hack 2026 — InCommodities case - JAKA Team")

    with st.sidebar:
        st.header("Controls")

        # Zones from disk + custom (dynamic) — union so new zones appear with any data
        flow_zones = get_flow_zones()
        spot_zones = get_spot_zones()
        all_zones = sorted(set(flow_zones) | set(spot_zones))

        # Sync zone between selectbox and map (use selected_zone; map must run before selectbox)
        if "selected_zone" not in st.session_state:
            st.session_state.selected_zone = "DE"

        zone = st.session_state.selected_zone

        # Map first so we can update selected_zone from clicks before the selectbox is created
        st.caption("Or click a country on the map")
        clicked_zone = render_zone_map(zone, key="zone_map")
        if clicked_zone and clicked_zone != zone and clicked_zone in all_zones:
            st.session_state.selected_zone = clicked_zone
            st.rerun()

        # Ensure selected zone is valid
        if st.session_state.selected_zone not in all_zones:
            st.session_state.selected_zone = all_zones[0] if all_zones else "DE"

        zone = st.session_state.selected_zone
        idx = all_zones.index(zone) if zone in all_zones else 0
        st.selectbox(
            "Zone",
            all_zones,
            index=idx,
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

        # Add new country section
        from dashboard.add_zone_ui import render_add_zone_section
        render_add_zone_section()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Data Overview",
        "Spot Prices",
        "Supply & Demand",
        "Flows",
        "Weather",
        "Market Coupling",
        "Marginal Pricing",
        "Prediction",
        ])

    with tab1:
        plot_data_overview(zone, start_str, end_str)

    with tab2:
        st.subheader("Spot prices across zones")
        plot_spot_prices(start_str, end_str)
        st.subheader(f"{zone} spot price (selected range)")
        plot_zone_spot_price(zone, start_str, end_str)
        st.subheader("Price by hour (all-time)")
        plot_price_by_hour(zone)

    with tab3:        
        st.subheader("Energy Generation Distribution")
        plot_generation_mix(zone, start_str, end_str)
        st.subheader("Energy Load vs Generation")
        plot_supply_demand(zone, start_str, end_str)
        st.subheader("Residual Load")
        plot_residual_load(zone, start_str, end_str)
        st.subheader("Renewable Share")
        plot_renewable_penetration(zone, start_str, end_str)
        st.subheader("Renewable Share vs Price")
        plot_penetration_vs_price(zone, start_str, end_str)
        st.subheader("Fossil Share vs Price")
        plot_fossil_ratio_vs_price(zone, start_str, end_str)

    with tab4:
        st.subheader("Flows into zone")
        plot_flows_in(zone, start_str, end_str)
        st.subheader("Flows out of zone")
        plot_flows_out(zone, start_str, end_str)
        st.subheader("Net import vs Spot price")
        plot_net_import_vs_price(zone, start_str, end_str)
        plot_net_import_vs_price_2(zone, start_str, end_str)

    with tab5:
        st.subheader("Temperature vs Load")
        plot_weather_load(zone, start_str, end_str)
        st.subheader("Wind vs Wind Generation")
        plot_wind_generation(zone, start_str, end_str)

    with tab6:
        st.subheader("Price correlation matrix")
        plot_correlation_matrix()
        st.subheader(f"{zone} vs neighbor price spreads")
        plot_price_spreads(zone, start_str, end_str)

    with tab7:
        st.header("Marginal Pricing Analysis")
        evaluate_renewable_business_case(zone, start_str, end_str)
        st.divider()

    with tab8:
        st.subheader(f"{zone}: Spot price prediction")
        st.caption("Choose model, click Start train. Trains on first 90%, predicts on last 10% of date range.")
        plot_prediction(zone, start_str, end_str)
        


if __name__ == "__main__":
    main()
