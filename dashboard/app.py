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
    plot_negative_prices,
    plot_price_volatility,
    plot_peak_offpeak,
    plot_renewable_penetration,
    plot_supply_demand,
    plot_residual_load,
    plot_generation_mix,
    plot_penetration_vs_price,
    plot_fossil_ratio_vs_price,
    plot_duck_curve,
    plot_flows_in,
    plot_flows_out,
    plot_net_import_vs_price,
    plot_net_import_vs_price_2,
    plot_congestion,
    plot_flow_drivers,
    plot_weather_load,
    plot_wind_generation,
    plot_cloud_solar,
    plot_weather_price_impact,
    plot_correlation_matrix,
    plot_price_spreads,
    plot_coupling_events,
    plot_arbitrage,
    plot_prediction,
    plot_feature_importance_comparison,
    plot_data_overview,
    evaluate_renewable_business_case,
)


def main():
    st.set_page_config(page_title="European Power Market", page_icon="⚡", layout="wide")

    # Wide cover image at top
    cover_path = Path(__file__).resolve().parent / "assets" / "in_commodities_a_s_cover.jpg"
    if cover_path.exists():
        st.image(str(cover_path), use_container_width=True)
    st.title("European Power Market Dashboard")
    st.caption("AU Hack 2026 — InCommodities case - JAKA Team")

    with st.sidebar:
        st.markdown(
            "<style>[data-testid='stSidebar'] { padding-top: 0 !important; }</style>",
            unsafe_allow_html=True,
        )
        sidebar_logo_path = Path(__file__).resolve().parent / "assets" / "small_logo.png"
        if sidebar_logo_path.exists():
            st.image(str(sidebar_logo_path), use_container_width=True)
        st.caption("Please start by choosing the region")

        # Zones from disk + custom (dynamic) — union so new zones appear with any data
        flow_zones = get_flow_zones()
        spot_zones = get_spot_zones()
        all_zones = sorted(set(flow_zones) | set(spot_zones))

        # Sync zone between selectbox and map (use selected_zone; map must run before selectbox)
        if "selected_zone" not in st.session_state:
            st.session_state.selected_zone = "DE"

        zone = st.session_state.selected_zone

        # Map first so we can update selected_zone from clicks before the selectbox is created
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Data Overview",
        "Spot Prices",
        "Supply & Demand",
        "Flows",
        "Weather",
        "Market Coupling",
        "Prediction",
        ])

    with tab1:
        st.markdown("**What data do we have, and is it complete?** What are the distributions and time ranges for each data type?")
        st.divider()
        plot_data_overview(zone, start_str, end_str)

    with tab2:
        st.markdown("**What do spot prices look like across zones?** When do they spike, go negative, or become volatile? What is the peak vs off-peak spread?")
        st.divider()
        st.subheader("Spot prices across zones")
        st.caption("Compare day-ahead electricity prices across European bidding zones. Price convergence signals market coupling; divergence signals congestion.")
        plot_spot_prices(start_str, end_str)
        st.subheader(f"{zone} spot price (selected range)")
        st.caption("Time series of the day-ahead spot price for the selected zone. Spikes often correlate with low renewable output or high demand.")
        plot_zone_spot_price(zone, start_str, end_str)
        st.subheader("Price by hour (all-time)")
        st.caption("Average price profile by hour of day. Peak hours (morning and evening) typically show higher prices due to demand patterns.")
        plot_price_by_hour(zone)
        st.subheader("Negative price analysis")
        st.caption("When renewable supply exceeds demand, prices can go negative. This shows how often, how long, and under what conditions (high wind, low load) that happens.")
        plot_negative_prices(zone, start_str, end_str)
        st.subheader("Price volatility & regime analysis")
        st.caption("Rolling standard deviation of price (24h and 7d windows). Red-shaded periods are high-volatility regimes driven by weather shifts or supply shocks.")
        plot_price_volatility(zone, start_str, end_str)
        st.subheader("Peak vs Off-peak spread")
        st.caption("Traders trade the peak (08-20 weekdays) vs off-peak price gap. A positive spread means daytime power is more expensive, which is normal for demand-driven markets.")
        plot_peak_offpeak(zone, start_str, end_str)

    with tab3:
        st.markdown("**How does supply meet demand, and what drives the spot price?** Why do renewables earn a profit despite near-zero fuel costs?")
        st.divider()
        st.subheader("Energy Generation Distribution")
        st.caption("Stacked area chart of generation by fuel type, ordered from cheapest (renewables) to most expensive (gas/oil). This is the merit order in action.")
        plot_generation_mix(zone, start_str, end_str)
        st.subheader("Energy Load vs Generation")
        st.caption("Total electricity consumption vs total generation. The gap is covered by imports (if load > generation) or exports (if generation > load).")
        plot_supply_demand(zone, start_str, end_str)
        st.subheader("Residual Load")
        st.caption("Load minus total generation. Positive residual means the zone needs imports; negative means it has surplus to export.")
        plot_residual_load(zone, start_str, end_str)
        st.subheader("Renewable Share")
        st.caption("Percentage of total generation coming from wind + solar. Higher penetration tends to push prices down via the merit order effect.")
        plot_renewable_penetration(zone, start_str, end_str)
        st.subheader("Renewable Share vs Price | Fossil Share vs Price")
        st.caption("Left: inverse relationship between renewable penetration and spot price. Right: fossil share correlates positively with price when gas/coal set the marginal price.")
        col_pen, col_fossil = st.columns(2)
        with col_pen:
            plot_penetration_vs_price(zone, start_str, end_str, figsize=(5, 3.5))
        with col_fossil:
            plot_fossil_ratio_vs_price(zone, start_str, end_str, figsize=(5, 3.5))
        st.subheader("Solar Duck Curve")
        st.caption("Compares the intraday residual load shape on high-solar vs low-solar days. The midday dip and steep evening ramp create the classic 'duck curve' that challenges grid operators.")
        plot_duck_curve(zone, start_str, end_str)
        st.divider()
        st.subheader("Marginal Pricing & Renewable Business Case")
        st.caption("In Europe's marginal pricing system, all generators are paid the price of the most expensive unit needed. Renewables with near-zero fuel costs earn 'inframarginal rent' — the gap between market price and their cost.")
        evaluate_renewable_business_case(zone, start_str, end_str)

    with tab4:
        st.markdown("**How is power distributed between countries?** What drives net imports and exports? When do interconnectors hit capacity limits?")
        st.divider()
        st.subheader("Flows into zone")
        st.caption("Cross-border electricity imports from neighboring zones. Stacked by origin to show which neighbors supply the most power.")
        plot_flows_in(zone, start_str, end_str)
        st.subheader("Flows out of zone")
        st.caption("Cross-border electricity exports to neighboring zones. High exports typically occur when the zone has surplus generation (e.g., strong wind).")
        plot_flows_out(zone, start_str, end_str)
        st.subheader("Congestion / capacity analysis")
        st.caption("Detects when cross-border flows flatline at capacity limits (congestion). When congested, prices between zones decouple — the core constraint on European power distribution.")
        plot_congestion(zone, start_str, end_str)
        st.subheader("Net import vs Spot price")
        st.caption("Relationship between net imports and spot price. High net imports with high prices signal domestic supply shortage; low imports with low prices signal surplus.")
        plot_net_import_vs_price(zone, start_str, end_str)
        plot_net_import_vs_price_2(zone, start_str, end_str)
        st.subheader("Flow driver decomposition")
        st.caption("What drives net power flow? Shows correlations between net import and price spread vs neighbors, wind generation, load, and renewable share.")
        plot_flow_drivers(zone, start_str, end_str)

    with tab5:
        st.markdown("**How does weather affect production and consumption?** Does temperature drive load? Does wind speed drive wind generation? Does cloud cover affect solar output?")
        st.divider()
        st.subheader("Temperature vs Load")
        st.caption("Electricity demand rises in cold weather (heating) and hot weather (cooling). This U-shaped relationship is a key fundamental for price forecasting.")
        plot_weather_load(zone, start_str, end_str)
        st.subheader("Wind vs Wind Generation")
        st.caption("Wind speed directly drives wind turbine output. The non-linear relationship reflects cut-in speed, rated power, and cut-out limits of turbines.")
        plot_wind_generation(zone, start_str, end_str)
        st.subheader("Cloud cover vs Solar generation")
        st.caption("Solar output drops sharply with increasing cloud cover. This daytime-only analysis shows how cloud conditions impact renewable supply and residual demand.")
        plot_cloud_solar(zone, start_str, end_str)
        st.subheader("Temperature extremes vs Price")
        st.caption("Cold and hot extremes (bottom/top 10% of temperature) drive load spikes for heating and cooling, which in turn push spot prices higher.")
        plot_weather_price_impact(zone, start_str, end_str)

    with tab6:
        st.markdown("**Which zones are price-linked, and when do they couple or decouple?** Where are cross-zone arbitrage opportunities?")
        st.divider()
        st.subheader("Price correlation by zone")
        st.caption("How tightly is the selected zone's spot price linked to others? High correlation means strong market coupling; low correlation means frequent congestion or different generation mixes.")
        plot_correlation_matrix(zone)
        st.subheader(f"{zone} vs neighbor price spreads")
        st.caption("Price difference between the selected zone and each neighbor over time. Large persistent spreads signal congestion on interconnectors.")
        plot_price_spreads(zone, start_str, end_str)
        st.subheader("Coupling event detection")
        st.caption("When the price spread between two zones is near zero, they are 'coupled' (same market price). When it diverges, congestion prevents price equalization.")
        plot_coupling_events(zone, start_str, end_str)
        st.subheader("Cross-zone arbitrage opportunities")
        st.caption("For each 15-min interval, the maximum price spread across all zone pairs. Traders profit by buying in the cheapest zone and selling in the most expensive.")
        plot_arbitrage(start_str, end_str)

    with tab7:
        st.markdown("**Can we predict spot prices from fundamentals?** Which features matter most for each zone?")
        st.divider()
        st.subheader(f"{zone}: Spot price prediction")
        st.caption("Choose model, click Start train. Trains on first 90%, predicts on last 10% of date range.")
        plot_prediction(zone, start_str, end_str)
        st.divider()
        st.subheader("Feature importance comparison across zones")
        st.caption("Train a quick RandomForest on each zone and compare which features matter most. E.g., wind dominates in DK1 while temperature matters more in FR.")
        plot_feature_importance_comparison()
        


if __name__ == "__main__":
    main()
