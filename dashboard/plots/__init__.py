"""Plot modules for the dashboard."""

from .spot_prices import plot_spot_prices, plot_zone_spot_price, plot_price_by_hour, plot_negative_prices, plot_price_volatility, plot_peak_offpeak
from .supply_demand import plot_supply_demand, plot_generation_mix, plot_residual_load, plot_renewable_penetration, plot_penetration_vs_price, plot_fossil_ratio_vs_price, plot_duck_curve
from .flows import plot_flows_in, plot_flows_out, plot_net_import_vs_price, plot_net_import_vs_price_2, plot_congestion, plot_flow_drivers
from .weather import plot_weather_load, plot_wind_generation, plot_cloud_solar, plot_weather_price_impact
from .market_coupling import plot_correlation_matrix, plot_price_spreads, plot_coupling_events, plot_arbitrage
from .prediction import plot_prediction, plot_feature_importance_comparison
from .data_info import plot_data_overview
from .marginal_pricing import evaluate_renewable_business_case
