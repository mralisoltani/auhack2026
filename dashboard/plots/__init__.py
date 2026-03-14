"""Plot modules for the dashboard."""

from .spot_prices import plot_spot_prices, plot_zone_spot_price, plot_price_by_hour
from .supply_demand import plot_supply_demand
from .flows import plot_flows_in, plot_flows_out, plot_net_import_vs_price
from .weather import plot_weather_load, plot_wind_generation
from .market_coupling import plot_correlation_matrix, plot_price_spreads
