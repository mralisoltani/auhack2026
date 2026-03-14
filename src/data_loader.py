"""
Data loader for European energy market data.
Designed for extensibility: new zones, data sources, or live feeds.
"""
from pathlib import Path
from typing import Optional

import glob
import pandas as pd

# Default paths - override for different environments
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Zone mapping for weather only (DK1/DK2 share DK weather file)
ZONE_TO_WEATHER = {"DK1": "DK", "DK2": "DK"}

# All zones in the dataset
SPOT_ZONES = ["AT", "BE", "CH", "CZ", "DE", "DK1", "FR", "NL", "PL"]
FLOW_ZONES = ["AT", "BE", "CH", "CZ", "DE", "DK1", "DK2", "FR", "NL", "NO2", "PL", "SE4"]


def load_spot_price(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load spot price for a zone. Returns time-indexed DataFrame."""
    path = data_root / "spot-price" / f"{zone}-spot-price.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"value (EUR/MWh)": "price"})
    df = df.set_index("time").sort_index()
    return df


def load_total_load(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load total load for a zone."""
    path = data_root / "total-load" / f"{zone}-total-load.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"value (MW)": "load"})
    df = df.set_index("time").sort_index()
    return df


def load_generation(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load generation by type. Returns long-format DataFrame."""
    path = data_root / "generation" / f"{zone}-generation.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def load_generation_pivot(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load generation pivoted wide (one column per type). Aggregates duplicates."""
    df = load_generation(zone, data_root)
    df = df.groupby(["time", "type"], as_index=False)["value (MW)"].mean()
    wide = df.pivot(index="time", columns="type", values="value (MW)")
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide.sort_index()


def load_flows_into(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load flows into a zone. zone column: 'ORIGIN->DEST'."""
    path = data_root / "flows" / f"{zone}-physical-flows-in.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def load_flows_out_of(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load flows out of a zone (from other zones' 'flows-in' files)."""
    # Flows out of DE appear in AT, BE, CH, etc. as DE->AT, DE->BE, ...
    all_flows = []
    for other in FLOW_ZONES:
        if other == zone:
            continue
        df = load_flows_into(other, data_root)
        out = df[df["zone"].str.startswith(f"{zone}->")]
        if not out.empty:
            out = out.copy()
            out["zone"] = out["zone"].str.replace(f"{zone}->", "", regex=False)
            all_flows.append(out)
    if not all_flows:
        return pd.DataFrame()
    return pd.concat(all_flows, ignore_index=True)


def _weather_zone(zone: str) -> str:
    """Map zone to weather file prefix (DK1/DK2 -> DK; only one weather file per country)."""
    return ZONE_TO_WEATHER.get(zone, zone)


def load_weather(zone: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load weather for a zone. Skips Open-Meteo metadata header."""
    wzone = _weather_zone(zone)
    pattern = str(data_root / "weather" / f"{wzone}-open-meteo-*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No weather file for zone {zone} (looked for {wzone})")
    df = pd.read_csv(matches[0], skiprows=3)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={
        "temperature_2m (°C)": "temperature",
        "wind_speed_10m (km/h)": "wind_speed_10m",
        "wind_speed_100m (km/h)": "wind_speed_100m",
        "relative_humidity_2m (%)": "humidity",
        "cloud_cover (%)": "cloud_cover",
        "wind_direction_10m (°)": "wind_direction_10m",
        "wind_direction_100m (°)": "wind_direction_100m",
        "precipitation (mm)": "precipitation",
    })
    df = df.set_index("time").sort_index()
    return df


def load_all_spot_prices(zones: Optional[list] = None, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load spot prices for all zones, wide format."""
    zones = zones or SPOT_ZONES
    dfs = []
    for z in zones:
        try:
            df = load_spot_price(z, data_root)
            df = df.rename(columns={"price": z})
            dfs.append(df)
        except FileNotFoundError:
            pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=1).sort_index()


def load_all_weather(zones: Optional[list] = None, data_root: Path = DATA_ROOT) -> dict[str, pd.DataFrame]:
    """Load weather for all zones. Returns dict zone -> DataFrame."""
    zones = zones or SPOT_ZONES
    result = {}
    for z in zones:
        try:
            result[z] = load_weather(z, data_root)
        except FileNotFoundError:
            pass
    return result
