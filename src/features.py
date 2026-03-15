"""Feature engineering for spot price prediction."""
import numpy as np
import pandas as pd

from src.data_loader import (
    load_total_load,
    load_generation_pivot,
    load_weather,
    load_flows_into,
)
from src.utils import to_15min


def build_features(zone: str) -> pd.DataFrame | None:
    """Build 15-min feature matrix for prediction. Returns None if data missing."""
    try:
        load = load_total_load(zone)
        gen = load_generation_pivot(zone)
        weather = to_15min(load_weather(zone))
        flows = load_flows_into(zone)
    except FileNotFoundError:
        return None

    pivot_flows = flows.pivot(index="time", columns="zone", values="value (MW)")
    pivot_flows.index = pd.to_datetime(pivot_flows.index, utc=True)
    pivot_flows = to_15min(pivot_flows)

    ren = [
        "SOLAR", "WIND-ONSHORE", "WIND-OFFSHORE",
        "HYDRO-ROR", "HYDRO-WATER-RESERVOIR", "BIOMASS",
    ]
    ren_cols = [c for c in ren if c in gen.columns]
    total_gen = gen.sum(axis=1)
    renewable = gen[ren_cols].sum(axis=1) if ren_cols else pd.Series(0, index=gen.index)
    gen = gen.copy()
    gen["renewable_share"] = renewable / total_gen.replace(0, np.nan)

    net_import = (
        pivot_flows.sum(axis=1).to_frame("net_import")
        if not pivot_flows.empty
        else pd.DataFrame()
    )

    idx = load.index
    features = pd.DataFrame(index=idx)
    features["hour"] = idx.hour
    features["dayofweek"] = idx.dayofweek
    features["month"] = idx.month

    for name, df in [
        ("load", load),
        ("renewable_share", gen[["renewable_share"]]),
        ("net_import", net_import),
    ]:
        if not df.empty:
            features = features.join(df.reindex(idx).ffill(), how="left")
    features = features.join(weather.reindex(idx).ffill(), how="left")
    features = features.dropna(how="all", axis=1)
    return features
