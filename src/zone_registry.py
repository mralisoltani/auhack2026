"""
Dynamic zone registry: discovers zones from data folders and custom uploads.
"""
import json

from src.config import DATA_ROOT, CUSTOM_ZONES_PATH

# Built-in zones (from original data)
BUILTIN_SPOT = ["AT", "BE", "CH", "CZ", "DE", "DK1", "FR", "NL", "PL"]
BUILTIN_FLOW = ["AT", "BE", "CH", "CZ", "DE", "DK1", "DK2", "FR", "NL", "NO2", "PL", "SE4"]


def _zones_from_folder(subfolder: str, pattern: str) -> set[str]:
    """Extract zone codes from filenames in a data subfolder."""
    folder = DATA_ROOT / subfolder
    if not folder.exists():
        return set()
    zones = set()
    for f in folder.glob(pattern):
        name = f.stem
        if "-" in name:
            zones.add(name.split("-")[0])
    return zones


def get_spot_zones() -> list[str]:
    """All zones with spot price data (built-in + disk + custom)."""
    disk = _zones_from_folder("spot-price", "*-spot-price.csv")
    custom = _load_custom_zones()
    all_zones = set(BUILTIN_SPOT) | disk | set(custom.keys())
    return sorted(all_zones)


def get_flow_zones() -> list[str]:
    """All zones with flow data (built-in + disk + custom)."""
    disk = _zones_from_folder("flows", "*-physical-flows-in.csv")
    custom = _load_custom_zones()
    all_zones = set(BUILTIN_FLOW) | disk | set(custom.keys())
    return sorted(all_zones)


def _load_custom_zones() -> dict:
    """Load custom zone metadata from JSON."""
    if not CUSTOM_ZONES_PATH.exists():
        return {}
    try:
        with open(CUSTOM_ZONES_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_custom_zones(data: dict) -> None:
    """Save custom zone metadata to JSON."""
    CUSTOM_ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_ZONES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def register_custom_zone(
    zone: str,
    name: str,
    iso: str,
    lat: float,
    lon: float,
    capital: str = "—",
    description: str = "Custom zone added via dashboard.",
) -> None:
    """Register a new custom zone."""
    data = _load_custom_zones()
    data[zone] = {
        "name": name,
        "iso": iso,
        "lat": lat,
        "lon": lon,
        "capital": capital,
        "description": description,
    }
    _save_custom_zones(data)


def get_custom_zone_info(zone: str) -> dict | None:
    """Get metadata for a custom zone."""
    return _load_custom_zones().get(zone)
