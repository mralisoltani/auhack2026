"""Zone metadata: coordinates and country information for the map."""

# Approximate center coordinates (lat, lon) for each bidding zone
ZONE_COORDS = {
    "AT": (47.5162, 14.5501),   # Austria
    "BE": (50.5039, 4.4699),    # Belgium
    "CH": (46.8182, 8.2275),    # Switzerland
    "CZ": (49.8175, 15.4730),   # Czech Republic
    "DE": (51.1657, 10.4515),   # Germany
    "DK1": (56.2639, 9.5018),   # Denmark West (Jutland + Funen)
    "DK2": (55.6761, 12.5683),  # Denmark East (Zealand)
    "FR": (46.2276, 2.2137),    # France
    "NL": (52.1326, 5.2913),    # Netherlands
    "NO2": (59.9139, 10.7522),  # Norway South
    "PL": (51.9194, 19.1451),   # Poland
    "SE4": (55.6050, 13.0038),  # Sweden South
}

ZONE_INFO = {
    "AT": {
        "name": "Austria",
        "full": "Austria (AT)",
        "capital": "Vienna",
        "description": "Major hydropower producer; key transit hub for Central European electricity flows.",
    },
    "BE": {
        "name": "Belgium",
        "full": "Belgium (BE)",
        "capital": "Brussels",
        "description": "Nuclear-heavy mix; interconnected with France, Netherlands, and UK via interconnectors.",
    },
    "CH": {
        "name": "Switzerland",
        "full": "Switzerland (CH)",
        "capital": "Bern",
        "description": "Pumped hydro storage hub; critical for Alpine power balancing and cross-border flows.",
    },
    "CZ": {
        "name": "Czech Republic",
        "full": "Czech Republic (CZ)",
        "capital": "Prague",
        "description": "Coal and nuclear base load; central position in CEE power market.",
    },
    "DE": {
        "name": "Germany",
        "full": "Germany (DE)",
        "capital": "Berlin",
        "description": "Europe's largest power market; major wind/solar expansion, phasing out nuclear.",
    },
    "DK1": {
        "name": "Denmark West",
        "full": "Denmark West — DK1 (Jutland, Funen)",
        "capital": "Aarhus",
        "description": "Wind-rich zone; strong interconnection to Norway and Germany.",
    },
    "DK2": {
        "name": "Denmark East",
        "full": "Denmark East — DK2 (Zealand)",
        "capital": "Copenhagen",
        "description": "Connected to Sweden and Germany; high wind share.",
    },
    "FR": {
        "name": "France",
        "full": "France (FR)",
        "capital": "Paris",
        "description": "Nuclear-dominated; major exporter; price setter for Western Europe.",
    },
    "NL": {
        "name": "Netherlands",
        "full": "Netherlands (NL)",
        "capital": "Amsterdam",
        "description": "Gas and wind; hub for North Sea offshore and continental flows.",
    },
    "NO2": {
        "name": "Norway South",
        "full": "Norway South — NO2",
        "capital": "Oslo",
        "description": "Hydropower-dominated; key storage partner for Nordic/Baltic markets.",
    },
    "PL": {
        "name": "Poland",
        "full": "Poland (PL)",
        "capital": "Warsaw",
        "description": "Coal-heavy; largest CEE market; increasing cross-border capacity.",
    },
    "SE4": {
        "name": "Sweden South",
        "full": "Sweden South — SE4",
        "capital": "Malmö",
        "description": "Hydropower and nuclear; linked to Denmark, Germany, Poland.",
    },
}


def get_zone_coords(zone: str) -> tuple[float, float]:
    """Return (lat, lon) for a zone. Falls back to Germany if unknown."""
    return ZONE_COORDS.get(zone, ZONE_COORDS["DE"])


def get_zone_info(zone: str) -> dict:
    """Return info dict for a zone. Falls back to generic if unknown."""
    return ZONE_INFO.get(zone, {
        "name": zone,
        "full": zone,
        "capital": "—",
        "description": "European power market bidding zone.",
    })
