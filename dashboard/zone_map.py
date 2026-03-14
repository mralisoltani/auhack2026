"""Folium map with country highlighting for selected zone."""

from pathlib import Path

from src.zone_registry import get_custom_zone_info, get_flow_zones

# Zone -> ISO_A2 used in GeoJSON (_ZONE_ISO property)
ZONE_TO_ISO = {
    "AT": "AT",
    "BE": "BE",
    "CH": "CH",
    "CZ": "CZ",
    "DE": "DE",
    "DK1": "DK",
    "DK2": "DK",
    "FR": "FR",
    "NL": "NL",
    "NO2": "NO",
    "PL": "PL",
    "SE4": "SE",
}

# ISO from GeoJSON -> zone (for map click; countries with multiple zones use first)
# Built-in mapping; custom zones added dynamically in _zone_from_click
ISO_TO_ZONE = {
    "AT": "AT",
    "BE": "BE",
    "CH": "CH",
    "CZ": "CZ",
    "DE": "DE",
    "DK": "DK1",
    "FR": "FR",
    "NL": "NL",
    "NO": "NO2",
    "PL": "PL",
    "SE": "SE4",
}


def _load_geojson():
    path = Path(__file__).resolve().parent / "data" / "europe_zones.geojson"
    import json
    with open(path) as f:
        return json.load(f)


def _point_in_polygon(lng: float, lat: float, coords: list) -> bool:
    """Ray-casting point-in-polygon. coords is list of [lng, lat] rings."""
    n = len(coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = coords[i][0], coords[i][1]
        xj, yj = coords[j][0], coords[j][1]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_geometry(lng: float, lat: float, geom: dict) -> bool:
    """Check if point is inside GeoJSON geometry (Polygon or MultiPolygon)."""
    if geom.get("type") == "Polygon":
        for ring in geom["coordinates"]:
            if _point_in_polygon(lng, lat, ring):
                return True
        return False
    if geom.get("type") == "MultiPolygon":
        for polygon in geom["coordinates"]:
            for ring in polygon:
                if _point_in_polygon(lng, lat, ring):
                    return True
        return False
    return False


def _zone_from_click(data: dict | None, geojson: dict | None = None) -> str | None:
    """Extract zone from st_folium click data. Uses last_clicked + point-in-polygon (GeoJSON
    last_object_clicked often doesn't work)."""
    if not data:
        return None
    # Build ISO -> zone including custom zones
    iso_to_zone = dict(ISO_TO_ZONE)
    zone_to_iso = _get_zone_to_iso()
    for z, iso in zone_to_iso.items():
        if iso not in iso_to_zone:
            iso_to_zone[iso] = z
    # Try last_object_clicked first (sometimes works)
    obj = data.get("last_object_clicked") or data.get("last_object_clicked_tooltip")
    if obj and isinstance(obj, dict):
        props = obj.get("properties", {})
        iso = props.get("_ZONE_ISO") or props.get("ISO2") or props.get("ISO_A2")
        if iso:
            return iso_to_zone.get(iso)
    # Fallback: last_clicked (lat/lng) + point-in-polygon
    click = data.get("last_clicked")
    if not click or not geojson:
        return None
    lat = click.get("lat")
    lng = click.get("lng")
    if lat is None or lng is None:
        return None
    for feat in geojson.get("features", []):
        if _point_in_geometry(lng, lat, feat.get("geometry", {})):
            props = feat.get("properties", {})
            iso = props.get("_ZONE_ISO") or props.get("ISO2") or props.get("ISO_A2")
            return iso_to_zone.get(iso) if iso else None
    return None


def _get_zone_to_iso() -> dict:
    """Zone -> ISO for GeoJSON, including custom zones."""
    out = dict(ZONE_TO_ISO)
    for z in get_flow_zones():
        if z not in out:
            custom = get_custom_zone_info(z)
            if custom:
                out[z] = custom.get("iso", z)
            else:
                out[z] = z[:2] if len(z) >= 2 else z
    return out


def _geojson_iso_set(geojson: dict) -> set[str]:
    """Set of ISO codes present in the GeoJSON."""
    out = set()
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        iso = props.get("_ZONE_ISO") or props.get("ISO2") or props.get("ISO_A2")
        if iso:
            out.add(iso)
    return out


def _make_circle_geojson(lat: float, lon: float, radius_km: float = 80) -> dict:
    """Create a GeoJSON polygon approximating a circle (for custom zones not in main GeoJSON)."""
    import math
    points = []
    for i in range(32):
        angle = 2 * math.pi * i / 32
        # ~111 km per degree lat; lon varies by cos(lat)
        dlat = radius_km / 111 * math.cos(angle)
        dlon = radius_km / (111 * math.cos(math.radians(lat))) * math.sin(angle)
        points.append([lon + dlon, lat + dlat])
    points.append(points[0])
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [points]},
    }


def render_zone_map(zone: str, height: int = 280, key: str = "zone_map") -> str | None:
    """Render a map with the selected country highlighted. Returns zone if a country was clicked."""
    import streamlit as st

    from dashboard.zone_info import get_zone_coords

    try:
        import folium
        from streamlit_folium import st_folium
    except (ImportError, SystemError, OSError):
        # Fallback: simple point map if folium fails
        import pandas as pd
        lat, lon = get_zone_coords(zone)
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=5, use_container_width=True)
        return None

    geojson = _load_geojson()
    zone_to_iso = _get_zone_to_iso()
    target_iso = zone_to_iso.get(zone, "DE")
    geojson_isos = _geojson_iso_set(geojson)

    lat, lon = get_zone_coords(zone)
    m = folium.Map(location=[lat, lon], zoom_start=4, tiles="CartoDB positron")

    def style_fn(feature):
        props = feature.get("properties", {})
        iso = props.get("_ZONE_ISO") or props.get("ISO2") or props.get("ISO_A2") or ""
        is_selected = iso == target_iso
        return {
            "fillColor": "#3388ff" if is_selected else "#e0e0e0",
            "color": "#1a5fb4" if is_selected else "#888888",
            "weight": 2 if is_selected else 1,
            "fillOpacity": 0.7 if is_selected else 0.4,
        }

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME"],
            aliases=["Country: "],
            localize=True,
        ),
    ).add_to(m)

    # Add circle markers for custom zones not in GeoJSON (e.g. Italy)
    for z in get_flow_zones():
        custom = get_custom_zone_info(z)
        if custom:
            iso = custom.get("iso", z)
            if iso not in geojson_isos:
                clat, clon = custom["lat"], custom["lon"]
                circle_feat = _make_circle_geojson(clat, clon)
                circle_feat["properties"] = {"_ZONE_ISO": iso, "NAME": custom.get("name", z)}
                circle_geojson = {"type": "FeatureCollection", "features": [circle_feat]}
                is_selected = zone_to_iso.get(zone) == iso
                folium.GeoJson(
                    circle_geojson,
                    style_function=lambda f, sel=is_selected: {
                        "fillColor": "#3388ff" if sel else "#22aa44",
                        "color": "#1a5fb4" if sel else "#1a7a33",
                        "weight": 2,
                        "fillOpacity": 0.7,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=["NAME"], aliases=["Zone: "]),
                ).add_to(m)

    data = st_folium(
        m,
        height=height,
        use_container_width=True,
        key=key,
        returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_tooltip"],
    )
    # Merge custom zone circles into geojson for point-in-polygon click detection
    merged = dict(geojson)
    merged["features"] = list(geojson["features"])
    for z in get_flow_zones():
        custom = get_custom_zone_info(z)
        if custom and custom.get("iso") not in geojson_isos:
            circle_feat = _make_circle_geojson(custom["lat"], custom["lon"])
            circle_feat["properties"] = {"_ZONE_ISO": custom["iso"], "NAME": custom.get("name", z)}
            merged["features"].append(circle_feat)
    return _zone_from_click(data, merged)
