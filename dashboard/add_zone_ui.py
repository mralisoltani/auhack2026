"""
UI for adding a new country/zone with CSV file uploads.
"""
import io
import streamlit as st
from pathlib import Path

from src.data_loader import DATA_ROOT
from src.zone_registry import register_custom_zone

# Country name -> (zone_code, iso_2, lat, lon) for map and dropdown
COUNTRY_OPTIONS = [
    ("Italy", "IT", "IT", 41.8719, 12.5674),
    ("Spain", "ES", "ES", 40.4637, -3.7492),
    ("Portugal", "PT", "PT", 39.3999, -8.2245),
    ("United Kingdom", "GB", "GB", 55.3781, -3.4360),
    ("Ireland", "IE", "IE", 53.1424, -7.6921),
    ("Greece", "GR", "GR", 39.0742, 21.8243),
    ("Hungary", "HU", "HU", 47.1625, 19.5033),
    ("Romania", "RO", "RO", 45.9432, 24.9668),
    ("Slovakia", "SK", "SK", 48.6690, 19.6990),
    ("Slovenia", "SI", "SI", 46.1512, 14.9955),
    ("Croatia", "HR", "HR", 45.1001, 15.2000),
    ("Bulgaria", "BG", "BG", 42.7339, 25.4858),
    ("Serbia", "RS", "RS", 44.0165, 21.0059),
    ("Finland", "FI", "FI", 61.9241, 25.7482),
    ("Estonia", "EE", "EE", 58.5953, 25.0136),
    ("Latvia", "LV", "LV", 56.8796, 24.6032),
    ("Lithuania", "LT", "LT", 55.1694, 23.8813),
    ("Luxembourg", "LU", "LU", 49.8153, 6.1296),
    ("Other", "", "", 50.0, 10.0),
]

# Expected columns per file type
FILE_SCHEMAS = {
    "spot-price": {"required": ["time", "value (EUR/MWh)"], "filename": "{zone}-spot-price.csv"},
    "total-load": {"required": ["time", "value (MW)"], "filename": "{zone}-total-load.csv"},
    "generation": {"required": ["type", "time", "value (MW)"], "filename": "{zone}-generation.csv"},
    "flows": {"required": ["zone", "time", "value (MW)"], "filename": "{zone}-physical-flows-in.csv"},
    "weather": {
        "required": ["time", "temperature_2m (°C)"],
        "filename": "{zone}-open-meteo.csv",
        "skip_rows": 3,
    },
}


def _validate_csv(uploaded_file, file_type: str) -> tuple[bool, str]:
    """Validate uploaded CSV has required columns. Returns (ok, error_msg)."""
    import pandas as pd

    schema = FILE_SCHEMAS[file_type]
    try:
        kwargs = {}
        if schema.get("skip_rows"):
            kwargs["skiprows"] = schema["skip_rows"]
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=5, **kwargs)
    except Exception as e:
        return False, str(e)
    for col in schema["required"]:
        if col not in df.columns:
            return False, f"Missing column: {col}"
    return True, ""


def render_add_zone_section() -> None:
    """Render the Add new country section in the sidebar."""
    st.divider()
    st.subheader("Add new country")
    st.caption("Upload CSV files to add a new bidding zone. Files must match the expected format.")

    country_sel = st.selectbox(
        "Country for new data",
        options=[c[0] for c in COUNTRY_OPTIONS],
        key="add_zone_country",
    )
    idx = next(i for i, c in enumerate(COUNTRY_OPTIONS) if c[0] == country_sel)
    name, zone_code, iso, lat, lon = COUNTRY_OPTIONS[idx][:5]

    if country_sel == "Other":
        zone_code = st.text_input("Zone code (e.g. IT, ES)", key="add_zone_code", max_chars=6).strip().upper()
        name = st.text_input("Country name", key="add_zone_name", value="Custom")
        iso = st.text_input("ISO code for map", key="add_zone_iso", value=zone_code or "XX", max_chars=3).strip().upper()
        lat = st.number_input("Latitude", key="add_zone_lat", value=50.0, format="%.2f")
        lon = st.number_input("Longitude", key="add_zone_lon", value=10.0, format="%.2f")

    if not zone_code:
        st.info("Select a country or enter a zone code.")
        return

    # File uploaders
    uploads = {}
    for file_type, schema in FILE_SCHEMAS.items():
        label = file_type.replace("-", " ").title()
        f = st.file_uploader(
            f"{label} ({schema['filename'].format(zone=zone_code)})",
            type=["csv"],
            key=f"add_zone_{file_type}",
        )
        if f:
            ok, err = _validate_csv(f, file_type)
            if ok:
                uploads[file_type] = f
                st.caption("✓ Valid")
            else:
                st.error(err)

    add_btn = st.button("Add zone", type="primary", key="add_zone_btn")

    if add_btn:
        if not uploads:
            st.error("Upload at least one CSV file (spot-price recommended).")
            return

        # Save files to data folder
        for file_type, f in uploads.items():
            schema = FILE_SCHEMAS[file_type]
            folder = DATA_ROOT / file_type
            folder.mkdir(parents=True, exist_ok=True)
            out_path = folder / schema["filename"].format(zone=zone_code)
            out_path.write_bytes(f.getvalue())
            st.success(f"Saved {out_path.name}")

        # Register custom zone for map/dropdown
        register_custom_zone(
            zone=zone_code,
            name=name,
            iso=iso,
            lat=lat,
            lon=lon,
            capital="—",
            description=f"Custom zone added via dashboard.",
        )

        st.cache_data.clear()
        st.success(f"Zone **{zone_code}** added. Refresh the page to see it in the dropdown and map.")
        st.info("If the map does not show your country, ensure the GeoJSON includes it. You can select the zone from the dropdown.")
        st.rerun()
