# AU Hack 2026 — European Power Market Analysis

Notebooks for the InCommodities case: **Decode the power market**.

## Evaluation mapping

| Notebook | Criterion | Focus |
|----------|-----------|-------|
| 01_data_loading_and_validation | **Craft** | Clean, extensible data loading |
| 02_supply_demand_balance | **Insight** | Supply/demand constraints in the data |
| 03_spot_price_fundamentals | **Insight** | What drives spot price |
| 04_germany_flows_analysis | **Insight** | Germany net flow drivers |
| 05_weather_fundamentals | **Insight** | Weather → production & consumption |
| 06_market_coupling | **Originality** | Germany's impact on neighbor prices |
| 07_prediction_model | **Extensibility** | Spot price prediction from fundamentals |
| 08_dashboard_artefact | **Working artefact** | Visualization + "why this matters" |

## Run order

1. Run `01_data_loading_and_validation` first (validates schema, sets up loader).
2. Notebooks 02–06 can run in any order.
3. `07_prediction_model` uses the loader and feature logic from earlier notebooks.
4. `08_dashboard_artefact` produces the shareable artefact.

## Dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode: `pip install -e .`

## Running

- **Dashboard**: `streamlit run dashboard/app.py`
- **Full instructions** (Docker, API mode): see [docs/RUNNING.md](docs/RUNNING.md)

## Why this matters

The European power market behaves like a complex distributed system. Spot prices, flows, and generation are driven by supply–demand balance, weather, and cross-border coupling. InCommodities and other participants need to interpret these signals in real time. These notebooks provide:

- **Insight** into how fundamentals drive prices and flows
- **Extensible** code that can support more zones, data sources, or live feeds
- **A working artefact** (dashboard) to explore and explain market dynamics
