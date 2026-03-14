# European Power Market Dashboard

Interactive Streamlit dashboard for the AU Hack 2026 energy market analysis.

## Structure

```
dashboard/
├── app.py              # Layout and design only
├── utils.py            # Shared helpers (caching, resampling)
├── plots/
│   ├── spot_prices.py      # Spot price plots
│   ├── supply_demand.py    # Load vs generation
│   ├── flows.py            # Flows in/out, net import vs price
│   ├── weather.py          # Temperature/wind vs load/generation
│   └── market_coupling.py  # Correlation matrix, price spreads
└── README.md
```

## Run

From the project root:

```bash
streamlit run dashboard/app.py
```

## Features

- **Zone selector** — Choose AT, BE, CH, CZ, DE, DK1, FR, NL, PL
- **Date range** — Pick start and end dates
- **Tabs:**
  - **Spot Prices** — Cross-zone prices, price by hour
  - **Supply & Demand** — Load vs generation
  - **Flows** — Flows in/out, net import vs price
  - **Weather** — Temperature vs load, wind vs generation
  - **Market Coupling** — Correlation matrix, price spreads

## Dependencies

Install with `pip install -r requirements.txt` from the project root.
