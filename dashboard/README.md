# European Power Market Dashboard

Interactive Streamlit dashboard for the AU Hack 2026 energy market analysis.

## Run

From the project root:

```bash
streamlit run dashboard/app.py
```

Or:

```bash
cd dashboard && streamlit run app.py
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
