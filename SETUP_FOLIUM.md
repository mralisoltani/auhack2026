# Fix Folium Map (Python 3.14 / markupsafe compatibility)

The country-highlighting map uses Folium, which depends on `markupsafe`. On **Python 3.14**, markupsafe's C extension can raise `SystemError`, causing Folium to fail.

## Solution 1: Patch markupsafe (Python 3.14)

Run this once to patch the installed markupsafe to use the pure-Python fallback:

```bash
cd /home/ali/auhack2026
python scripts/patch_markupsafe.py
```

Then restart the dashboard. Folium should work.

**Note:** The patch is overwritten if you upgrade markupsafe. Re-run the script after upgrading.

## Solution 2: Use Python 3.12

Create a new conda environment with Python 3.12 (no patch needed):

```bash
conda create -n energy312 python=3.12 -y
conda activate energy312
cd /home/ali/auhack2026
pip install -r requirements.txt
streamlit run dashboard/app.py
```
