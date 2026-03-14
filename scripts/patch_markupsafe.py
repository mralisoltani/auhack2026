#!/usr/bin/env python3
"""
Patch markupsafe to catch SystemError (Python 3.14 compatibility).
Run once: python scripts/patch_markupsafe.py
"""
import sys
import importlib.util

spec = importlib.util.find_spec("markupsafe")
if not spec or not spec.origin:
    print("markupsafe not found")
    sys.exit(1)

init_path = spec.origin
with open(init_path) as f:
    content = f.read()

old = "except ImportError:"
new = "except (ImportError, SystemError):"

if new in content:
    print("markupsafe already patched")
    sys.exit(0)

if old not in content:
    print("Could not find target line in markupsafe __init__.py")
    sys.exit(1)

content = content.replace(old, new)
with open(init_path, "w") as f:
    f.write(content)

print("Patched markupsafe for Python 3.14 compatibility")
print("You can now use folium. Restart the dashboard.")
