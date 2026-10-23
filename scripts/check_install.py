#!/usr/bin/env python3
import importlib.util
import sys

required = ["numpy", "scipy", "pandas", "anndata", "h5py"]
optional = ["torch", "skimage", "ot"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print("required missing:", missing)
print("optional available:", [m for m in optional if importlib.util.find_spec(m)])
if missing:
    sys.exit(1)
