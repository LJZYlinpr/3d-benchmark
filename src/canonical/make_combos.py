#!/usr/bin/env python
"""Write combos.json (slice lists + bin counts) for the missing-slice experiment."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anndata as ad
import common as C

a = ad.read_h5ad(C.OBS)
path = C.write_combos_json(a)
print(path)
print(open(path).read())
