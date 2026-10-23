#!/usr/bin/env python3
"""Make a visible-only observed h5ad for volume methods (e.g. NTF).

The full-volume methods train on an observed matrix and predict a regular grid.
For a held-out scenario, give NTF only the visible sections:
    python scripts/make_visible_observed.py \
      --observed observed_50um.h5ad \
      --scenarios canonical/scenarios.json \
      --combo M1a --out ntf_visible_M1a.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--combo", default="M1a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = ad.read_h5ad(args.observed)
    import json
    meta = json.loads(Path(args.scenarios).read_text())
    entries = meta.get("combos") or meta.get("scenarios") or meta
    if isinstance(entries, list):
        spec = next((x for x in entries if str(x.get("name")) == args.combo), None)
        if spec is None:
            raise KeyError(args.combo)
    else:
        spec = entries[args.combo]
    visible = spec.get("visible_slice_ids") or spec.get("visible")
    if not visible:
        raise ValueError(f"no visible slice list for {args.combo}")
    labels = a.obs["slice"].astype(str)
    mask = labels.isin([str(s) for s in visible]).to_numpy()
    b = a[mask].copy()
    X = b.X.toarray() if hasattr(b.X, "toarray") else np.asarray(b.X)
    b.X = np.ascontiguousarray(X, dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    b.write_h5ad(out, compression=None)
    print(f"[write] {out} shape={b.shape} visible={list(map(str, visible))}")


if __name__ == "__main__":
    main()
