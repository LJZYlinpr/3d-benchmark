#!/usr/bin/env python3
"""Patch PyG's optional `torch_cluster.random_walk` accesses.

Some torch_cluster builds (including the pinned torch 2.3/PyG 2.3 stack) do not
export the `random_walk` op, but `torch_geometric.nn.models.node2vec` and
`torch_geometric.utils.dropout` read it at import time.  isoST does not use
either, so make the attribute optional instead of failing the import.

Run with the target environment's Python:
    $PY_ISOST scripts/patch_pyg_torch_cluster.py
Idempotent.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

MARK = "patch_pyg_torch_cluster"
PAT = re.compile(r"^(\s*)random_walk = torch\.ops\.torch_cluster\.random_walk\s*$", re.M)


def main() -> int:
    spec = importlib.util.find_spec("torch_geometric")
    if spec is None or not spec.submodule_search_locations:
        print("[patch-pyg] torch_geometric not installed; skip")
        return 0
    root = Path(next(iter(spec.submodule_search_locations)))
    count = 0
    for p in root.rglob("*.py"):
        s = p.read_text(errors="ignore")
        if MARK in s or "random_walk = torch.ops.torch_cluster.random_walk" not in s:
            continue

        def repl(m: re.Match) -> str:
            ind = m.group(1)
            return (
                f"{ind}try:  # {MARK}\n"
                f"{ind}    random_walk = torch.ops.torch_cluster.random_walk\n"
                f"{ind}except AttributeError:\n"
                f"{ind}    random_walk = None"
            )

        s2 = PAT.sub(repl, s)
        if s2 != s:
            p.write_text(s2)
            count += 1
    print(f"[patch-pyg] patched {count} file(s) under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
