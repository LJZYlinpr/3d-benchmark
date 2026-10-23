from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def test_build_grids(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 2000
    xyz = rng.normal(size=(n, 3)) * 0.2
    xyz[:, 2] = np.repeat(np.arange(5) * 0.15, n // 5)[:n]
    cells = tmp_path / "cells.npz"
    np.savez(cells, xyz=xyz)
    out = tmp_path / "grids"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(repo / "src" / "benchmark" / "grid.py"),
         "--cells", str(cells), "--out-dir", str(out),
         "--resolutions", "10", "20", "40", "100"],
        check=True,
    )
    for res in (10, 20, 40, 100):
        p = out / f"grid_{res}um.npz"
        assert p.exists()
        g = np.load(p)
        assert g["obs_mask"].any()
        assert g["query_mask"].shape == g["obs_mask"].shape
        assert g["inside"].sum() >= g["obs_mask"].sum()
