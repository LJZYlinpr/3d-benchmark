import sys, time
sys.path.insert(0, "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/heart_10slice/methods")
import numpy as np
import _rb_common as C
import importlib.util
spec = importlib.util.spec_from_file_location(
    "paste", "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/heart_10slice/methods/PASTE-OT-Linear/run_paste_heart.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import torch
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
bins = C.load_bins()
sec = {s: np.flatnonzero(bins["section"] == s) for s in ["24", "26"]}
def xy(s): return np.ascontiguousarray(np.stack([bins["x"][sec[s]], bins["y"][sec[s]]], 1), dtype=np.float32)
for eps in [0.05, 0.01, 0.003]:
    t = time.time()
    ot, meta = P.solve_pair(np.ascontiguousarray(bins["X"][sec["24"]]), xy("24"),
                            np.ascontiguousarray(bins["X"][sec["26"]]), xy("26"),
                            0.1, eps, 1000, dev, print)
    dl = np.linalg.norm(ot["cy"] - xy("24"), axis=1)
    du = np.linalg.norm(ot["cx"] - xy("26"), axis=1)
    print(f"eps={eps} ot={time.time()-t:.1f}s disp_l med={np.median(dl):.1f} p95={np.percentile(dl,95):.1f} "
          f"disp_u med={np.median(du):.1f} p95={np.percentile(du,95):.1f} meta={meta}")
