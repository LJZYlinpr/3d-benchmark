import sys, time
sys.path.insert(0, "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/heart_10slice/methods")
import numpy as np
import _rb_common as C
import importlib.util
spec = importlib.util.spec_from_file_location(
    "paste", "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/heart_10slice/methods/PASTE-OT-Linear/run_paste_heart.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import torch
from ot.gromov import entropic_fused_gromov_wasserstein as efgw
from ot.gromov import fused_gromov_wasserstein as fgw
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
bins = C.load_bins()
sec = {s: np.flatnonzero(bins["section"] == s) for s in ["24", "26"]}
def xy(s): return np.ascontiguousarray(np.stack([bins["x"][sec[s]], bins["y"][sec[s]]], 1), dtype=np.float32)
Xl = np.ascontiguousarray(bins["X"][sec["24"]]); Xu = np.ascontiguousarray(bins["X"][sec["26"]])
xl, xu = xy("24"), xy("26")
M = P.kl_cost_gpu(Xl, Xu, dev)
Mnp = M.cpu().numpy()
print("M stats", float(Mnp.min()), float(Mnp.mean()), float(Mnp.std()), float(np.percentile(Mnp, [5,50,95]).tolist() if False else 0))
q = np.percentile(Mnp, [1,5,50,95,99]); print("M pct", q)
xlt = torch.from_numpy(xl).to(dev); xut = torch.from_numpy(xu).to(dev)
C1 = torch.cdist(xlt, xlt); C1 = C1 / C1.max()
C2 = torch.cdist(xut, xut); C2 = C2 / C2.max()
a = torch.ones(len(xl), device=dev) / len(xl); b = torch.ones(len(xu), device=dev) / len(xu)
print("C1 stats", float(C1.mean()), float(C1.max()))
for name, kw in [("PGD e=0.1", dict(solver="PGD", epsilon=0.1)),
                 ("PPA e=0.05", dict(solver="PPA", epsilon=0.05)),
                 ("PGD e=0.5", dict(solver="PGD", epsilon=0.5))]:
    t = time.time()
    try:
        pi, lg = efgw(M, C1, C2, a, b, loss_fun="square_loss", alpha=0.1,
                      max_iter=1000, tol=1e-9, log=True, **kw)
        torch.cuda.synchronize()
        pi = pi.float(); Pm = (pi / pi.sum(1, keepdim=True)).cpu().numpy()
        cy = Pm @ xu
        d = np.linalg.norm(cy - xl, axis=1)
        ent = float(-(pi * (pi.clamp_min(1e-30)).log()).sum())
        print(f"{name}: {time.time()-t:.1f}s disp med={np.median(d):.1f} p95={np.percentile(d,95):.1f} "
              f"entropy={ent:.1f} log={ {k: (float(v) if np.isscalar(v) else v) for k,v in list(lg.items())[:4]} }")
    except Exception as e:
        print(name, "FAILED", repr(e))
# non-entropic CG fused GW
t = time.time()
pi = fgw(M.cpu().numpy(), C1.cpu().numpy(), C2.cpu().numpy(), a.cpu().numpy(), b.cpu().numpy(),
         loss_fun="square_loss", alpha=0.1, max_iter=200, tol_rel=1e-9, verbose=False)
d = np.linalg.norm((pi / pi.sum(1, keepdims=True)) @ xu - xl, axis=1)
print(f"CG: {time.time()-t:.1f}s disp med={np.median(d):.1f} p95={np.percentile(d,95):.1f}")
