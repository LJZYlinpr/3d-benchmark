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
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
bins = C.load_bins()
sec = {s: np.flatnonzero(bins["section"] == s) for s in ["24", "26"]}
def xy(s): return np.ascontiguousarray(np.stack([bins["x"][sec[s]], bins["y"][sec[s]]], 1), dtype=np.float32)
Xl = np.ascontiguousarray(bins["X"][sec["24"]]); Xu = np.ascontiguousarray(bins["X"][sec["26"]])
xl, xu = xy("24"), xy("26")
rng = np.random.default_rng(42)
for n_anc in [2500, 6152]:
    il = np.sort(rng.choice(len(xl), n_anc, replace=False)) if n_anc < len(xl) else np.arange(len(xl))
    iu = np.sort(rng.choice(len(xu), n_anc, replace=False)) if n_anc < len(xu) else np.arange(len(xu))
    M = P.kl_cost_gpu(Xl[il], Xu[iu], dev)
    C1 = torch.cdist(torch.from_numpy(xl[il]).to(dev), torch.from_numpy(xl[il]).to(dev)); C1 /= C1.max()
    C2 = torch.cdist(torch.from_numpy(xu[iu]).to(dev), torch.from_numpy(xu[iu]).to(dev)); C2 /= C2.max()
    a = torch.ones(len(il), device=dev) / len(il); b = torch.ones(len(iu), device=dev) / len(iu)
    for eps, it in [(0.05, 20), (0.05, 50), (0.02, 30)]:
        t = time.time()
        try:
            pi, lg = efgw(M, C1, C2, a, b, loss_fun="square_loss", epsilon=eps, alpha=0.1,
                          max_iter=it, tol=1e-7, solver="PPA", log=True)
            torch.cuda.synchronize()
            pi = pi.float()
            cy = ((pi / pi.sum(1, keepdim=True)) @ torch.from_numpy(xu[iu]).to(dev)).cpu().numpy()
            d = np.linalg.norm(cy - xl[il], axis=1)
            print(f"n_anc={n_anc} eps={eps} it={it} {time.time()-t:.1f}s "
                  f"med={np.median(d):.1f} p95={np.percentile(d,95):.1f} "
                  f"fgw={float(lg['fgw_dist']):.5f} last_err={float(lg['err'][-1]):.2e}", flush=True)
        except Exception as e:
            print(n_anc, eps, it, "FAILED", repr(e), flush=True)
