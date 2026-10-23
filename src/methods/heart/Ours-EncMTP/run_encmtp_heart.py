#!/usr/bin/env python
"""Ours-EncMTP for the heart 10-slice target-only benchmark.

Adapted from Workspace/merfish_test/Ours-EncMTP/run_encmtp.py:
  * one token per visible 50 um bin; features = Fourier(x,y,z) + soft slice
    embedding, 2-layer MLP token encoder (hidden 256);
  * 2 cross-attention blocks over per-visible-slice kNN context tokens
    (3 nearest visible bins per visible slice);
  * main head 238 genes + K=4 multi-token-prediction auxiliary heads;
  * MSE in per-gene standardised space (stats from visible bins only),
    context token dropout 15% + whole-slice masking p=0.5, aux weight 0.5;
  * model selection on a 5% held-out subset of *visible* bins (MSE only).

Training uses visible-section bins only; hidden target expression is never
read.  Inference predicts hidden target bins from coordinates + visible
context and writes target-only predictions.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _rb_common as C  # noqa: E402

METHOD = "Ours-EncMTP"
K_PER_SLICE = 3
HIDDEN = 256
N_BLOCKS = 2
N_HEADS = 4
FREQ_LEVELS = 6
REL_LEVELS = 3
AUX_K = 4
TOKEN_DROPOUT = 0.15
SLICE_MASK_PROB = 0.5
AUX_WEIGHT = 0.5
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 2024
VAL_FRAC = 0.05


class Fourier(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.register_buffer("freq", torch.pi * (2.0 ** torch.arange(levels)))

    def forward(self, x):
        a = x.unsqueeze(-1) * self.freq
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(-2)


class TokenEncoder(nn.Module):
    def __init__(self, n_slices, hidden, levels=FREQ_LEVELS, slice_emb=16):
        super().__init__()
        self.slice_emb = nn.Embedding(n_slices, slice_emb)
        self.fourier = Fourier(levels)
        in_dim = 3 * 2 * levels + 3 + slice_emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.LayerNorm(hidden))

    def forward(self, coords_n, slice_w):
        feat = torch.cat([self.fourier(coords_n), coords_n,
                          slice_w @ self.slice_emb.weight], dim=-1)
        return self.net(feat)


class RelEncoder(nn.Module):
    def __init__(self, hidden, levels=REL_LEVELS):
        super().__init__()
        self.fourier = Fourier(levels)
        self.net = nn.Sequential(nn.Linear(3 * 2 * levels, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, offs_vox):
        return self.net(self.fourier(offs_vox))


class FusionBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=0.0)
        self.ln1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                nn.Linear(2 * hidden, hidden))
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, q, kv, key_padding_mask):
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.ln1(q + a)
        return self.ln2(q + self.ff(q))


class EncMTP(nn.Module):
    def __init__(self, n_slices, n_genes, hidden=HIDDEN, blocks=N_BLOCKS):
        super().__init__()
        self.tok = TokenEncoder(n_slices, hidden)
        self.rel = RelEncoder(hidden)
        self.blocks = nn.ModuleList([FusionBlock(hidden, N_HEADS) for _ in range(blocks)])
        self.head = nn.Linear(hidden, n_genes)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, 128), nn.SiLU(), nn.Linear(128, n_genes))
            for _ in range(AUX_K)])

    def forward(self, tok_feat, ctx_feat, rel_emb, key_mask):
        h = tok_feat.unsqueeze(1)
        kv = ctx_feat + rel_emb
        for blk in self.blocks:
            h = blk(h, kv, key_mask)
        h = h.squeeze(1)
        return h, self.head(h)

    def aux(self, h, rel_aux):
        outs = [head(torch.cat([h, rel_aux[:, k]], dim=-1))
                for k, head in enumerate(self.aux_heads)]
        return torch.stack(outs, dim=1)


def scenario_arrays(sc, bins, sec_rows):
    vis = [s for s in sc["visible"] if s in sec_rows]
    vr = np.concatenate([sec_rows[s] for s in vis])
    vr_sec = np.concatenate([np.full(sec_rows[s].size, i) for i, s in enumerate(vis)])
    tgt = sorted(sc["target"], key=int)
    tr = np.concatenate([sec_rows[s] for s in tgt])
    tr_sec = np.concatenate([np.full(sec_rows[s].size, i) for i, s in enumerate(tgt)])
    return vis, vr, vr_sec, tgt, tr, tr_sec


def context_indices(q_coords_um, slice_coords_um, k):
    """k nearest neighbours per slice; returns (n_q, n_slices*k)."""
    n_slices = len(slice_coords_um)
    out = np.empty((q_coords_um.shape[0], n_slices * k), dtype=np.int64)
    for t, pts in enumerate(slice_coords_um):
        tree = cKDTree(pts)
        _, nb = tree.query(q_coords_um, k=k, workers=4)
        out[:, t * k:(t + 1) * k] = nb
    return out


def train_scenario(sc, bins, sec_rows, dev, args, log):
    vis, vr, vr_sec, tgt, tr, tr_sec = scenario_arrays(sc, bins, sec_rows)
    n_slices = len(vis)
    X = np.ascontiguousarray(bins["X"][vr])
    coords = np.stack([bins["x"][vr], bins["y"][vr], bins["z"][vr]], 1).astype(np.float32)
    q_coords = np.stack([bins["x"][tr], bins["y"][tr], bins["z"][tr]], 1).astype(np.float32)
    slice_pts = [np.ascontiguousarray(coords[vr_sec == t]) for t in range(n_slices)]

    t0 = time.time()
    ctx = np.empty((vr.size, n_slices * K_PER_SLICE), dtype=np.int64)
    for t, pts in enumerate(slice_pts):
        tree = cKDTree(pts)
        own = vr_sec == t
        dist, nb = tree.query(coords, k=K_PER_SLICE + 1, workers=4)
        nb_k = nb[:, :K_PER_SLICE]
        nb_own = nb[:, 1:]  # nearest is the token itself for its own slice
        ctx[:, t * K_PER_SLICE:(t + 1) * K_PER_SLICE] = np.where(own[:, None], nb_own, nb_k)
    # ctx above is per-slice local; convert to pool positions
    offsets = np.concatenate([[0], np.cumsum([p.shape[0] for p in slice_pts])])
    ctx_pool = np.empty_like(ctx)
    for t in range(n_slices):
        ctx_pool[:, t * K_PER_SLICE:(t + 1) * K_PER_SLICE] = \
            ctx[:, t * K_PER_SLICE:(t + 1) * K_PER_SLICE] + offsets[t]
    ctx_slice = np.repeat(np.arange(n_slices), K_PER_SLICE)
    ctx_t = context_indices(q_coords, slice_pts, K_PER_SLICE)
    ctx_t_pool = np.empty_like(ctx_t)
    for t in range(n_slices):
        ctx_t_pool[:, t * K_PER_SLICE:(t + 1) * K_PER_SLICE] = \
            ctx_t[:, t * K_PER_SLICE:(t + 1) * K_PER_SLICE] + offsets[t]
    nn_sec = time.time() - t0

    mean = X.mean(0).astype(np.float32)
    std = np.maximum(X.std(0).astype(np.float32), 1e-3)
    Z = torch.from_numpy((X - mean) / std).to(dev)
    g_coords = torch.from_numpy(coords * 1000.0).to(dev)
    g_ctx = torch.from_numpy(ctx_pool).to(dev)
    g_ctx_slice = torch.from_numpy(ctx_slice).to(dev)
    slice_z_um = np.array([coords[vr_sec == t, 2].mean() * 1000.0 for t in range(n_slices)],
                          dtype=np.float32)
    g_slice_z_um = torch.from_numpy(slice_z_um).to(dev)

    model = EncMTP(n_slices, C.N_GENES).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    g_mean = torch.from_numpy(mean).to(dev)
    g_std = torch.from_numpy(std).to(dev)
    bs_um = 50.0
    lo = torch.from_numpy(np.array([bins["x"].min(), bins["y"].min(), bins["z"].min()],
                                   dtype=np.float32) * 1000.0).to(dev)
    hi = torch.from_numpy(np.array([bins["x"].max(), bins["y"].max(), bins["z"].max()],
                                   dtype=np.float32) * 1000.0).to(dev)
    sigma_um = float(np.median(np.diff(sorted(slice_z_um))) * 0.75) if n_slices > 1 else 150.0

    def tok_feats(cum):
        cn = 2.0 * (cum - lo) / (hi - lo) - 1.0
        w = torch.softmax(-0.5 * ((cum[:, 2:3] - g_slice_z_um[None, :]) / sigma_um) ** 2, dim=-1)
        return model.tok(cn, w)

    def rel_feats(delta_um):
        return model.rel(delta_um / bs_um)

    rng = np.random.default_rng(SEED)
    n = vr.size
    val_n = max(512, int(VAL_FRAC * n))
    perm = rng.permutation(n)
    val_idx = torch.from_numpy(np.sort(perm[:val_n]).astype(np.int64)).to(dev)
    train_idx = torch.from_numpy(np.sort(perm[val_n:]).astype(np.int64)).to(dev)

    @torch.no_grad()
    def predict_z(idx, ctx=None):
        model.eval()
        out = torch.empty((idx.numel(), C.N_GENES), dtype=torch.float32, device=dev)
        for c0 in range(0, idx.numel(), 8192):
            ii = idx[c0:c0 + 8192]
            tc = g_coords[ii]
            ci = (g_ctx[ii] if ctx is None else ctx[c0:c0 + 8192])
            cc = g_coords[ci]
            tf = tok_feats(tc)
            cf_ = tok_feats(cc.reshape(-1, 3)).reshape(ii.numel(), ci.shape[1], -1)
            re = rel_feats(cc - tc[:, None, :])
            _, pred = model(tf, cf_, re, None)
            out[c0:c0 + 8192] = pred
        return out

    def val_mse():
        p = predict_z(val_idx)
        return float(F.mse_loss(p, Z[val_idx]).item())

    best = (float("inf"), None)
    t_train = time.time()
    for step in range(args.steps):
        model.train()
        bi = train_idx[torch.randint(0, train_idx.numel(), (args.batch,), device=dev)]
        tc = g_coords[bi]
        ci = g_ctx[bi]
        valid = torch.rand(args.batch, ci.shape[1], device=dev) > TOKEN_DROPOUT
        if torch.rand(1, device=dev).item() < SLICE_MASK_PROB:
            hs = int(torch.randint(0, n_slices, (1,), device=dev).item())
            valid[:, g_ctx_slice == hs] = False
        need = valid.sum(1, keepdim=True)
        bad = need < (AUX_K + 2)
        if bad.any():
            valid[bad.expand_as(valid)] = True
        cc = g_coords[ci]
        tf = tok_feats(tc)
        cf_ = tok_feats(cc.reshape(-1, 3)).reshape(args.batch, ci.shape[1], -1)
        re = rel_feats(cc - tc[:, None, :])
        h, pred = model(tf, cf_, re, ~valid)
        yb = Z[bi]  # visible-bin training target (never canonical target_gt)
        loss = F.mse_loss(pred, yb)
        score = torch.where(valid, torch.rand_like(valid, dtype=torch.float32),
                            torch.full_like(valid, -1.0, dtype=torch.float32))
        pos = score.topk(AUX_K, dim=1).indices
        ar = torch.arange(args.batch, device=dev)[:, None]
        aux_pred = model.aux(h, rel_feats(cc[ar, pos] - tc[:, None, :]))
        aux_loss = F.mse_loss(aux_pred, Z[ci[ar, pos]])
        loss = loss + AUX_WEIGHT * aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            v = val_mse()
            if v < best[0]:
                best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()})
            log(f"[train] {sc['name']} step={step}/{args.steps} loss={loss.item():.4f} "
                f"val_mse={v:.4f} best={best[0]:.4f}")
    train_sec = time.time() - t_train
    if best[1] is not None:
        model.load_state_dict(best[1])
    final_val = val_mse()

    ctx_t_dev = torch.from_numpy(ctx_t_pool).to(dev)
    pred = predict_z(torch.arange(tr.size, device=dev), ctx=ctx_t_dev)
    pred = torch.clamp(pred * g_std + g_mean, min=0.0).cpu().numpy().astype(np.float32)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, dict(
        n_visible_bins=int(vr.size), n_target_bins=int(tr.size),
        n_tokens=int(vr.size), n_context_tokens=int(n_slices * K_PER_SLICE),
        neighbor_sec=round(nn_sec, 2), train_sec=round(train_sec, 2),
        best_val_mse=round(float(best[0]) if np.isfinite(best[0]) else -1, 5),
        final_val_mse=round(final_val, 5), n_params=int(n_params),
        visible_sections=vis, target_sections=tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    outdir = C.METHODS_ROOT / METHOD
    outdir.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(outdir / "run.log")
    status = C.Status(outdir / "status.json", METHOD)
    status.update(status="RUNNING", stage="load", pilot=args.pilot,
                  command=" ".join([sys.executable] + sys.argv))
    mon = C.GpuSampler()
    mon.start()

    def body():
        t0 = time.time()
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "4")))
        log(f"[env] host={__import__('os').uname().nodename} torch={torch.__version__} "
            f"cuda={torch.cuda.is_available()} dev={dev} steps={args.steps} batch={args.batch}")
        bins = C.load_bins()
        tg = C.load_target_gt()
        scs = C.select_scenarios(C.load_scenarios(), "k1_s25" if args.pilot else args.scenarios)
        sec_rows = {s: np.flatnonzero(bins["section"] == s) for s in C.SECTIONS}
        per, ver = {}, []
        for i, sc in enumerate(scs):
            ts = time.time()
            rows = C.scenario_rows(tg, sc)
            pred, info = train_scenario(sc, bins, sec_rows, dev, args, log)
            path = C.save_pred(METHOD, sc, tg, rows, pred)
            v = C.verify_pred(path, tg, rows, sc, METHOD)
            ver.append(v)
            info["sec"] = round(time.time() - ts, 2)
            info["n_rows"] = int(rows.size)
            per[sc["name"]] = info
            log(f"[pred] {sc['name']} rows={rows.size} train={info['train_sec']}s "
                f"val_mse={info['best_val_mse']} pass={v['pass']}")
            status.update(stage="predict", scenarios_done=i + 1, last_scenario=sc["name"],
                          scenarios=list(per), gpu_peak_mb=mon.peak_mb)
        mon.stop()
        total = time.time() - t0
        ok = all(v["pass"] for v in ver)
        metrics = dict(
            method=METHOD, status="DONE" if ok else "FAILED_VERIFY", pilot=args.pilot,
            model="coordinate token encoder (Fourier + soft slice emb, 2x cross-attn) + MTP K=4",
            lane=C.LANE, host=__import__("os").uname().nodename,
            slurm_job=__import__("os").environ.get("SLURM_JOB_ID"),
            n_scenarios=len(scs), scenarios=[s["name"] for s in scs],
            n_target_rows=int(sum(p["n_rows"] for p in per.values())),
            config=dict(hidden=HIDDEN, blocks=N_BLOCKS, heads=N_HEADS,
                        freq_levels=FREQ_LEVELS, k_per_slice=K_PER_SLICE,
                        aux_k=AUX_K, aux_weight=AUX_WEIGHT, token_dropout=TOKEN_DROPOUT,
                        slice_mask_prob=SLICE_MASK_PROB, steps=args.steps, batch=args.batch,
                        lr=LR, seed=SEED, loss="MSE in per-gene standardised space",
                        selection="best held-out visible-bin MSE (no PCC)"),
            compute=dict(cpu_cores=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 1)),
                         gpu_peak_mb=mon.peak_mb, gpu_util_peak_pct=mon.peak_util,
                         gpu_alloc_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if torch.cuda.is_available() else 0.0,
                         ram_peak_mb=round(C.ram_peak_mb(), 1), total_sec=round(total, 2)),
            per_scenario=per, verification=ver, finished_at=C.now())
        C.write_json(outdir / ("compute_metrics_pilot.json" if args.pilot else "compute_metrics.json"),
                     metrics)
        status.update(status="DONE" if ok else "FAILED_VERIFY", stage="done",
                      n_scenarios=len(scs), total_sec=round(total, 2),
                      n_target_rows=metrics["n_target_rows"], gpu_peak_mb=mon.peak_mb,
                      verification_all_pass=ok)
        C.update_runner_status(METHOD, dict(
            status="DONE" if ok else "FAILED_VERIFY", n_scenarios=len(scs),
            n_target_rows=metrics["n_target_rows"], total_sec=round(total, 2),
            gpu_peak_mb=mon.peak_mb, metrics=str(outdir / "compute_metrics.json")))
        log(f"[done] {len(scs)} scenarios rows={metrics['n_target_rows']} "
            f"total={total:.1f}s gpu_peak={mon.peak_mb}MB all_pass={ok}")

    C.run_guard(METHOD, status, log, body)


if __name__ == "__main__":
    main()
