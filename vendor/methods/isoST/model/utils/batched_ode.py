"""Batched fixed-step euler + discrete adjoint for isoST training.

GPU-batching reorganization (2026-08-20, registered in benchmark/METHOD_CHANGES.md).
Replaces the per-slice-pair sequential `odeint_adjoint(..., method='euler')` calls in
IsoST._compute_total_loss with ONE batched integration over all pairs of a training
step. The math is exactly torchdiffeq's (upstream code, unmodified):

  forward  : y_{k+1} = y_k + dt_k * f(t_k, y_k)       (dt per pair)
  backward : torchdiffeq _impl/adjoint.py (euler): per interval i (top-down) one euler
             step of the augmented dynamics with dt_step = t[i-1]-t[i]; the vjp is
             computed via autograd.grad. Grouping pairs with the same dt reproduces the
             per-pair result bit-for-bit:
               vjp = autograd.grad(f(y_i), (y_i, params), grad_outputs=-dt*adj)
               adj_{i-1} = adj_i - vjp_y ;  param_grad -= vjp_p

Why per-pair arithmetic is bit-identical:
  - variable-size batches (torch_geometric batch semantics): per-graph edge_index is
    computed with ONE torch_cluster.knn_graph call with a per-graph batch vector, so
    each pair sees exactly its own points as in the un-batched code (no padding rows
    that could enter the KNN or message passing),
  - dt is applied per pair,
  - the SDE noise (PCooStep/PUnetStep raw draws) is drawn HERE in the exact upstream
    pair-major order (forward: pairs 0..B-1, steps bottom-up; backward: pairs B-1..0,
    steps top-down) and injected, so the global RNG stream advances exactly as in the
    un-batched code.

Memory: for the formal 54-slice config the per-step batch is split into
consecutive pair chunks (_chunk_pairs, <= CHUNK_POINTS points each). Pairs share
no edges and no per-pair tensors, so chunked and un-chunked values agree up to
float32 accumulation-order noise only (see module-level comment).
"""
import os

import torch
from torch_cluster import knn_graph

# 2026-08-20 campaign: the formal 54-slice config (1M spots) exceeds 24 GB GPU
# memory when ALL slice pairs of one training step are batched together. Pairs
# are mutually independent (per-pair KNN via the batch vector; no cross-pair
# edges; params are the only shared tensors), so each step is computed in
# consecutive PAIR CHUNKS. Per-pair values and RNG draws are unchanged (noise
# is pre-drawn in upstream pair-major order and only indexed per pair); only
# the GEMM/message-passing grouping changes, i.e. float32 accumulation-order
# noise of the same class as the original batching (user ruling 2026-08-20:
# not precision loss). Registered in METHOD_CHANGES.md.
#
# A/B (2026-08-20): ONE PAIR PER CHUNK reproduces the un-batched per-pair
# semantics most closely — 3-epoch smoke vs single-batch: generated-point
# values agree to ~1e-4 relative (vs ~0.7 for multi-pair 120k chunks): the
# residual is the amplification of ~1e-7 float32 step noise through the
# hard-KNN SDE dynamics, inherent to GPU float32 runs. PAIR_CHUNKS (default
# on) forces one pair per chunk; ISOST_CHUNK_POINTS / ISOST_PAIR_CHUNKS env
# overrides exist for A/B verification only.
CHUNK_POINTS = int(os.environ.get("ISOST_CHUNK_POINTS", "120000"))
PAIR_CHUNKS = os.environ.get("ISOST_PAIR_CHUNKS", "1") == "1"

# A/B debug hook (verification only): dumps per-step diagnostics when set.
_DEBUG_FILE = os.environ.get("ISOST_DEBUG_CHUNK")
if _DEBUG_FILE:
    import hashlib
    _dbg_fh = open(_DEBUG_FILE, "a")


def _dbg(tag, *vals):
    if not _DEBUG_FILE:
        return
    h = hashlib.sha1()
    for v in vals:
        h.update(str(float(v)).encode())
    _dbg_fh.write(f"{tag} {h.hexdigest()[:12]} {' '.join(f'{float(v):.9e}' for v in vals)}\n")
    _dbg_fh.flush()


def _chunk_pairs(offsets, pair_ids):
    """Split pair ids into consecutive chunks of <= CHUNK_POINTS total points.

    PAIR_CHUNKS (formal campaign default): one pair per chunk — reproduces the
    un-batched per-pair GEMM shapes and keeps the chunked-vs-unbatched
    divergence at the float32-noise floor (see module header A/B note).
    """
    if PAIR_CHUNKS:
        return [[k] for k in pair_ids]
    chunks, cur, cur_pts = [], [], 0
    for k in pair_ids:
        nk = offsets[k + 1] - offsets[k]
        if cur and cur_pts + nk > CHUNK_POINTS:
            chunks.append(cur)
            cur, cur_pts = [], 0
        cur.append(k)
        cur_pts += nk
    if cur:
        chunks.append(cur)
    return chunks


class BatchedEulerAdjoint(torch.autograd.Function):
    """Batched euler trajectories with the exact torchdiffeq discrete adjoint.

    Tensor inputs: u0_cat (sum N_k, D) — one segment per pair, in pair order;
                   *params — the model parameters (tensor inputs, as in torchdiffeq's
                   odeint_adjoint, so the output carries requires_grad).
    Non-tensor inputs: model, offsets (tuple of B+1 ints, cumulative N_k),
                       dt_list (tuple of B 1-D tensors, dt per pair),
                       T_list (tuple of B ints, grid length per pair).
    Output:           y_final_cat (sum N_k, D) — each pair's trajectory at its real
                      final step, same segmentation as u0_cat.
    """

    @staticmethod
    def forward(ctx, model, u0_cat, offsets, dt_list, T_list, *params):
        ctx.model = model
        ctx.params = tuple(params)
        ctx.offsets = tuple(offsets)
        ctx.dt_list = tuple(dt_list)
        ctx.T_list = tuple(T_list)
        B = len(T_list)
        D = u0_cat.shape[1]
        device = u0_cat.device

        with torch.no_grad():
            # --- pre-draw the SDE noise in the exact upstream pair-major order ---
            noise_coo = []
            noise_seq = []
            for k in range(B):
                nk = offsets[k + 1] - offsets[k]
                Tk = T_list[k]
                nc = []
                ns = []
                for s in range(Tk - 1):
                    nc.append(torch.randn(nk, 3, device=device))
                    ns.append(torch.randn(nk, D - 3, device=device))
                noise_coo.append(nc)
                noise_seq.append(ns)

            # --- batched explicit euler (same per-pair arithmetic as upstream;
            #     pair chunks keep peak memory within the GPU budget) ---
            traj = [[u0_cat[offsets[k]:offsets[k + 1]]] for k in range(B)]
            for s in range(max(T_list) - 1):
                active = [k for k in range(B) if T_list[k] > s + 1]
                if not active:
                    continue
                for chunk in _chunk_pairs(offsets, active):
                    xs = torch.cat([traj[k][s] for k in chunk], dim=0)
                    bvec = torch.cat([torch.full((offsets[k + 1] - offsets[k],), i,
                                                 dtype=torch.long, device=device)
                                      for i, k in enumerate(chunk)])
                    with torch.no_grad():
                        edge_index = knn_graph(xs[:, :3].cpu(), model.topk,
                                               batch=bvec.cpu(), loop=False).to(device)
                    BatchedEulerAdjoint._dbg_edges(f"F{s}", edge_index, 200)
                    f = model.forward_batched(
                        None, xs, bvec, edge_index,
                        noise_coo=torch.cat([noise_coo[k][s] for k in chunk], dim=0),
                        noise_seq=torch.cat([noise_seq[k][s] for k in chunk], dim=0),
                    )
                    off = 0
                    for i, k in enumerate(chunk):
                        nk = offsets[k + 1] - offsets[k]
                        traj[k].append(traj[k][s] + dt_list[k][s] * f[off:off + nk])
                        off += nk

        ctx.traj = traj
        final = torch.cat([traj[k][-1] for k in range(B)], dim=0)
        if _DEBUG_FILE:
            for k in range(B):
                _dbg(f"fwd_pair{k}_mean", float(traj[k][-1].float().mean()))
        return final

    @staticmethod
    def _dbg_edges(step, edge_index, first_k):
        """Hash the first `first_k` edges of a step's KNN output (A/B debug)."""
        if not _DEBUG_FILE:
            return
        h = hashlib.sha1()
        idx = edge_index[:, :first_k].cpu()
        h.update(idx.numpy().tobytes())
        _dbg_fh.write(f"edges_s{step}_{first_k} {h.hexdigest()[:12]}\n")
        _dbg_fh.flush()

    @staticmethod
    def backward(ctx, grad_out):
        model = ctx.model
        params = ctx.params
        offsets = ctx.offsets
        dt_list = ctx.dt_list
        T_list = ctx.T_list
        traj = ctx.traj
        B = len(T_list)
        D = grad_out.shape[1]
        device = grad_out.device

        with torch.no_grad():
            # --- pre-draw the adjoint-phase noise in the exact upstream order ---
            # (autograd visits trajectory functions in reverse creation order: pairs
            #  B-1..0, intervals top-down; one func eval per interval)
            noise_coo = [None] * B
            noise_seq = [None] * B
            for k in range(B - 1, -1, -1):
                nk = offsets[k + 1] - offsets[k]
                Tk = T_list[k]
                nc = [None] * (Tk - 1)
                ns = [None] * (Tk - 1)
                for s in range(Tk - 2, -1, -1):
                    nc[s] = torch.randn(nk, 3, device=device)
                    ns[s] = torch.randn(nk, D - 3, device=device)
                noise_coo[k] = nc
                noise_seq[k] = ns

            # upstream initializes the running adjoint with the loss grad at the
            # final step (stride=1: all per-interval grad_y adds are zero)
            adj = [grad_out[offsets[k]:offsets[k + 1]].clone() for k in range(B)]
            vjp_params = [torch.zeros_like(p) for p in params]

            for s in range(max(T_list) - 2, -1, -1):
                active = [k for k in range(B) if T_list[k] > s + 1]
                if not active:
                    continue
                # mirror torchdiffeq's augmented_dynamics: the func eval (and its vjp)
                # runs with grad ENABLED inside the otherwise no-grad backward.
                # Pair chunks: per-pair adj/vjp identical; param grads accumulate
                # across chunks in deterministic order.
                for chunk in _chunk_pairs(offsets, active):
                    with torch.enable_grad():
                        ys = torch.cat([traj[k][s + 1].detach().requires_grad_(True)
                                        for k in chunk], dim=0)
                        bvec = torch.cat([torch.full((offsets[k + 1] - offsets[k],), i,
                                                     dtype=torch.long, device=device)
                                          for i, k in enumerate(chunk)])
                        with torch.no_grad():
                            edge_index = knn_graph(ys[:, :3].cpu(), model.topk,
                                                   batch=bvec.cpu(), loop=False).to(device)
                        BatchedEulerAdjoint._dbg_edges(f"B{s}", edge_index, 200)
                        f_eval = model.forward_batched(
                            None, ys, bvec, edge_index,
                            noise_coo=torch.cat([noise_coo[k][s] for k in chunk], dim=0),
                            noise_seq=torch.cat([noise_seq[k][s] for k in chunk], dim=0),
                        )
                        g_out = torch.cat([-dt_list[k][s] * adj[k] for k in chunk], dim=0)
                        vjp_y, *vjp_p = torch.autograd.grad(
                            f_eval, (ys,) + params, grad_outputs=g_out, allow_unused=True)
                    off = 0
                    for i, k in enumerate(chunk):
                        nk = offsets[k + 1] - offsets[k]
                        # upstream: adj_new = adj + (-dt)*vjp_y with vjp_y = grad(f,y,-adj);
                        # with grad_outputs=-dt*adj: vjp_y' = dt*vjp_y -> adj_new = adj - vjp_y'
                        adj[k] = adj[k] - vjp_y[off:off + nk]
                        off += nk
                    for p_acc, vp in zip(vjp_params, vjp_p):
                        if vp is not None:
                            # upstream: p_acc += (-dt)*grad(f,p,-adj) = +dt*adj*df/dp
                            # with grad_outputs=-dt*adj: vjp_p' = -dt*adj*df/dp -> p_acc -= vjp_p'
                            p_acc -= vp

        if _DEBUG_FILE:
            _dbg("bwd_p0_mean", float(vjp_params[0].float().mean()))
            for k in range(B):
                _dbg(f"bwd_adj{k}_mean", float(adj[k].float().mean()))
        return (None, torch.cat(adj, dim=0), None, None, None) + tuple(vjp_params)
