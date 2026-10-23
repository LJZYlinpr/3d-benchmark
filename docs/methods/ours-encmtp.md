# Ours-EncMTP

## Implementation

- 5-slice: `src/methods/merfish/Ours-EncMTP/run_encmtp_heart.py`
- Heart: `src/methods/heart/Ours-EncMTP/run_encmtp_heart.py`

## Environment

`ntf` (or `spateo` for the 5-slice adapter, depending on how the run was
launched).

## Model

- coordinate-aware encoder: Fourier(x,y,z) + soft slice embedding;
- 2-layer MLP(256);
- 2 cross-attention blocks with 15 visible context tokens (3 nearest visible
  bins per visible slice);
- main gene head + K=4 Multi-Token Prediction auxiliary heads;
- 2.34M parameters; 3,000 steps; batch 1,024; 5% visible-only validation;
  MSE-only checkpoint selection.

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/ntf/bin/python src/methods/merfish/Ours-EncMTP/run_encmtp_heart.py \
  --scenarios all --steps 3000
```

## Output

Target-only predictions; no volume.

## Leakage status

- No target-expression leakage found: target-X zero/NaN controls are bitwise
  unchanged.
- It uses target coordinates, which is allowed by fixed-coordinate but must be
  disclosed as oracle geometry.

## Known issues

- **P1 relative-offset unit bug**: coordinates are multiplied by 1000 (nm) but
  relative offsets are divided by 50 (µm). Exact 50 µm lattice offsets collapse
  the xy Fourier features. Fix `bs_um` to 50,000 nm or remove the `*1000` for
  relative offsets.
- **P0 provenance**: current `compute_metrics.json` covers only M3a; M1a–M2c
  came from a concurrent run. No checkpoint, per-scenario seed, or environment
  hash is preserved. Rerun all 7 in one isolated process before publishing.
- Old-vs-rerun M1a max difference 4.6 is consistent with a pre-fix inference
  coordinate bug, not target leakage.
