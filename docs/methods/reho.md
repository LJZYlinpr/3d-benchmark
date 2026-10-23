# ReHo

## Implementation

- Package: `src/methods/merfish/ReHo/run_reho_heart.py`
- So3D adapter: `src/methods/so3d/ReHo/run_reho_so3d.py`
- Original workspace reference: `Workspace/merfish_test/ReHo/run_reho.py`.

## Environment

`benchmark` (CPU only; no GPU required).

## Rule as implemented

For each target bin:

1. choose the visible source slice with the **nearest z** to the target slice;
   ties go to the lower z;
2. copy the visible expression vector at the same `(ix, iy)`;
3. if that bin is absent in the source, use the nearest visible xy bin in that
   source slice.

This is **nearest-visible Repeat Holding**, not the upstream
preceding/lower-z rule.

## Command template

```bash
python src/methods/merfish/ReHo/run_reho_heart.py --scenarios all
```

## Output

Target-only predictions; every predicted row is an exact copy of a visible row.

## Known deviations / caveats

- **Name mismatch**: upstream `run_reho.py` uses preceding/lower-z visible
  slices; 5/12 target-section source choices differ.
- No target expression leakage.
- No training/tuning; deterministic and reproducible.
- xy fallback is used for 9.51% of So3D rows; record fallback rate and distance.
- If the paper says "ReHo (preceding)", implement preceding-z or rename this
  baseline to "ReHo-nearest".
