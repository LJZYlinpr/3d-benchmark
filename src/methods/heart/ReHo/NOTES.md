# ReHo (Repeat Holding) — heart 10-slice

CPU-only copy baseline, adapted from `Workspace/merfish_test/ReHo/run_reho.py`.
For a hidden section the source is the visible slice with the nearest z (ties
-> lower/preceding); the expression of the same (ix, iy) bin is copied. When
that lattice bin is absent in the source slice, the Euclidean-nearest bin in
x/y of the same visible slice is copied. No interpolation, no model, no target
expression: values are measured visible-bin means, so sparsity/zeros are
preserved (nonzero fraction ~0.82, same as observed bins).

Reproduces the infra pilot exactly (prediction for k1_s25 byte-identical to
`pilot/pred_k1_s25_ReHo_pilot.npz`).
