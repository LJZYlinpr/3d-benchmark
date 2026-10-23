# PASTE-OT-Linear — heart 10-slice (sections 21-30, 50 um bins)

Adapted from `Workspace/merfish_test/PASTE-OT-Linear/run_paste_ot_linear.py`.
For each scenario the two visible slices bracketing the hidden block are aligned
with entropic fused Gromov-Wasserstein OT (KL expression cost, square_loss
spatial cost, alpha=0.1, epsilon=0.05, full-slice supports, uniform marginals).
Both slices are transported into the other frame, z-linearly blended at the
hidden section's depth, and hidden target bins take inverse-distance k=4
interpolation of the aligned clouds. Only visible sections feed the OT; target
expression is never read. Output: target bins only.

## Numerical deviation (documented)

The reference used `solver="PGD"`. On these 238-gene bins the KL cost is weak
(mean 0.07, max 1 after max-normalisation) and PGD returns the *uniform* plan
(POT warns "numerical errors at iteration 0"; plan entropy 17.44 == log(n*m)),
giving meaningless ~1.7 mm displacements and ~366 um map distances.
The identical objective is instead solved with `solver="PPA"`, max_iter=100,
tol=1e-7: plan entropy 9.5-10.1 (max 17.44), fgw_dist 0.034-0.036,
median cross-frame displacement ~360-400 um, hidden-bin map distance median
18-19 um (reference MERFISH 50um PASTE: median 30 um). Evidence:
`diag_solver.py`, `diag_eps.py`, `diag_ppa.py`.

Frozen config otherwise unchanged: alpha 0.1, epsilon 0.05, KL, square_loss,
knn 4, inverse-distance^2.
