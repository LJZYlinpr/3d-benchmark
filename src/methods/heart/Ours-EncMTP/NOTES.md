# Ours-EncMTP — heart 10-slice

Adapted from `Workspace/merfish_test/Ours-EncMTP/run_encmtp.py`, retrained per
scenario on visible sections only.

Token = Fourier(x,y,z) + soft slice embedding -> 2-layer MLP (256) -> 2
cross-attention blocks over 3 nearest visible bins per visible slice (21-27
context tokens) -> 238-gene head + K=4 MTP auxiliary heads (1.65M params).
MSE in per-gene standardised space (mean/std from visible bins only), context
token dropout 0.15 + whole-slice masking p=0.5, aux weight 0.5, AdamW lr 1e-3
cosine, 3000 steps, batch 1024. Model selection = best held-out (5% of visible
bins) MSE; no PCC anywhere. Hidden target bins are predicted from coordinates
+ visible context only; predictions clamped at 0.
