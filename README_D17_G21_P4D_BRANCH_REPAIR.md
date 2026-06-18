# D17-G2.1 P4D/random_walk branch failure isolation and repair

This package does not touch frozen-test labels and does not overwrite G2 trainer/model code.

Added files:

- `gv1/d17_g/g21_p4d_branch_tools.py`
- `scripts/d17_g21_p4d_branch_failure_isolation.py`
- `scripts/d17_g21_p4d_branch_repair.py`
- `scripts/d17_g21_inspect_summary.py`
- `configs/d17_g21_p4d_branch_repair.json`

Main idea: G2 failed in train-internal heldout on `Batch-5_random_walk_battery-8`, `semantic_branch=D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH`, target `theta_a`, with large inventory/phase bias. G2.1 first diagnoses this from existing G2 CSVs, then reruns G2-style training with:

- the known P4D failure profile pinned into fit-train;
- `min_fit_per_group=2` and `max_internal_per_group=1` for protocol+branch groups;
- inventory targets slightly up-weighted and phie not allowed to dominate;
- validation soft labels still report-only;
- frozen-test soft labels not read.

Enter G3 only if `D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json` has `status=PASS` and `g3_ready=true`.
