# D17-G6.5 exact provenance replay test

Purpose: verify whether the P4D/GEO soft-label inventory trajectory can be exactly replayed from the recorded provenance/config and the copied soft-label input current. This is **not** training and does **not** run the radial solver.

Inputs:
- `D17_G64_PROFILE_PROVENANCE_DETAILS.json` from G6.4.
- Local D15-P4D config and prior JSON.
- Selected P4D/GEO soft-label NPZ files.

Pass gate:
- `exact_replay_ready = true`
- `patch_ready = true`
- `min_deployable_formula_r2 >= 0.999`

If it fails, stop: do not train, do not patch, and inspect `D17_G65_FORMULA_CANDIDATE_METRICS.csv`.
