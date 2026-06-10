# D15-P4C file map

| File | Purpose |
|---|---|
| `configs/d15_p4c_batch56_remaining14_replay_config.json` | Target-cell list, default paths, resource-smoke settings. |
| `scripts/d15_p4c_utils.py` | Script-local helpers; intentionally under `scripts/`, not `gv1/`. |
| `scripts/d15_p4c_selftest.py` | Minimal package selftest. |
| `scripts/d15_p4c_preflight.py` | Finds raw `.mat` files for the missing Batch-5/6 14 cells. |
| `scripts/d15_p4c_build_batch56_replay_profiles.py` | Builds `solution_replay_profile.npz` for the 14 target cells. |
| `scripts/d15_p4c_audit_replay_profiles.py` | Audits replay profiles for time monotonicity, core fields, finite values. |
| `scripts/d15_p4c_softlabel_resource_smoke.py` | Resource smoke for the future P4D soft-label step. |
| `scripts/d15_p4c_collect_scorecard.py` | Builds final P4C scorecard. |
| `scripts/d15_p4c_pack_review.py` | Packs JSON/CSV review zip. |
| `scripts/d15_p4c_run_all.ps1` | One-click runner. |
