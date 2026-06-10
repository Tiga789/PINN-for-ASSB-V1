# D15-P0 file map

| Path | Purpose | Overwrites old mainline? |
|---|---|---:|
| `configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json` | P2Dlite-RG prior and audit thresholds | No |
| `gv1/p2dlite_rg/radial_solver.py` | Isolated zero-mean implicit spherical FVM RG solver | No |
| `gv1/p2dlite_rg/audit.py` | Radial-gradient audit metrics | No |
| `gv1/p2dlite_rg/io_utils.py` | NPZ discovery/loading helpers | No |
| `scripts/d15_p0_preflight.py` | Source/output no-overwrite checks | No |
| `scripts/d15_p0_radial_gradient_audit.py` | Audit old and new labels | No |
| `scripts/d15_p0_generate_p2dlite_rg_softlabels.py` | Generate new RG labels from source P2Dlite labels | No |
| `scripts/d15_p0_compare_radial_audits.py` | Compare old vs RG audit | No |
| `scripts/d15_p0_run_all.ps1` | PowerShell wrapper for D15-P0 | No |
| `scripts/d15_p0_selftest_rg_solver.py` | Synthetic solver self-test | No |
