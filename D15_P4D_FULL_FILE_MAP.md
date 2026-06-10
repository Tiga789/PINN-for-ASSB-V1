# D15-P4D full file map

| File | Purpose |
|---|---|
| `configs/d15_p4d_full_remaining14_config.json` | Paths, target cells, generation defaults, monitor thresholds. |
| `scripts/d15_p4d_full_selftest.py` | Verifies required scripts and existing `gv1.p2dlite_rg.radial_solver` import. |
| `scripts/d15_p4d_full_generate_one_rg_softlabel.py` | Generates one full P2Dlite-RG soft-label profile. |
| `scripts/d15_p4d_full_collect_generation_report.py` | Aggregates per-cell status into generation report. |
| `scripts/d15_p4d_full_collect_scorecard.py` | Combines generation and radial audit into final scorecard. |
| `scripts/d15_p4d_full_pack_review.py` | Packs JSON/CSV/logs only, excluding large NPZ files. |
| `scripts/d15_p4d_full_run_all.ps1` | Fanout/resume runner with resource monitoring. |
