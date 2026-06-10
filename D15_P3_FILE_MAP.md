# D15-P3 file map

```text
configs/d15_p3_batch2_applicability_config.json
configs/d15_p3_batch2_nn_smoke_config.json

gv1/p2dlite_rg_batch2/__init__.py
gv1/p2dlite_rg_batch2/utils.py
gv1/p2dlite_rg_batch2/batch2_io.py

scripts/d15_p3_selftest_batch2.py
scripts/d15_p3a_preflight_batch2.py
scripts/d15_p3a_discover_batch2.py
scripts/d15_p3a_build_batch2_replay_profiles.py
scripts/d15_p3b_select_batch2_representatives.py
scripts/d15_p3c_generate_batch2_rg_softlabels.py
scripts/d15_p3_collect_scorecard.py
scripts/d15_p3_pack_review.py
scripts/d15_p3_run_all.ps1

README_D15_P3.md
D15_P3_FILE_MAP.md
D15_P3_MANIFEST.json
```

This package depends on the already installed D15-P0 and D15-P1 modules/scripts:

```text
gv1/p2dlite_rg/*
scripts/d15_p0_radial_gradient_audit.py
gv1/p2dlite_rg_nn/*
scripts/d15_p1_*_rg_closedset_nn_smoke.py
```
