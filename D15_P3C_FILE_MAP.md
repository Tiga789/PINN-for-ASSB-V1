# D15-P3C file map

| File | Purpose |
|---|---|
| `configs/d15_p3c_batch2_15cell_applicability_config.json` | P3C stage paths, thresholds, generation parameters |
| `configs/d15_p3c_batch2_15cell_nn_config.json` | 15-cell NN benchmark config |
| `configs/d15_p3c_boundary_repair_config.json` | 15-cell theta projection repair config |
| `gv1/p2dlite_rg_batch2_15cell/utils.py` | Small JSON/CSV helpers |
| `scripts/d15_p3c_selftest_15cell.py` | Lightweight dependency/selftest |
| `scripts/d15_p3c_make_batch2_all15_manifest.py` | Select all 15 Batch-2 replay profiles |
| `scripts/d15_p3c_generate_batch2_15cell_rg_softlabels.py` | Generate 15-cell P2Dlite-RG labels |
| `scripts/d15_p3c_collect_scorecard.py` | Collect final P3C scorecard |
| `scripts/d15_p3c_pack_review.py` | Pack lightweight review zip |
| `scripts/d15_p3c_run_all.ps1` | One-click P3C workflow |
