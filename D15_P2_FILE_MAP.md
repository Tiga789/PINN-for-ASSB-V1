# D15-P2 file map

```text
configs/d15_p2_precision_benchmark_config.json
  Precision benchmark config: stricter thresholds, larger model/training budget, audit settings.

gv1/p2dlite_rg_nn_precision/__init__.py
  Package marker.

gv1/p2dlite_rg_nn_precision/audit.py
  Precision audit helpers: R2/NRMSE, per-profile metrics, boundary/outside fraction, transition audit, top-k errors.

scripts/d15_p2_selftest_precision_benchmark.py
  Synthetic selftest for precision audit code.

scripts/d15_p2_preflight.py
  Checks D15-P0 RG soft labels and required D15-P1 NN modules.

scripts/d15_p2_train_rg_precision_benchmark.py
  Wrapper around validated D15-P1 trainer using D15-P2 config and output aliases.

scripts/d15_p2_eval_rg_precision_benchmark.py
  Wrapper around validated D15-P1 evaluator; saves full-profile prediction NPZ files for audit.

scripts/d15_p2_precision_audit.py
  Runs per-profile/electrode/transition/top-k/cycle-level precision audit.

scripts/d15_p2_collect_scorecard.py
  Collects train/eval/audit into D15_P2_FINAL_SCORECARD.json.

scripts/d15_p2_pack_review.py
  Creates small review zip without model weights or prediction arrays.

scripts/d15_p2_run_all.ps1
  One-click PowerShell runner.

README_D15_P2.md
  User instructions.

D15_P2_MANIFEST.json
  Package manifest.
```
