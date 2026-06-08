# D14-P5B Expected Outputs

Output root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision
```

Expected files:

```text
D14_P5B_CLOSEDSET_PRECISION_console.log
D14_P5B_MANIFEST_REPORT.json
D14_P5B_CLOSEDSET_MANIFEST.csv
D14_P5B_CLOSEDSET_MANIFEST.json
D14_P5B_MANIFEST_CHECKS.csv
D14_P5B_VERIFY_REPORT.json

ModelFin_D14_P5B_8cell_closedset_precision/
  best.pt
  feature_stats.json
  tensor_memory_summary.json
  training_summary.json
  loss_history.csv

EvalFin_D14_P5B_8cell_closedset_precision/
  D14_P5B_EVAL_REPORT.json
  metrics_by_profile.csv
  metrics_by_batch.csv
  metrics_by_protocol.csv
  metrics_global.json
  predictions/<cell_uid>/prediction_sampled.npz
```
