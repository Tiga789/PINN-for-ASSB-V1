# D14-P5 Expected Outputs

Output root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5_p2dlite_nn_smoke
```

Expected files:

```text
D14_P5_NN_SMOKE_console.log
D14_P5_MANIFEST_stdout.log
D14_P5_MANIFEST_stderr.log
D14_P5_TRAIN_stdout.log
D14_P5_TRAIN_stderr.log
D14_P5_EVAL_stdout.log
D14_P5_EVAL_stderr.log
D14_P5_VERIFY_stdout.log
D14_P5_VERIFY_stderr.log

D14_P5_MANIFEST_REPORT.json
D14_P5_MANIFEST_CHECKS.csv
D14_P5_SOFTLABEL_NN_MANIFEST.csv
D14_P5_SOFTLABEL_NN_MANIFEST.json
D14_P5_VERIFY_REPORT.json

ModelFin_D14_P5_p2dlite_nn_smoke/
  best.pt
  feature_stats.json
  training_summary.json
  loss_history.csv

EvalFin_D14_P5_p2dlite_nn_smoke/
  D14_P5_EVAL_REPORT.json
  metrics_by_profile.csv
  metrics_by_split.csv
  predictions/<cell_uid>/prediction_sampled.npz
```

A successful smoke should at least prove:

```text
1. all profiles are readable;
2. train/val/test splits exist;
3. model forward/backward succeeds;
4. output shapes are theta_a/theta_c = (N, 17), phie/phis_c = (N,);
5. metrics files are generated;
6. no SOH is generated.
```
