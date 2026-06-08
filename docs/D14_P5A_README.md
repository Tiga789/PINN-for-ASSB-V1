# D14-P5A XJTU P2Dlite NN Eval/Verify Fix

## Why P5A is needed

The D14-P5 training step succeeded and generated:

```text
ModelFin_D14_P5_p2dlite_nn_smoke/best.pt
EvalFin_D14_P5_p2dlite_nn_smoke/metrics_by_profile.csv
EvalFin_D14_P5_p2dlite_nn_smoke/predictions/<cell_uid>/prediction_sampled.npz
```

But the evaluator crashed while aggregating split metrics:

```text
ValueError: could not convert string to float: 'Batch-4'
```

The bug was in `aggregate_metrics()`: it attempted to convert metadata columns
such as `batch` and `protocol` to float.

P5A fixes only eval/verify. It does not retrain the model and does not regenerate
soft labels.

## Files changed

```text
gv1/softlabels_nn/xjtu_p2dlite_metrics.py
scripts/gv1_d14_p5a_eval_p2dlite_softlabel_nn.py
scripts/gv1_d14_p5_verify_outputs.py
scripts/run_gv1_d14_p5a_eval_verify_fix.ps1
configs/d14_p5a_eval_verify_fix_config.json
```

## Recommended command

Because your previous P5 run already produced `metrics_by_profile.csv` and all
prediction NPZ files, use repair-only mode first:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p5a_eval_verify_fix.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5_p2dlite_nn_smoke" `
  -RepairOnly `
  -AllowWarn
```

If repair-only fails because `metrics_by_profile.csv` is missing, rerun without
`-RepairOnly`; it will reload `best.pt` and regenerate the evaluation outputs.

## Expected repaired files

```text
EvalFin_D14_P5_p2dlite_nn_smoke/D14_P5_EVAL_REPORT.json
EvalFin_D14_P5_p2dlite_nn_smoke/metrics_by_split.csv
D14_P5_VERIFY_REPORT.json
D14_P5A_EVAL_VERIFY_console.log
D14_P5A_EVAL_stdout.log
D14_P5A_EVAL_stderr.log
D14_P5A_VERIFY_stdout.log
D14_P5A_VERIFY_stderr.log
```
