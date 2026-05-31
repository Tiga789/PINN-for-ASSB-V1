# D12 TRUE SMOKE rescue scorecard fix

This package fixes the D12 TRUE SMOKE scorecard collection issue.

The previous collector marked runs as `read_error` when `d10_voltage_metrics.json`
was absent, even if `prediction.npz` had been produced by the 50-epoch smoke run.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_rescue_true_smoke_metrics_and_scorecard.ps1"
```

Outputs:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_runtime_metadata_true_smoke_scorecard_rescued
```

This script does not launch training. It only reads existing TRUE_SMOKE run folders
and computes voltage metrics from `prediction.npz` if present.
