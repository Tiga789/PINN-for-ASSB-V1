# D11-S5B low-target gate/sign diagnostic

This package performs a diagnostic-only analysis of existing D11-S5A predictions.
It does **not** launch training and does **not** promote any model to mainline.

## Why this stage exists

D11-S5A completed successfully, but its low-target candidates improved global MAE while worsening:

- `low_target`
- `low_target_le_2p75`

Therefore the next step is not 200ks confirmation. D11-S5B checks whether the failure is caused by:

1. low-target gate not activating on low-target points;
2. correction sign not being downward when prediction is too high;
3. duplicated mode-to-args mapping (`lowtarget_gate_probe` and `lowtarget_downward_mild` looked identical in S5A);
4. missing saved component arrays in `prediction.npz`.

## Files

```text
scripts/gv1_d11_s5b_lowtarget_gate_sign_analysis.py
scripts/run_gv1_d11_s5b_preflight_check.ps1
scripts/run_gv1_d11_s5b_lowtarget_gate_sign_analysis.ps1
README_D11_S5B_LOWTARGET_GATE_SIGN_DIAGNOSTIC.md
RUN_ORDER_D11_S5B.txt
```

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

D:\Anaconda\envs\torchgpu\python.exe -m compileall gv1 scripts
.\scripts\run_gv1_d11_s5b_preflight_check.ps1
.\scripts\run_gv1_d11_s5b_lowtarget_gate_sign_analysis.ps1
```

## Outputs

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5b_lowtarget_gate_sign_analysis
```

Important files:

```text
D11_S5B_summary.json
D11_S5B_RECOMMENDATION.md
D11_S5B_by_profile_gate_sign.csv
D11_S5B_segment_focus_metrics.csv
D11_S5B_component_gate_sign_metrics.csv
D11_S5B_mode_segment_summary.csv
D11_S5B_mode_component_summary.csv
D11_S5B_duplicate_mode_check.csv
```

## Interpretation

Do not continue to 200ks confirmation unless a future redesign shows direct improvement in both:

- `low_target` MAE
- `low_target_le_2p75` MAE

while preserving global MAE/correlation and not introducing high-voltage overshoot.
