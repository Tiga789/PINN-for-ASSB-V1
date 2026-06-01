# D11-S9 trainable localized P2D-like correction head

## Purpose

D11-S8 showed that a P2D-like downward transport-deficit correction can improve low-target segments, but fixed post-hoc amplitudes damage global voltage metrics.  D11-S9 tests a more localized deterministic trainable correction head without modifying the GV1 mainline.

The correction is fitted from existing baseline `prediction.npz` files:

```text
V_corr = V_base - deficit_head(features)
deficit_head >= 0
```

The features use only replay/predicted quantities such as low-voltage gate, discharge gate, current magnitude, time/SOC proxy, OCV-like baseline and predicted voltage.  The target for fitting is `max(V_base - V_true, 0)` with stronger weight on `low_target` / `low_target_le_2p75` and preservation weights on normal/rest/high-voltage regions.

This is **not** a full P2D solver and does **not** overwrite `gv1/output_transform.py` or `scripts/gv1_train_conditioned_pinn.py`.

## Run order

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

D:\Anaconda\envs\torchgpu\python.exe -m compileall gv1 scripts

.\scripts\run_gv1_d11_s9_preflight_check.ps1
.\scripts\run_gv1_d11_s9_trainable_p2dlike_correction.ps1
.\scripts\run_gv1_d11_s9_collect_scorecard.ps1
```

## Outputs

Prediction root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s9_trainable_p2dlike_correction
```

Scorecard root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s9_trainable_p2dlike_correction_scorecard
```

Main files:

```text
D11_S9_scorecard_summary.json
D11_S9_RECOMMENDATION.md
D11_S9_mode_summary.csv
D11_S9_mode_split_summary.csv
D11_S9_mode_segment_summary.csv
D11_S9_mode_split_segment_summary.csv
D11_S9_global_vs_lowtarget_tradeoff.csv
D11_S9_candidate_decisions.csv
D11_S9_component_summary.csv
```

## Promotion rule

A candidate can advance only if:

```text
low_target MAE decreases by at least 20 mV
low_target_le_2p75 MAE decreases by at least 20 mV
global MAE does not increase by more than 5 mV
all-corr does not drop more than 0.005
rest/high-target segments remain stable
```

If no candidate passes, do not run 200ks confirmation; redesign with protocol-specific adapter or integrate the correction inside the voltage model.
