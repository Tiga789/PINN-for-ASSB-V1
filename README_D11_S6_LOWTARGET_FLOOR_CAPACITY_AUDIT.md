# D11-S6 low-target floor / model-capacity audit

This package does **not** launch training. It audits existing D11-S5C prediction files after D11-S5C showed that all amplitude-repair candidates improved global MAE but failed the low-target criterion.

## Purpose

D11-S6 answers:

1. Are low-target points still systematically over-predicted?
2. Is there an apparent voltage-floor or output-transform barrier?
3. Which voltage components dominate low-target predictions?
4. Should the next step be output-transform redesign, low-voltage anchor, or P2D-like/protocol-specific correction?

## Run order

From project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\run_gv1_d11_s6_preflight_check.ps1
.\scripts\run_gv1_d11_s6_lowtarget_floor_capacity_audit.ps1
```

## Output directory

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s6_lowtarget_floor_capacity_audit
```

## Main output files

```text
D11_S6_summary.json
D11_S6_RECOMMENDATION.md
D11_S6_candidate_vs_baseline.csv
D11_S6_candidate_decisions.csv
D11_S6_mode_segment_summary.csv
D11_S6_mode_component_summary.csv
D11_S6_component_lowtarget_audit.csv
D11_S6_output_transform_static_audit.json
```

## Decision rule

Do **not** proceed to 200 ks confirmation unless a candidate reduces both `low_target` and `low_target_le_2p75` MAE while preserving global, rest, and high-target metrics.

If D11-S6 confirms that low-target predictions remain far above the target range, the next step should be a redesign, not another amplitude sweep.
