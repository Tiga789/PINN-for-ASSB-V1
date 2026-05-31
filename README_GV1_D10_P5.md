# GV1 D10-P5 Regime Policy and D11 Plan

D10-P5 is a report-only package. It freezes the D10 conclusion, creates a flagged-profile registry for `B1_2C battery-8`, and prepares D11 route options.

It does **not** modify the D9.6 / D9.5.1 mainline files.

## Files added

```text
scripts/gv1_d10_p5_regime_policy_and_d11_plan.py
scripts/run_gv1_d10_p5_regime_policy_and_d11_plan.ps1
manifests/d10_p5_expected_outputs.json
README_GV1_D10_P5.md
docs/D10_P5_SCOPE.md
```

## Run

From the project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p5_regime_policy_and_d11_plan.ps1"
```

Optional strict mode:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p5_regime_policy_and_d11_plan.ps1" -Strict
```

## Output directory

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p5_regime_policy_d11_plan
```

## Main outputs

```text
D10_P5_RECOMMENDATION.md
d10_p5_regime_policy_summary.json
d10_p5_mainline_acceptance_checklist.csv
d10_p5_flagged_profile_registry.csv
d10_p5_d11_candidate_routes.csv
```

## Read results

```powershell
$D10P5 = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p5_regime_policy_d11_plan"

Get-Content "$D10P5\D10_P5_RECOMMENDATION.md" -Raw
Get-Content "$D10P5\d10_p5_regime_policy_summary.json" -Raw

Import-Csv "$D10P5\d10_p5_mainline_acceptance_checklist.csv" |
  Format-Table check_id,status,observed,action -AutoSize

Import-Csv "$D10P5\d10_p5_flagged_profile_registry.csv" |
  Format-Table batch_id,protocol,battery_id,flag_status,flag_reason -AutoSize

Import-Csv "$D10P5\d10_p5_d11_candidate_routes.csv" |
  Format-Table route_id,status,allowed,route_name,risk -AutoSize
```

## Expected successful verdict

```text
d10_p5_mainline_freeze_and_regime_policy_ready_for_d11
```

## Meaning

- Keep D9.6 / D9.5.1 as the current GV1 non-outlier mainline.
- Keep B1_2C battery-8 flagged/excluded.
- Adopt no D10-P3 battery-8 correction.
- Start D11 with a regime feature-distance audit, not with hard guards or direct 24-profile 200ks mainline claims.
