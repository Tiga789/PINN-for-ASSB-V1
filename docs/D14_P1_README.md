# D14-P1 Evidence Boundary / README Boundary Package

## Purpose

This package implements **D14-P1** for QJW-2 / PINN-for-ASSB-V1.

D14-P1 is a documentation and evidence-boundary step. It does **not** train a model and does **not** modify the GV1/ASSB mainline.

It should be run after D14-P0 freeze audit has passed or produced only acceptable warnings.

## What D14-P1 does

It reads the D14-P0 output directory and generates:

```text
D14_P1_EVIDENCE_BOUNDARY_REPORT.json
D14_P1_EVIDENCE_BOUNDARY_REPORT.md
D14_P1_CLAIMS_MATRIX.csv
D14_P1_TERMINOLOGY_GUARDRAILS.csv
README_D14_P1_PATCH.md
D14_P1_RUN_SUMMARY.txt
D14_P1_OUTPUT_INDEX.json
```

The report fixes the current evidence boundary:

```text
ASSB:
ModelFin_112_deterministic_wrapper is an engineering wrapper / unified package.

XJTU:
D9.6/D9.5.1 + D12-S1K support non-outlier measured-current voltage replay / voltage-surrogate validation.

Battery-8:
Batch-1_2C_battery-8 remains flagged/excluded and should be treated as stress-test / outlier.

XJTU internal states:
cs_a / cs_c / phie / phis_c cannot be called experimental ground truth without additional P2D-consistent labels or external validation.

XJTU SOH:
SOH should come from original cycle/capacity records. The voltage soft-label generator should not fabricate SOH.
```

## How to run

From project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p1_evidence_boundary.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -P0Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary" `
  -AllowWarn
```

Use `-ReadmeOnly` if you want to scan only `README.md` instead of all Markdown files.

## How to interpret result

- `PASS`: evidence boundary is clean.
- `WARN`: usually acceptable if D14-P0 had acceptable warning or README contains wording that needs manual review.
- `FAIL`: stop before D14-P2. Fix documentation or mainline boundary issue first.

## After D14-P1

Proceed to **D14-P2** only after accepting this boundary. D14-P2 should build a unified XJTU generalization scorecard over global / segment / protocol / cell / candidate / outlier-aware metrics.
