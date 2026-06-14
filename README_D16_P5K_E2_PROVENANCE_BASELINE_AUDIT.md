# D16-P5K-E2 provenance + baseline-only audit

This package performs a no-training audit. It does not modify checkpoints.

It fixes the P5K-E diagnostic weakness by resolving the correct P5K-D model directory:

```text
model_generator_aligned_hard_cbar_ocp_residual
```

It then compares:

1. existing final P5K-C / P5K-D scorecards;
2. checkpoint/config/train-audit provenance;
3. hard-baseline-only theta/cs exact metrics against D15 ALL55 P2Dlite-RG soft labels.

Primary output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke2_provenance_baseline_only_audit\D16_P5K_E2_PROVENANCE_BASELINE_AUDIT_REPORT.md
```

Run smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5ke2_audit.ps1 `
  -LimitProfiles 4 `
  -ChunkSize 200000
```

Run full audit:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5ke2_audit.ps1 `
  -Models "P5K-C,P5K-D" `
  -ChunkSize 200000
```

If memory is tight:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5ke2_audit.ps1 `
  -Models "P5K-C,P5K-D" `
  -ChunkSize 100000
```

Check outputs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5ke2_outputs.ps1
```

The report is designed to answer:

- Did P5K-E misreport P5K-D checkpoint/audit as missing because of a path bug?
- Is P5K-D worse because the generator-aligned hard baseline is already wrong?
- Does the residual network improve or worsen the hard baseline?
- Should generator prior be used as a strong output anchor or only as weak audit/initializer/no-regression guard?
