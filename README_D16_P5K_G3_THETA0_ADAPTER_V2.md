# D16-P5K-G3 v3 observed-only theta0 adapter audit

No training, no checkpoint loading, no model mutation.

This v3 package fixes three issues from v1/v2:

1. The PowerShell wrapper disables positional binding and rejects accidental trailing `\`, preventing `OutDir=\`.
2. The Python script no longer assumes the G1 by-profile CSV always contains `oracle_shift_a/c`. If those columns are absent, it falls back to the G0 baseline-repair by-profile CSV and reconstructs oracle shifts as `-theta_a0_error` and `-theta_c0_error`.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg3_theta0_adapter_v2_audit.ps1 `
  -AllowOverwrite `
  -G1ByProfile "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg1_MINI_EVIDENCE\D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv"
```

If automatic G0 fallback fails, provide it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg3_theta0_adapter_v2_audit.ps1 `
  -AllowOverwrite `
  -G1ByProfile "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg1_MINI_EVIDENCE\D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv" `
  -G0ByProfile "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg0_baseline_repair_audit\D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv"
```

Do not put `\` after the last argument. In PowerShell, line continuation is the backtick character: `` ` ``.

## Check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg3_outputs.ps1
```

## Output to paste

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg3_theta0_adapter_v2_audit\D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md
```


## v3 fix notes
- Guards against accidental PowerShell positional `OutDir=\` caused by a trailing backslash token.
- If the G1 by-profile CSV lacks `oracle_shift_a/oracle_shift_c`, the Python script falls back to G0 baseline-repair CSV `theta_a0_error/theta_c0_error` and computes `oracle_shift=-theta0_error`.
- Optional PowerShell parameter added: `-G0ByProfile`.
- Correct command syntax: do **not** append a trailing `\` after the CSV path.

## v3 additional fix
- Removed all use of `pandas.DataFrame.to_markdown()` so the script no longer requires the optional `tabulate` package.
