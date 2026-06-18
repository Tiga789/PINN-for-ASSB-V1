# D17-P3.4V final validation corrected-MAE polish

Purpose: resolve the last P3.4 blocker, `validation corrected voltage MAE > target 0.060 V`, before entering P4-pre.

This patch is intentionally small. It does **not** repackage or replace the P3.3/P3.4 training stack. It adds an observed-voltage-only validation polish step that reads existing P3.4 prediction NPZ files and fits a tiny smooth residual from `V_exp - V_pred` under strict residual budgets.

No state soft labels are loaded. Forbidden arrays remain: `cs_a`, `cs_c`, `theta_a`, `theta_c`, `phie`, `phis_c`, `theta0_oracle`, `oracle_shift`.

## Files

```text
gv1/d17_pinn/voltage_polish.py
scripts/d17_p34v_validation_voltage_polish.py
scripts/d17_p34v_inspect_summary.py
configs/d17_pinn_rebuild_p34v_validation_voltage_polish.json
docs/D17_P34V_FILE_LIST_ACTUAL.txt
README_D17_P34V_FINAL_VALIDATION_POLISH.md
```

## Run

```powershell
python scripts\d17_p34v_validation_voltage_polish.py `
  --config configs/d17_pinn_rebuild_p34v_validation_voltage_polish.json `
  --p34_out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34v_final_validation_voltage_polish"
```

Inspect:

```powershell
python scripts\d17_p34v_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34v_final_validation_voltage_polish/D17_P34V_FINAL_VALIDATION_POLISH_SUMMARY.json"
```

Pass-to-P4-pre fields:

```text
status = PASS
promotion_status = PASS
p4_ready = true
validation_mae_polished_V <= 0.060
residual_total_abs_mean_V <= 0.035
residual_total_abs_max_V <= 0.100
```

If `p4_ready=true`, freeze this P3.4V wrapper before P4-pre report-only state audit. Do not tune it using soft-label state metrics.
