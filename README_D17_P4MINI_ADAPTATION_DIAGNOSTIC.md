# D17-P4-mini adaptation diagnostic

This package adds a **minimal diagnostic** only. It does not modify the P3.4/P3.4V candidate, the model checkpoint, the P4 state-audit engine, or any training script.

## Purpose

The first P4 smoke used only `adaptation_steps=10` and produced very poor frozen-test state R². This diagnostic reruns the same first normal frozen-test profile twice:

- short run: 10 observed-only adaptation steps
- long run: 120 observed-only adaptation steps

It then compares report-only state R². If 120 steps do not recover the profile, formal P4 is not worth running for promotion.

## Files

```text
scripts/d17_p4mini_adaptation_diagnostic.py
scripts/d17_p4mini_inspect_summary.py
configs/d17_pinn_rebuild_p4mini_adaptation_diagnostic.json
docs/D17_P4MINI_FILE_LIST_ACTUAL.txt
README_D17_P4MINI_ADAPTATION_DIAGNOSTIC.md
```

## Run

```powershell
python scripts\d17_p4mini_adaptation_diagnostic.py `
  --config configs/d17_pinn_rebuild_p4mini_adaptation_diagnostic.json `
  --candidate_p34_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion" `
  --candidate_p34v_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34v_final_validation_voltage_polish" `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion/D17_P34_RESOLVED_P2DLITE_RG_SPEC_ALIGNED.json" `
  --checkpoint "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion/model/best_model_and_latents.pt" `
  --no_state_label_audit "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p4mini_adaptation_diagnostic" `
  --short_steps 10 `
  --long_steps 120 `
  --n_r 17 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_p4mini_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p4mini_adaptation_diagnostic/D17_P4MINI_ADAPTATION_DIAGNOSTIC_SUMMARY.json"
```

## Decision fields

Read these fields from the console or summary JSON:

```text
status
decision
recommendation
long_step_r2_mean
long_step_r2_min
failed_min_recovery_keys
```

Interpretation:

```text
RECOVERED_ENOUGH_TO_RUN_FORMAL_P4
  120-step adaptation rescued the same frozen-test profile enough to justify formal P4.

PARTIAL_RECOVERY_DIAGNOSE_BEFORE_FORMAL_P4
  Some improvement, but not enough for promotion. Inspect failed targets before formal P4.

STOP_FORMAL_P4_FIX_STATE_ALIGNMENT
  120-step adaptation did not rescue state alignment. Do not run formal P4 for promotion.
```

## Boundary

Soft labels are loaded only after prediction files are written, for report-only metrics. This script does not perform training, checkpoint selection, rule design, or candidate modification.
