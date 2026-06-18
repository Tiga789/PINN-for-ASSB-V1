# D17-P4 report-only state audit package

This package adds the P4 frozen internal-state audit stage.

P4 freezes the P3.4/P3.4V candidate and runs observed-only inference/adaptation on replay profiles.  It does not train model weights and does not use `cs/theta/phie/phis` soft labels during inference/adaptation.  After prediction NPZ files are written, it loads D15 P2Dlite-RG soft labels only to compute report-only MAE/R2/bias/correlation and mechanism audits.

## Files

```text
gv1/d17_pinn/p4_state_audit.py
scripts/d17_p4_report_only_state_audit.py
scripts/d17_p4_inspect_scorecard.py
configs/d17_pinn_rebuild_p4_report_only_state_audit.json
docs/D17_P4_FILE_LIST_ACTUAL.txt
README_D17_P4_REPORT_ONLY_STATE_AUDIT.md
```

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_p4_report_only_state_audit.py `
  --config configs/d17_pinn_rebuild_p4_report_only_state_audit.json `
  --candidate_p34_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion" `
  --candidate_p34v_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34v_final_validation_voltage_polish" `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion/D17_P34_RESOLVED_P2DLITE_RG_SPEC_ALIGNED.json" `
  --checkpoint "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion/model/best_model_and_latents.pt" `
  --no_state_label_audit "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p4_report_only_state_audit" `
  --n_r 17 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --adaptation_steps 120 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_p4_inspect_scorecard.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p4_report_only_state_audit/D17_P4_SCORECARD.json"
```

## Key outputs

```text
D17_P4_SCORECARD.json
D17_P4_DECISION_REPORT.md
D17_P4_STATE_AUDIT_PROFILE_METRICS.csv
D17_P4_STATE_AUDIT_SPLIT_METRICS.csv
D17_P4_RADIAL_MECHANISM_AUDIT.csv
D17_P4_VOLTAGE_STATE_DECOMPOSITION.csv
D17_P4_FLAGGED_PROBE_REPORT.json
D17_P4_INFERENCE_MANIFEST.csv
00_freeze_candidate/D17_P4_FREEZE_MANIFEST.json
01_inference_predictions/<split>/*.npz
```

## Decision fields

```text
status
promotion_status
p5_ready
normal_frozen_test_state_r2
promotion_reasons
```

P4 `promotion_status=PASS` means the frozen-test report-only state R2 gates pass.  `promotion_status=REVIEW` means the protocol ran, but the internal states are not yet good enough for P5.
