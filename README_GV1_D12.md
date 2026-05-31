# GV1 D12 Metadata On/Off Ablation Package

Purpose: prepare a separate `metadata_off` vs `metadata_on` ablation plan using the D11C2 enriched metadata manifest, without overwriting the D9.6/D9.5.1 mainline.

This package is conservative by design:

- It does not launch training.
- It does not modify `gv1/model.py`, `gv1/output_transform.py`, `gv1/losses.py`, `gv1/trainer.py`, or `scripts/gv1_train_conditioned_pinn.py`.
- It keeps B1_2C battery-8 flagged/excluded from the 23-profile mainline scope.
- It does not generate a direct 24-profile 200ks mainline command.

## Run preparation

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_prepare_metadata_on_off_ablation.ps1"
```

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan
```

Review:

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan\D12_RECOMMENDATION.md" -Raw
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan\d12_metadata_on_off_ablation_summary.json" -Raw
Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan\d12_guardrail_checklist.csv" | Format-Table check_id,status,description -AutoSize
```

## Expected verdict

```text
d12_metadata_on_off_ablation_plan_ready_no_mainline_overwrite
```

## Scorecard collector

After a future separate runtime metadata-input ablation has been run, collect on/off results with:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_collect_on_off_scorecard.ps1" `
  -MetadataOffDir "<metadata_off_results_dir>" `
  -MetadataOnDir "<metadata_on_results_dir>"
```

The collector does not train; it only compares available JSON metric files.
