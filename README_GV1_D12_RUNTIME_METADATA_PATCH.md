# GV1 D12 Runtime Metadata Model Patch

D12 already prepared metadata_off / metadata_on manifests and found that the metadata training backend was absent. This package adds the missing backend as separate, opt-in files.

## After covering into the project root

Run guardrail first:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_runtime_patch_guardrail.ps1"
```

Expected verdict:

```text
d12_runtime_metadata_patch_guardrail_pass
```

Prepare 1-profile smoke commands:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_prepare_runtime_onoff_commands.ps1" -ProfileLimit 1
```

Optional one-profile smoke training:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_runtime_metadata_smoke_pair.ps1"
```

Collect scorecard after smoke or training:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_collect_runtime_onoff_scorecard.ps1"
```

## Files added

```text
gv1/d12_metadata_runtime.py
scripts/gv1_train_conditioned_pinn_d12_metadata_runtime.py
scripts/gv1_d12_runtime_patch_guardrail.py
scripts/gv1_d12_prepare_runtime_onoff_commands.py
scripts/gv1_d12_collect_runtime_onoff_scorecard.py
scripts/run_gv1_d12_runtime_patch_guardrail.ps1
scripts/run_gv1_d12_prepare_runtime_onoff_commands.ps1
scripts/run_gv1_d12_runtime_metadata_smoke_pair.ps1
scripts/run_gv1_d12_collect_runtime_onoff_scorecard.ps1
```

## Mainline protection

This package does not overwrite:

```text
gv1/model.py
gv1/output_transform.py
gv1/profile_adaptive.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
```
