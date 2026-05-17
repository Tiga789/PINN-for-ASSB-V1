# PINN-for-ASSB-V1 · ASSB-111 seed42-locked strict30 SOH route

This repository is being used for QJW-2 / ASSB strict30 SOH prediction experiments on top of the frozen ModelFin_107A four-state benchmark.

## Current ASSB-111 engineering route

The current route is **ModelFin_111_seed42_locked**:

- four electrochemical states (`cs_a`, `cs_c`, `phie`, `phis_c`) are protected by frozen `ModelFin_107A` evaluation outputs;
- the SOH branch uses the original `saturating_v2` head;
- SOH supervision uses only train cycles 5–139;
- validation cycles 140–159 are used only for visible checkpoint/candidate selection;
- test cycles 160–521 are held out until final reporting;
- cycle 522 is treated as partial/incomplete and is excluded from the main complete-cycle SOH metric.

This is a **seed42-locked engineering benchmark**, not a multi-seed robust claim. The seed is fixed to 42 because the original saturating_v2 seed42 run was the closest strict30/test70 candidate, while other seeds were unstable. Do not describe this route as a multi-seed robust predictor.

## Run command

After installing the corresponding新增文件 and修改文件 packages, run:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"

.\scripts\run_ModelFin111_saturating_v2_seed42locked.ps1 `
  -PythonExe $py `
  -ProjectRoot "." `
  -Seed 42 `
  -Device "cuda" `
  -RunOverdecayDiagnostics `
  -ForceClean
```

The wrapper runs a small pre-declared candidate grid and selects candidates using train/val visible metadata only. Final test metrics are reported after selection and must not be used to add new candidates or retune parameters.

## Expected outputs

```text
Data\assb111_seed42_locked
ModelFin_111_seed42_locked
EvalFin_111_seed42_locked_strict30_test70
EvalFin_111_seed42_locked_selection
ASSB111_seed42_locked_candidates
```

Key audit files:

```text
ModelFin_111_seed42_locked\train_summary.json
ModelFin_111_seed42_locked\leakage_audit.json
ModelFin_111_seed42_locked\seed42_locked_protocol_audit.json
EvalFin_111_seed42_locked_selection\selected_candidate.json
EvalFin_111_seed42_locked_selection\selection_audit.json
```

## Reporting boundary

Acceptable wording:

> ASSB-111 seed42-locked strict30 engineering benchmark.

Avoid:

> multi-seed robust SOH predictor;
> StageB-equivalent full-cycle calibration;
> test-selected final model.
