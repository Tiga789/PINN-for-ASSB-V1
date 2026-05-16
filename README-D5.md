# PINN-for-ASSB-V1

QJW-2 / PINN-for-ASSB-V1 is a PyTorch/CUDA adaptation of PINNSTRIPES-style battery PINN surrogates for an NMC811||Li-In all-solid-state battery (ASSB). The current model backbone is an effective single-particle model (effective SPM) with fixed positive/negative electrode identities:

- `c`: NMC811 composite positive electrode.
- `a`: Li-In/In negative electrode represented as an equivalent pseudo-particle / diffusion length.
- `I(t)` changes flux direction and potential/ohmic signs only; it does not switch material identity, OCP functions, diffusivities, or electrode labels.

## Current status after ASSB-D5

The current best **five-target** result is a **hybrid benchmark**, not yet a single unified model:

```text
cs_a / cs_c / phie / phis_c: ModelFin_107A
SOH:                         ModelFin_110_stageB
```

This should be referred to as:

```text
ModelFin_107A four-state benchmark + ModelFin_110_stageB SOH aging-head benchmark
```

Do **not** describe the current result as a single end-to-end `ModelFin_110` model. The next stage, **ASSB-D6**, is reserved for building a single model that jointly outputs:

```text
cs_a, cs_c, phie, phis_c, SOH
```

## Current benchmark artifacts

| Artifact | Role | Notes |
|---|---|---|
| `ModelFin_107A` | Four-state electrochemical benchmark | Best current full-cycle soft-label calibration benchmark for `cs_a/cs_c/phie/phis_c`. |
| `EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only` | Four-state evaluation output | Use corrected metrics / paired true-pred NPZ fields. |
| `ModelFin_110_stageB` | SOH aging-head benchmark | Best current cycle-level SOH model. |
| `EvalFin_110_stageB_aging` | SOH evaluation output | Main SOH metrics should use `capacity_by_split_complete_only`. |
| `EvalFin_110_joint_StageB_SOH_107A_states_fix2` | Current five-target hybrid scorecard | Uses 107A paired state arrays + Stage B complete-only SOH. |
| `Data/assb_capacity_soh_targets/capacity_soh_targets.csv` | Experimental SOH labels | Generated from `ZHB_ASSB_NCM811.xlsx → step → 放电容量(Ah)`. |

## Five-target hybrid scorecard

| Variable | Source | MAE | RMSE | NMAE | NRMSE | R² | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| cs_a | state_prediction_npz_internal_true_pred | 0.0234896 | 0.0333314 | 0.99% | 1.41% | 0.995257 | 0.997626 |
| cs_c | state_prediction_npz_internal_true_pred | 0.392294 | 0.54137 | 1.63% | 2.25% | 0.989205 | 0.996226 |
| phie | state_prediction_npz_internal_true_pred | 0.00617205 | 0.00724753 | 1.60% | 1.88% | 0.998558 | 0.999441 |
| phis_c | state_prediction_npz_internal_true_pred | 0.00945861 | 0.0111716 | 0.96% | 1.13% | 0.998466 | 0.999667 |
| SOH | StageB_complete_only | 0.0030745 | 0.00395658 | 1.15% | 1.47% | 0.996579 | 0.998303 |


The SOH result uses complete cycles only (`n=517`). Cycle 522 is treated as incomplete/partial and should be reported separately, not used as the primary complete-cycle SOH metric.

## SOH benchmark details

`ModelFin_110_stageB` complete-cycle result:

```text
SOH_MAE   = 0.003074499
SOH_RMSE  = 0.003956582
SOH_R2    = 0.996579444
SOH_corr  = 0.998303304
SOH_NMAE  = 1.1457%
SOH_NRMSE = 1.4744%
SOH_obs_min  = 0.731645570
SOH_pred_min = 0.734077863
```

## Stage summary

### D1

ASSB priors, I(t) training adaptation, v3 soft labels, and early physics-only training were established. `ModelFin_52` failed against v3 soft labels, motivating a return to cycle5-only closure.

### D2

`ModelFin_101` became the cycle5_v4 benchmark. Key successful structures:

- I(t)-cbar hard baseline.
- Asymmetric radial deviations.
- Current-aware potential baseline.

### D3

The workflow moved to continuous cycle5-522 soft labels. `ModelFin_102` showed useful potential trends but failed positive-electrode concentration/state (`cs_c/theta_c`). This motivated cbar mass closure diagnostics and a six-variable evaluation requirement.

### D4

The v2 mass-closed candidate, radial ablation, positive-electrode radial shrinkage, linear-cycle common-mode gauge, and anode correction produced `ModelFin_107A`, the current best full-cycle soft-label four-state calibration benchmark.

### D5

D5 explored SOH/aging:

- `ModelFin_108` failed as an independent capacity/SOH curve head: it did not embed electrochemical aging mechanisms.
- `ModelFin_109` failed because the aging head was weakly coupled and largely stayed near initialization.
- `ModelFin_110_stageB` succeeded as a standalone cycle-level SOH aging head.
- `ModelFin_110` Stage C injection was usable but weaker than Stage B.
- The corrected joint evaluator (`fix2`) proved the current best five-target result is the hybrid benchmark: `107A states + 110_stageB SOH`.

## Important engineering rules

1. Do not assume remote GitHub equals the local project state. The local project has many manual overlays and replacements.
2. Deliver complete overwrite files by default. Do not create overlay files or rely on `*_base.py` unless explicitly requested.
3. The user manually backs up old files outside the project. Do not force in-project backup wrappers.
4. Keep `DATA_LOSS=False` for the original soft-label data loss unless explicitly instructed otherwise.
5. Always distinguish:
   - soft-label electrochemical-state evaluation;
   - experimental capacity/SOH evaluation;
   - experimental voltage fitting.
6. For sampled concentration arrays, use paired `*_true` / `*_pred` fields from the evaluation NPZ. Do not flatten-truncate sampled arrays against full-length references.
7. Main SOH metrics should use complete cycles only; cycle 522 is incomplete and should be listed separately.

## Reproduction commands

### Stage A: verify 107A core

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\compare_assb_107A_core_integrity.py `
  --model_dir ".\ModelFin_107A" `
  --solution_npz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  --output_dir ".\EvalFin_110_stageA_core_integrity" `
  --device cuda
```

### Stage B: train / evaluate SOH aging head

```powershell
.\scripts\run_ModelFin110_stageB.ps1 `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -Device "cuda"

Get-Content ".\EvalFin_110_stageB_aging\metrics_capacity_by_split.json"
```

### Joint five-target hybrid evaluation

```powershell
.\scripts\run_assb_joint_stageB_soh_state_eval.ps1 `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -StageBEvalDir ".\EvalFin_110_stageB_aging" `
  -ReferenceNpz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  -StateEvalDir ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only" `
  -OutputDir ".\EvalFin_110_joint_StageB_SOH_107A_states_fix2"
```

## D6 roadmap

D6 should create a single model that preserves `ModelFin_107A` four-state quality while embedding the `ModelFin_110_stageB` aging/SOH mechanism. The minimum acceptable D6 target is:

```text
One model directory + one evaluation interface + five outputs:
cs_a, cs_c, phie, phis_c, SOH
```

Recommended D6 route:

1. Freeze 107A core first.
2. Attach or inject Stage B aging mechanism in a way that does not alter 107A four-state outputs.
3. Validate five-state output with the fix2 alignment/provenance logic.
4. Only then consider partial unfreezing or deeper coupling through `LAM_c(k)`, `theta_window_c(k)`, `R_ohm(k)`, and later `LLI(k)`, `Ds_c(k)`, `i0_c(k)`.

---

Generated from ASSB-D5 project recap, 2026-05-15.
