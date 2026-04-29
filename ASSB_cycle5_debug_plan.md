# ASSB cycle5-only physics-only debug experiment plan

## Goal

Use `Data\assb_soft_labels_cycle5_v3` as the only soft-label source and keep data loss disabled. The goal is not to claim final model performance. The goal is to answer one debugging question:

> Can the current PINN training/evaluation workflow learn and reproduce the cycle5 v3 soft-label trajectory under physics-only training?

If cycle5 still fails, the root cause is likely model loading, summary/rescale mismatch, output mapping, current sign, or boundary-flux training failure. If cycle5 passes, then the previous cycles5plus failure is likely caused by continuous multi-cycle difficulty.

## Files in this package

| File | Put in project root? | Role |
|---|---:|---|
| `assb_cycle5_softlabel_sanity.py` | Yes | Checks `Data\assb_soft_labels_cycle5_v3` before training. |
| `evaluate_assb_cycle5_pinn_vs_softlabels_debug.py` | Yes | Enhanced evaluator with checkpoint, summary/rescale, output-index and first-batch debug. |
| `input_assb_cycle5_debug_smoke` | Yes | 200-epoch smoke training, ID=60. |
| `input_assb_cycle5_debug_sgd` | Yes | Main baseline cycle5 physics-only training, ID=61. |
| `input_assb_cycle5_debug_boundary` | Yes | Continue from ModelFin_61 with stronger r=R boundary weighting, ID=62. |
| `input_assb_cycle5_debug_lbfgs` | Yes | Optional L-BFGS refinement from ModelFin_62, ID=63. |
| `run_assb_cycle5_debug.ps1` | Yes | PowerShell runner for one stage at a time. |

## Important environment variables

```powershell
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\Data\assb_soft_labels_cycle5_v3"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
```

The training input files do not encode the soft-label path. The path is controlled by `ASSB_SOFT_LABEL_DIR`.

## Step 0: copy files

Copy all files into:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

## Step 1: soft-label sanity check

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe assb_cycle5_softlabel_sanity.py `
  --soft_label_dir Data\assb_soft_labels_cycle5_v3 `
  --output_dir Eval_cycle5_softlabel_sanity
```

Expected output files:

```text
Eval_cycle5_softlabel_sanity\softlabel_sanity_report.txt
Eval_cycle5_softlabel_sanity\softlabel_sanity_report.json
Eval_cycle5_softlabel_sanity\softlabel_phis_c_vs_experiment.png
Eval_cycle5_softlabel_sanity\solution_current_profile.png
```

Pass condition:

```text
phis_c label vs experiment corr > 0.95
phis_c label vs experiment MAE approximately 0.05 V or lower
all required data_*.npz files exist
x_train and y_train row counts match
```

If this fails, do not train. The soft-label folder is not the expected v3 cycle5 folder.

## Step 2: smoke training, ID=60

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage smoke
```

This creates:

```text
ModelFin_60
LogFin_60
EvalFin_60_cycle5_smoke
```

Pass condition: script runs without path, CUDA, or loader errors. Do not judge final accuracy from this stage.

## Step 3: baseline cycle5 training, ID=61

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage baseline
```

This creates:

```text
ModelFin_61
LogFin_61
EvalFin_61_cycle5_baseline
```

Preliminary success target:

```text
phis_c corr >= 0.7
phis_c MAE <= 0.12 V
pred_std / label_std should not be near 0
```

Good cycle5 closure target:

```text
phis_c corr >= 0.85
phis_c MAE <= 0.08 V
theta_c corr >= 0.5
theta_a and theta_c are not nearly constant
```

## Step 4: boundary-focused continuation, ID=62

Run only if ID=61 still looks underfit or nearly constant:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage boundary
```

This continues from `ModelFin_61\best.pt`, increases the surface boundary weights from 250 to 500, increases boundary batch size, and evaluates `ModelFin_62`.

Interpretation:

- If `phis_c` and surface `cs_a/cs_c` improve, the failure was mostly a boundary-flux/trivial-solution problem.
- If correlation stays near 0, inspect `debug_model_and_data.json` before continuing training.

## Step 5: optional L-BFGS refinement, ID=63

Run only if ID=62 already has a reasonable trend:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage lbfgs
```

Do not run L-BFGS first. It is a refinement stage, not a substitute for fixing wrong loading/rescale/output mapping.

## How to read the enhanced evaluator output

Open:

```text
EvalFin_xx_cycle5_*\metrics_summary.txt
EvalFin_xx_cycle5_*\debug_model_and_data.json
EvalFin_xx_cycle5_*\timeseries_phis_c.png
EvalFin_xx_cycle5_*\surface_timeseries_cs_a.png
EvalFin_xx_cycle5_*\surface_timeseries_cs_c.png
EvalFin_xx_cycle5_*\correlation_phis_c.png
```

Key failure signatures:

| Signature | Likely cause |
|---|---|
| `checkpoint` is not the expected file | Wrong model folder or missing `best.pt`. |
| `chosen_train_summary_json` is not cycle5 summary | Summary/rescale mismatch. |
| `pred_std / label_std < 0.2` | Constant/trivial solution or severe rescale issue. |
| `phis_c corr ≈ 0` | Training did not learn time trajectory or evaluation mapping is wrong. |
| `theta_c corr < 0` | Current sign, cathode theta/cs mapping, or bidirectional rescale issue. |
| `phis_c MAE` high but corr high | Mainly offset/R_ohm/alignment problem. |

## When to stop

Stop and inspect scripts/data if any of these happen:

```text
1. soft-label sanity check fails;
2. chosen_train_summary_json is not Data\assb_soft_labels_cycle5_v3\soft_label_summary.json;
3. first batch predictions are NaN/Inf;
4. ModelFin_61 and ModelFin_62 both have phis_c corr < 0.2;
5. theta_c correlation is negative after boundary continuation.
```

Do not open data loss until cycle5 physics-only loading/evaluation is understood.
