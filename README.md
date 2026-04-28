# PINN-for-ASSB-V1

Physics-informed neural-network surrogate workflow for an **NMC811 || Li-In/In all-solid-state battery (ASSB)** using an adapted effective single-particle model (**effective SPM**).

This repository is an active adaptation of **NREL/PINNSTRIPES** for the QJW-2 ASSB workflow. It is not yet a finalized or fully validated predictive model. The current priority is to close the soft-label generation, PINN training, and evaluation loop before using the model for formal parameter inference.

---

## Current project status

The project currently contains three connected workflows:

1. **ASSB effective SPM prior**
   - Positive electrode: NMC811 representative spherical particle.
   - Negative electrode: Li-In/In foil represented as an equivalent pseudo-particle / effective diffusion length.
   - Electrolyte concentration and electrolyte potential variables are retained from the original SPM notation but reinterpreted as effective quantities of the solid-state ionic conduction network.

2. **Soft-label generation**
   - `integration_spm/spm_int_assb_cycle.py` generates ASSB soft labels from the effective SPM.
   - The v3 generator supports both single-cycle generation and merged continuous `cycle >= 5` generation.
   - Soft labels are treated as **model-generated training targets**, not as experimentally measured internal states.

3. **PINN training and evaluation**
   - `main.py` trains the PyTorch/CUDA PINN model.
   - `evaluate_assb_pinn_vs_softlabels.py` compares trained PINN outputs against soft labels.
   - The current physics-only `ModelFin_52` evaluation against `cycles5plus_v3` is not acceptable yet; this is a known debugging target.

### Validation status

| Item | Status |
|---|---|
| ASSB prior parameters | Implemented as first-version effective SPM prior |
| Cycle-5 v3 soft-label voltage fit | Good first workflow benchmark: MAE about 0.0456 V, RMSE about 0.0745 V, correlation about 0.972 |
| Continuous `cycle >= 5` v3 soft labels | Generated, but harder for the current PINN training loop |
| `ModelFin_52` vs `cycles5plus_v3` | Failed current benchmark: `phis_c` MAE about 0.3359 V and correlation near 0 |
| Recommended next step | Debug evaluation + cycle5-only closure before formal data-loss fine-tuning |

---

## Important modeling assumptions

This repository uses an **effective SPM** rather than a full P2D ASSB model.

The intended physical interpretation is:

- `cs_c(r,t)` is the NMC811 positive-electrode solid-phase lithium concentration.
- `cs_a(r,t)` is the Li-In/In negative-side effective state variable over an equivalent diffusion length.
- `ce` is retained as an effective mobile lithium-ion concentration scale in the solid-state ionic network.
- `phie` is retained as an effective solid-state ionic network potential.
- The current profile `I(t)` drives both charge and discharge through a unified sign convention.
- Surface fluxes are closed using current, particle/effective radius, active material volume fraction, and total electrode volume:

```text
J_a(t) = -I(t) * R_a / (3 * eps_a * F * V_a)
J_c(t) =  I(t) * R_c / (3 * eps_c * F * V_c)
```

The terminal voltage closure includes:

```text
positive OCP - negative OCP
+ positive/negative reaction overpotentials
+ effective lumped ohmic term R_ohm_eff
+ v3 empirical voltage-alignment offset
```

The current v3-aligned training parameters include:

```text
theta_c_bottom = 0.834
theta_c_top    = 0.432
R_ohm_eff      = 105.0 ohm
voltage_alignment_offset = -0.11588681607942332 V
csanmax        = 6.0  # Li-In/In effective scaling value, not a strict material constant
```

---

## Relationship to PINNSTRIPES

This project is based on the PINNSTRIPES workflow for physics-informed neural-network surrogates of battery models. The upstream framework uses interior physics losses, boundary losses, optional data losses, and optional regularization losses. For the SPM, particle-surface flux boundary residuals are especially important because the solid concentration dynamics are driven by the surface flux condition.

In this ASSB adaptation, the main changes are:

- use of ASSB-specific effective SPM parameters;
- use of NMC811 and Li-In/In OCP priors;
- time-dependent experimental current `I(t)` instead of a simple fixed discharge current;
- bidirectional charge/discharge concentration rescaling;
- v3 soft-label generation and evaluation utilities;
- PyTorch/CUDA training entry point.

---

## Main workflow files

Some upstream, historical, IDE, and experimental files may still be present in the repository during the active debugging stage. The current ASSB workflow should be understood through the files below.

| Path | Role |
|---|---|
| `main.py` | Current PyTorch/CUDA training entry point |
| `util/spm_assb_train_discharge.py` | ASSB parameter entry point; reads soft-label or experimental current profile |
| `util/thermo_assb.py` | OCP, Butler-Volmer, exchange current, voltage-alignment, and effective SPM parameters |
| `util/_losses.py` | Physics loss definitions, including time-dependent current flux closure |
| `util/_rescale.py` | Neural-network output rescaling; adapted for charge/discharge bidirectional behavior |
| `integration_spm/spm_int_assb_cycle.py` | v3 ASSB soft-label generator; supports single-cycle and merged-cycle generation |
| `evaluate_assb_pinn_vs_softlabels.py` | Evaluation script comparing PINN outputs with `.npz` soft labels |
| `input_assb_cycles5plus_pretrain` | Current physics-only pretraining input for merged `cycle >= 5` soft-label workflow |

---

## External data layout

Large/generated data files are intentionally kept outside the Git repository during the current debugging stage.

Recommended local paths on the current workstation:

```text
Project root:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

Experimental record CSV:
C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv

OCP prior directory:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs

Merged cycle>=5 v3 soft labels:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycles5plus_v3

Recommended cycle5-only debug soft labels:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_v3
```

At minimum, a soft-label directory should contain files such as:

```text
solution.npz
data_phie.npz
data_phis_c.npz
data_cs_a.npz
data_cs_c.npz
soft_label_summary.json
```

The `.npz` soft labels are generated model targets. They are not ground-truth internal-state measurements.

---

## Environment

The current training entry point requires a CUDA-enabled PyTorch runtime. `main.py` checks `torch.cuda.is_available()` and exits if CUDA is unavailable.

The existing `requirements.txt` may still reflect upstream or legacy dependencies and should not be treated as a complete current ASSB environment specification. For now, use the local CUDA PyTorch environment that has been used in development, for example:

```powershell
D:\Anaconda\envs\torchgpu\python.exe --version
D:\Anaconda\envs\torchgpu\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Common Python packages used by the current workflow include:

```text
numpy
pandas
matplotlib
torch
```

Additional dependencies may be required by upstream PINNSTRIPES utilities or older scripts.

---

## Generate v3 soft labels

### Single cycle 5 debug target

Use this first when debugging the training/evaluation loop:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe integration_spm\spm_int_assb_cycle.py `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --cycle 5 `
  --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_v3" `
  --n_r 64
```

### Merged continuous cycle >= 5 target

Use this only after the cycle5-only loop is confirmed:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe integration_spm\spm_int_assb_cycle.py `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --merge_cycles `
  --cycle_from 5 `
  --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycles5plus_v3" `
  --n_r 64
```

After generation, check `soft_label_summary.json` and confirm that the voltage alignment terms match the intended training configuration.

---

## Train the PINN

Set the external data paths before training:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycles5plus_v3"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"

D:\Anaconda\envs\torchgpu\python.exe main.py -i input_assb_cycles5plus_pretrain
```

For debugging, prefer a cycle5-only input file and set:

```powershell
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_v3"
```

Recommended training order:

```text
1. cycle5-only physics-only training
2. cycle5-only evaluation
3. increase boundary sampling/weight if surface-flux residual remains dominant
4. add L-BFGS only after ADAM/SGD has reached a stable plateau
5. only then test merged cycle>=5
6. only after physics/evaluation consistency is confirmed, introduce small-weight data loss
```

---

## Evaluate a trained model

Example for the current known failed benchmark:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe evaluate_assb_pinn_vs_softlabels.py `
  --model_dir ModelFin_52 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycles5plus_v3" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --output_dir EvalFin_52_vs_softlabels
```

A successful first debug target should not merely reduce voltage MAE; it should also recover the time trend. For cycle5-only debugging, a practical first acceptance target is:

```text
phis_c correlation > 0.8
phis_c MAE < 0.08 V
theta_c correlation > 0.5
no obvious constant-output solution for theta_a/theta_c
```

If `phis_c` correlation is close to zero or negative, first check model loading, input normalization, output variable index mapping, and `theta = cs / csmax` conversion before changing physical parameters.

---

## Known current issue

`ModelFin_52` does not yet reproduce the `cycles5plus_v3` soft-label time series. The current result is interpreted as a failed training/evaluation closure, not as a validated model result.

Most likely causes to check first:

1. evaluation script mismatch;
2. checkpoint not loaded as expected;
3. time or radius rescaling mismatch;
4. output index mismatch between model and evaluator;
5. `theta_a/theta_c` conversion mismatch;
6. continuous `cycle >= 5` task being too hard for the current network/collocation setup;
7. insufficient learning of particle-surface flux boundary residuals.

Do not use `ModelFin_52` for physical conclusions until the above checks pass.

---

## Debugging roadmap

### P0 — close the simplest loop

Use `cycle5_v3` only. Confirm that the generator, training input, trained checkpoint, and evaluator all use the same:

```text
I(t)
OCP directory
R_ohm_eff
voltage_alignment_offset
theta_c_bottom / theta_c_top
csanmax / cscmax
rescale_T / rescale_R
output variable ordering
```

### P1 — strengthen physics training

If the evaluator is correct but the model remains poor:

- increase surface boundary collocation points or boundary weights;
- check whether predictions are constant, reversed, or only offset-biased;
- run L-BFGS only after the ADAM/SGD loss plateaus;
- consider a larger network for merged multi-cycle time series.

### P2 — introduce data loss carefully

Data loss should be introduced only after the physics-only/evaluation loop is verified. The goal is to refine a physically consistent model, not to hide a mismatch between the soft-label generator and the PINN physics loss.

---

## Notes on repository cleanliness

During the current debugging stage, some historical, upstream, IDE, or experimental files may remain in the repository. This README identifies the current ASSB mainline workflow and does not require immediate removal of those files.

Before a formal public release, recommended cleanup tasks include:

```text
- add or update .gitignore for generated .npz labels, ModelFin_*, LogFin_*, EvalFin_*, and IDE files;
- replace the legacy requirements.txt with a PyTorch/CUDA-oriented environment file;
- add a small reproducible example or smoke-test dataset;
- move project notes into docs/;
- keep raw experimental data and generated soft labels outside normal Git history unless intentionally using Git LFS.
```

---

## Acknowledgements

This project adapts ideas and code structure from NREL/PINNSTRIPES:

```text
Hassanaly et al., PINN surrogate of Li-ion battery models for parameter inference,
Part I: Implementation and multi-fidelity hierarchies for the single-particle model.
Journal of Energy Storage 98 (2024) 113103.

Hassanaly et al., PINN surrogate of Li-ion battery models for parameter inference,
Part II: Regularization and application of the pseudo-2D model.
Journal of Energy Storage 98 (2024) 113104.
```

The present repository is an ASSB-specific adaptation for NMC811 || Li-In/In cells and should be cited or described separately from the upstream PINNSTRIPES project.
