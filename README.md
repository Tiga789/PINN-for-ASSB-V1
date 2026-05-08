# PINN-for-ASSB-V1

Physics-informed neural-network surrogate workflow for an **NMC811 || Li-In/In all-solid-state battery (ASSB)** using an adapted **effective single-particle model (effective SPM)**.

This repository is an active QJW-2 research workflow adapted from the PINNSTRIPES idea. It is **not yet a finalized aging-prediction model**. The best single-cycle closed-loop baseline is still **ModelFin_101 on cycle5_v4**, while the current D3 work has moved into **continuous cycle5-522 soft labels** and **ModelFin_102 / ModelFin_103 long-sequence evaluation**.

---

## 1. Battery system

Target cell: ZHB all-solid-state NMC811 / Li-In cell.

**Positive electrode**

- 5 mg composite positive electrode.
- Composition: **30% LAOC + 70% single-crystal NMC811 + 1% VGCF**.
- In the effective SPM, the positive electrode is represented by an **NMC811 representative spherical active particle**.

**Negative electrode**

- **10 mm diameter, 100 μm In foil + 8 mm diameter, 50 μm Li foil**, forming a **Li-In alloy / In negative side**, plus stainless-steel sheet.
- In the effective SPM, the negative side is represented as a **Li-In/In effective pseudo-particle with an equivalent diffusion length**, not as a real porous particle electrode.

**Solid electrolyte stack**

- 60 mg LPSC layer.
- 60 mg LAOC layer.
- The SPM variables `ce` and `phie` are retained, but reinterpreted as effective variables of the solid-state ionic conduction network.

**Voltage and temperature**

- Full-cell charge cutoff voltage: **3.68 V**.
- This corresponds approximately to **4.3 V vs. Li/Li+** because the Li-In negative electrode contributes an approximately **0.62 V** reference offset.
- Temperature: **303.15 K**.

---

## 2. Current project status

### D2 closed-loop baseline

The D2-stage workflow returned to **cycle5-only** debugging and produced a strong baseline:

```text
Soft labels:  Data/assb_soft_labels_cycle5_v4
Model:        ModelFin_101
Evaluation:   EvalFin_101_cycle5_v4_cbarAC_potentialBaseline
Input:        input_assb_cycle5_v4_cbarAC_potentialBaseline_ID101
```

ModelFin_101 metrics against cycle5_v4 soft labels:

```text
phis_c   MAE ≈ 0.00405 V, RMSE ≈ 0.00568 V, corr ≈ 0.999956
phie     MAE ≈ 0.00633 V, RMSE ≈ 0.00840 V, corr ≈ 0.999114
theta_a  MAE ≈ 0.01706,   corr ≈ 0.9834
theta_c  MAE ≈ 0.00369,   corr ≈ 0.9996
```

### D3 continuous-cycle expansion

D3 generated a continuous all-cycle soft-label dataset from cycle 5 to 522:

```text
Soft-label directory:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1

Main file:
solution.npz
```

Integrity check result:

```text
Status: PASS
N_time_points: 373235
cycle_min / cycle_max / cycle_count: 5 / 522 / 518
t_global_s range: 0.0 to 3727659.0 s
dt min / median / max: 1.0 / 10.0 / 10.0 s
I_profile: charge positive, discharge negative, rest zero
flux signs: j_a opposite to I, j_c same as I
fixed-B voltage global: MAE ≈ 0.16384 V, RMSE ≈ 0.21083 V, corr ≈ 0.93561
```

The fixed-B voltage error is expected to be larger than cycle5_v4 because no explicit SOH / aging mechanism is currently included.

---

## 3. Core physical conventions

This repository uses fixed **positive / negative electrode identity** rather than switching material identity with charge/discharge role.

```text
a = negative electrode = Li-In/In effective pseudo-particle
c = positive electrode = NMC811 representative particle
```

During discharge, the positive electrode has the cathode reaction role and the negative electrode has the anode reaction role. During charge, the reaction roles switch. However, the following properties **do not switch**:

```text
OCP tables
geometry
solid diffusivity
exchange-current functions
maximum concentration scales
```

The current sign controls the flux direction:

```text
+I = charge
-I = discharge
 I = 0 = rest
```

Effective SPM surface-flux closure:

```text
J_a(t) = -I(t) * R_a / (3 * eps_a * F * V_a)
J_c(t) =  I(t) * R_c / (3 * eps_c * F * V_c)
```

---

## 4. Soft-label datasets

### cycle5_v4

Used for ModelFin_101 single-cycle baseline.

```text
Data/assb_soft_labels_cycle5_v4
```

Known cycle5_v4 values:

```text
n_t = 925
n_r = 64
tmax_s = 9232.0
I_min_A = -0.00033
I_max_A =  0.00033
Rs_a = 50 μm
Rs_c = 1.8 μm
eps_s_a = 0.95
eps_s_c = 0.55
csanmax = 6.0
cscamax = 51.8
T = 303.15 K
```

### cycle5-522 continuous v1

Used for ModelFin_102 / ModelFin_103 long-sequence expansion.

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1
```

Important fields in `solution.npz`:

```text
t_global_s
cycle_id
step_id / step_type
I_profile
voltage_exp
r_a / r_c
cs_a / cs_c
theta_a / theta_c
phie / phis_c
j_a / j_c
eta_a / eta_c
Uocp_a / Uocp_c
```

This is a single continuous trajectory. Do **not** reset time or concentration at cycle boundaries unless intentionally running a separate ablation.

---

## 5. Model status

### ModelFin_101

Current best single-cycle benchmark.

Key design:

- I(t)-cbar hard baseline.
- Negative Li-In/In: small zero-mean radial deviation.
- Positive NMC811: stronger zero-mean radial deviation.
- Current-aware potential baseline for `phie` and `phis_c`.
- Physics-only cycle5_v4 training.

### ModelFin_102

First full continuous cycle5-522 trial.

Soft-label-only global metrics:

```text
phis_c   MAE ≈ 0.07836 V, corr ≈ 0.9720, R2 ≈ 0.8772
phie     MAE ≈ 0.01238 V, corr ≈ 0.9977, R2 ≈ 0.9949
theta_a  MAE ≈ 0.02159,   corr ≈ 0.9410, R2 ≈ 0.8568
theta_c  MAE ≈ 0.21384,   corr ≈ 0.4715, R2 ≈ -3.497
cs_c     MAE ≈ 11.08,     corr ≈ 0.4715, R2 ≈ -3.497
```

Interpretation: potential branches and negative concentration are usable, but the positive concentration/state branch is not closed. ModelFin_102 is therefore **not** the current full-state baseline.

### ModelFin_103

Current smoke / short-range long-sequence model.

Current use:

```text
Input: input_assb_cycles5to522_v4_continuous_ID103_smoke
Training slice: cycle 5-20
Evaluation directory: EvalFin_103_cycles5to20_smoke
```

Observed potential per-cycle behavior:

```text
phis_c cycle5  MAE ≈ 0.01478 V, corr ≈ 0.999892, R2 ≈ 0.99675
phis_c cycle20 MAE ≈ 0.03423 V, corr ≈ 0.999740, R2 ≈ 0.98570
phie worst early cycles: MAE ≈ 0.0316-0.0343 V, corr ≈ 0.99998-0.999999
```

The old evaluator did not output per-cycle `theta_a`, `theta_c`, `cs_a`, or `cs_c`. Use the new cycle-range evaluator below to obtain all six variables.

---

## 6. Current evaluation target

Do **not** create a new training ID only to evaluate cycle 5-100. Use ModelFin_103 directly:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_cycles5_100_softlabels.py `
  --model_dir ModelFin_103 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --cycle_from 5 `
  --cycle_to 100 `
  --output_dir EvalFin_103_cycles5_100_v1_softlabel_only `
  --debug_print_first_batch
```

Expected outputs:

```text
EvalFin_103_cycles5_100_v1_softlabel_only/metrics_global.json
EvalFin_103_cycles5_100_v1_softlabel_only/metrics_by_cycle.csv
EvalFin_103_cycles5_100_v1_softlabel_only/debug_model_and_data.json
EvalFin_103_cycles5_100_v1_softlabel_only/eval_sampled_arrays_cycles5_100_softlabel_only.npz
EvalFin_103_cycles5_100_v1_softlabel_only/plots_softlabel_only/*.png
```

`metrics_by_cycle.csv` should contain:

```text
phis_c
phie
theta_a
theta_c
cs_a
cs_c
```

---

## 7. Important diagnostics and known issues

### 7.1 Stale summary environment variable

Before running long-sequence training/evaluation, clear stale cycle5_v4 summary paths:

```powershell
Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
```

### 7.2 Windows path delimiter issue

Some input parsers split lines by `:`. Absolute paths such as `C:\...` can therefore break parsing. Prefer relative paths in input files, for example:

```text
ASSB_SOFT_LABEL_DIR : ..\assb_soft_lable_cycle5-522_v1
ASSB_OCP_DIR : ..\ocp_estimation_outputs
```

### 7.3 Positive concentration closure issue

Diagnostics showed the negative-electrode cbar closure is much more consistent than the positive-electrode cbar closure in the continuous soft-label file. If `theta_c` / `cs_c` remains bad in ModelFin_103 cycle5-100 evaluation, check the positive-electrode soft-label mass closure before opening data loss.

Useful scripts:

```text
diagnose_cbar_mass_weights.py
diagnose_cbar_mass_weights_v2.py
repair_assb_solution_mass_closure.py
```

The repair script exists, but the repaired solution should not be promoted to the main training dataset until its OCP, voltage, theta, and mass-closure consistency are verified.

---

## 8. Main workflow files

```text
main.py
util/spm_assb_train_discharge.py
util/thermo_assb.py
util/_losses.py
util/_rescale.py
util/init_pinn.py
util/myNN.py
integration_spm/spm_int_assb_cycle.py
integration_spm/generate_assb_soft_labels_cycle5_522_v1.py
integration_spm/generate_assb_softlabel_allcycle.py
evaluate_assb_pinn_vs_softlabels.py
evaluate_assb_pinn_cycles5_100_softlabels.py
plot_cs_surface_cycle5.py
plot_cs_surface_cycle5_plotly.py
inspect_assb_softlabel_solution.py
diagnose_cbar_mass_weights_v2.py
repair_assb_solution_mass_closure.py
```

---

## 9. Recommended local paths

```text
Project root:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

Experimental record CSV:
C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv

OCP prior directory:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs

cycle5_v4 soft labels:
Data\assb_soft_labels_cycle5_v4

continuous cycle5-522 soft labels:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1

Best cycle5 model:
ModelFin_101

Current long-sequence models:
ModelFin_102 / ModelFin_103
```

---

## 10. Next-step policy

1. Keep ModelFin_101 as the frozen cycle5_v4 baseline.
2. Evaluate ModelFin_103 on cycle 5-100 before creating a new training ID.
3. Check all six variables in `metrics_by_cycle.csv`.
4. If potential is good but `theta_c` / `cs_c` is bad, debug soft-label mass closure and positive-electrode cbar baseline first.
5. Only after cycle5-100 is understood, expand to cycle5-200 and then cycle5-522.
6. Add SOH / aging parameters only if per-cycle residuals show systematic drift with cycle number.
7. Do not open formal data loss until physics/output mapping and soft-label consistency are confirmed.

---

## 11. Repository cleanup notes

Before public GitHub release:

```text
- Keep ModelFin_*, LogFin_*, EvalFin_* directories out of normal Git history unless intentionally archived.
- Keep large .npz soft-label files outside Git or use Git LFS.
- Store project progress summaries under docs/.
- Keep generated patch zips outside source history.
- Add a small reproducible smoke-test example.
- Update requirements/environment files for the CUDA PyTorch workflow.
```

---

## 12. Acknowledgements

This work adapts the PINN surrogate concept and workflow style from NREL/PINNSTRIPES and the related PINN surrogate papers for Li-ion battery models. The present repository is an ASSB-specific adaptation for an NMC811 || Li-In/In all-solid-state cell and should be described separately from the upstream project.
