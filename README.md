# PINN-for-ASSB-V1

Physics-informed neural-network surrogate workflow for an **NMC811 || Li-In/In all-solid-state battery (ASSB)** using an adapted **effective single-particle model (effective SPM)**.

This repository is an active QJW-2 research workflow adapted from the PINNSTRIPES idea. It is **not yet a finalized aging-prediction model**. The best single-cycle closed-loop baseline is still **ModelFin_101 on cycle5_v4**, while the current D4 continuous-cycle benchmark is **ModelFin_107A on cycle5-522 v2 mass-closed candidate soft labels**.

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

## 2. Core physical conventions

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

## 3. Current project status

### D2 single-cycle baseline

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
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1
```

The v1 continuous dataset passed basic integrity checks, but ModelFin_102 / ModelFin_103 exposed a major positive-electrode state issue. ModelFin_102 full-cycle evaluation had usable potentials and negative state, but failed for positive concentration/state:

```text
theta_c MAE ≈ 0.21384, corr ≈ 0.4715, R2 ≈ -3.497
cs_c    MAE ≈ 11.08,   corr ≈ 0.4715, R2 ≈ -3.497
```

### D4 current continuous-cycle benchmark

D4 replaced the v1 continuous target with a **v2 mass-closed candidate** and built calibrated wrappers:

```text
Soft-label candidate:
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate

Current best full-cycle benchmark:
ModelFin_107A

Current best evaluation directory:
EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only
```

ModelFin_107A full-cycle corrected metrics against v2 mass-closed candidate soft labels:

```text
phis_c   MAE=0.0094586 V, RMSE=0.0111716 V, R2=0.998466, corr=0.999667
phie     MAE=0.0061720 V, RMSE=0.0072475 V, R2=0.998558, corr=0.999441
theta_a  MAE=0.0039149,   RMSE=0.0055552,   R2=0.995257, corr=0.997626
theta_c  MAE=0.0075732,   RMSE=0.0104512,   R2=0.989205, corr=0.996226
cs_a     MAE=0.0234896,   RMSE=0.0333314,   R2=0.995257, corr=0.997626
cs_c     MAE=0.392294,    RMSE=0.541370,    R2=0.989205, corr=0.996226
```

Important: **ModelFin_107A is a full-cycle calibration benchmark, not a held-out validation result.** Its anode correction was calibrated over cycle 5-522. A stricter check using cycle 5-100 calibration and cycle 5-522 evaluation failed for `cs_a/theta_a`:

```text
cs_a MAE after ≈ 0.3613, R2 ≈ -0.3723
theta_a MAE after ≈ 0.0602, R2 ≈ -0.3723
```

---

## 4. Soft-label datasets

### `Data/assb_soft_labels_cycle5_v4`

Used for the ModelFin_101 single-cycle baseline.

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

### `assb_soft_lable_cycle5-522_v1`

D3 continuous all-cycle v1 soft labels.

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1
```

Status:

```text
Integrity: PASS
cycle range: 5-522
N_time_points: 373235
t_global_s is not reset at cycle boundaries
I_profile contains charge / discharge / rest
j_a has opposite sign to I; j_c has same sign as I
```

Limitation discovered in D4:

```text
positive cbar and radial state were not consistent enough with the hard I(t)-cbar output map.
```

### `assb_soft_labels_cycle5_522_v2_massclosed_candidate`

D4 main continuous soft-label candidate.

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate
```

The D4 v2 generator repairs the positive-electrode mean concentration by enforcing:

```text
d<c_c>/dt = -I(t) / (eps_s_c * F * V_c)
```

It preserves the positive radial shape while shifting the radial profile to match the target spherical average, then recomputes:

```text
theta_c
Uocp_c
eta_c
phie
phis_c
```

Expected solution fields include:

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
cbar_c_before_repair
cbar_c_from_I
cbar_c_after_repair
cbar_c_shift_applied
```

---

## 5. Model evolution

| Model | Main purpose | Status |
|---|---|---|
| ModelFin_101 | cycle5_v4 single-cycle baseline | Best single-cycle baseline. Do not overwrite. |
| ModelFin_102 | first full cycle5-522 attempt on v1 | Positive concentration/state failed. Not a benchmark. |
| ModelFin_103 | cycle5-20 smoke / cycle5-100 evaluation on v1/v2 | Useful diagnostic model; v2 improved but radial issue remained. |
| ModelFin_104 | reduce positive radial freedom (`CBAR_BASELINE_DEVIATION_FRACTION_C=0.10`) | Positive state improved; potential common-mode bias became large. |
| ModelFin_105 | potential-only soft-label anchor from ModelFin_104 | Concentrations stayed good; potential common-mode bias remained. |
| ModelFin_106 | ModelFin_105 + linear-cycle common-mode gauge wrapper | Current strong potential + positive-state wrapper. |
| ModelFin_107A | ModelFin_106 + anode cs_a/theta_a residual correction | Current best full-cycle soft-label calibration benchmark. |

---

## 6. Key D4 results

### ModelFin_103 v2 massclosed cycle5-100

```text
phis_c   MAE ≈ 0.02280 V, R2 ≈ 0.99317
theta_c  MAE ≈ 0.04050,   R2 ≈ 0.79465
cs_c     MAE ≈ 2.098,     R2 ≈ 0.79465
```

### Positive radial ablation

Using:

```text
cs_c_pred_ablation = cbar_c_pred + scale * (cs_c_pred - cbar_c_pred)
```

Best scale:

```text
scale = 0.0
cs_c MAE ≈ 0.543
theta_c MAE ≈ 0.0105
```

Original scale:

```text
scale = 1.0
cs_c MAE ≈ 2.098
theta_c MAE ≈ 0.0405
```

Conclusion: the positive radial deviation freedom was too strong.

### ModelFin_104 cycle5-100

```text
phis_c   MAE ≈ 0.112 V
phie     MAE ≈ 0.113 V
theta_c  MAE ≈ 0.00556
cs_c     MAE ≈ 0.288
```

Conclusion: positive concentration fixed, but potential common-mode bias became large.

### ModelFin_105 raw cycle5-100

```text
phis_c   MAE ≈ 0.0821 V
phie     MAE ≈ 0.0842 V
theta_c  MAE ≈ 0.00566
cs_c     MAE ≈ 0.293
```

Common-mode diagnostic:

```text
common_mode_error MAE ≈ 0.0832 V
phis_c - phie differential MAE ≈ 0.00723 V
```

Conclusion: shape/differential potential was already good; absolute gauge was wrong.

### ModelFin_106 corrected cycle5-100

```text
phis_c   MAE ≈ 0.00725 V, R2 ≈ 0.999331
phie     MAE ≈ 0.00151 V, R2 ≈ 0.999913
theta_c  MAE ≈ 0.00566,   R2 ≈ 0.996052
cs_c     MAE ≈ 0.293,     R2 ≈ 0.996052
```

### ModelFin_106 corrected cycle5-522

```text
phis_c   MAE ≈ 0.00946 V, R2 ≈ 0.998466
phie     MAE ≈ 0.00617 V, R2 ≈ 0.998558
theta_a  MAE ≈ 0.02043,   R2 ≈ 0.907169
theta_c  MAE ≈ 0.00757,   R2 ≈ 0.989205
cs_a     MAE ≈ 0.12261,   R2 ≈ 0.907169
cs_c     MAE ≈ 0.39229,   R2 ≈ 0.989205
```

Conclusion: linear-cycle gauge extrapolated well to full cycle5-522, but negative state remained the bottleneck.

### ModelFin_107A corrected cycle5-522

```text
phis_c   MAE ≈ 0.00946 V, R2 ≈ 0.998466
phie     MAE ≈ 0.00617 V, R2 ≈ 0.998558
theta_a  MAE ≈ 0.00391,   R2 ≈ 0.995257
theta_c  MAE ≈ 0.00757,   R2 ≈ 0.989205
cs_a     MAE ≈ 0.02349,   R2 ≈ 0.995257
cs_c     MAE ≈ 0.39229,   R2 ≈ 0.989205
```

Conclusion: current best full-cycle soft-label calibration benchmark.

---

## 7. Important caveats

1. **ModelFin_107A is not held-out validation.** It uses full-cycle calibration for the anode correction.
2. **Do not use v1 continuous soft labels as the main concentration target** unless intentionally reproducing D3 failure.
3. **Do not confuse corrected metrics with before metrics.** The 107A evaluation directory contains both:

```text
metrics_global_before_ModelFin106.json
metrics_global_corrected.json
```

4. **voltage_exp is ignored in the current PINN soft-label-only evaluation.** Experimental voltage can be used to judge fixed-B label quality, but should not be mixed into the current surrogate-label metric unless explicitly designing a new experiment.
5. **No explicit SOH/aging model is implemented yet.** Cycle-dependent gauge/correction is empirical at this stage.

---

## 8. Reproduction commands

### Evaluate ModelFin_106 full-cycle with linear-cycle gauge

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\check_ModelFin106_cycle5_522_package.ps1
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1
```

### Diagnose ModelFin_106 negative-state error

```powershell
.\scripts\run_diagnose_ModelFin106_csA_fullcycle.ps1
```

### Build and evaluate ModelFin_107A full-cycle calibration benchmark

```powershell
.\scripts\check_ModelFin107A_package.ps1
.\scripts\run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1
.\scripts\show_ModelFin107A_cycle5_522_worst_cycles.ps1
```

### Strict extrapolation check

```powershell
.\scripts\run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1
```

### Read the correct 107A metrics file

```powershell
Get-Content .\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only\metrics_global_corrected.json
```

---

## 9. Current key files and directories

```text
# Soft labels
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate

# Current wrappers
ModelFin_106
ModelFin_107A
ModelFin_107A_calib5_100

# Current evaluations
EvalFin_106_cycles5_522_v2_massclosed_candidate_linearCycleGauge_softlabel_only
EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only
EvalFin_107A_calib5_100_eval5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only

# Key D4 scripts
integration_spm/generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py
evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py
evaluate_assb_pinn_cycles5_522_v2_massclosed_softlabels.py
apply_ModelFin106_linear_cycle_gauge.py
apply_ModelFin106_linear_cycle_gauge_cycle5_522.py
build_ModelFin106_from_ModelFin105_linearCycleGauge.py
diagnose_ModelFin106_csA_cbar_radial_fullcycle.py
fit_apply_ModelFin107A_anode_state_correction.py

# Diagram
ASSB_PINN_architecture_v3_english_with_detailed_nn.svg
```

---

## 10. Next recommended work

### P0: freeze D4 benchmark

Back up:

```text
ModelFin_106
ModelFin_107A
EvalFin_106_*
EvalFin_107A_*
assb_soft_labels_cycle5_522_v2_massclosed_candidate
ASSB-D4.docx
README.md
D4 scripts/packages
```

### P0: design ModelFin_108

ModelFin_108 should convert the successful post-hoc mechanisms into a formal, testable model:

```text
1. output-map or wrapper-level common-mode gauge;
2. anode state residual correction;
3. train/validation cycle split;
4. held-out cycle metrics.
```

### P1: held-out validation

Use multiple splits:

```text
calib/train: cycle 5-100      eval: cycle 101-200 / 201-522
calib/train: cycle 5-200      eval: cycle 201-522
rolling or blocked calibration for aging-aware tests
```

### P1: audit v2 soft labels further

Check:

```text
OCP consistency
theta windows
voltage alignment
mass closure for both electrodes
per-cycle fixed-B residual trend
```

### P2: introduce aging/SOH if needed

If held-out residuals drift with cycle number, consider:

```text
R_ohm(k)
Q_loss(k)
theta window(k)
Ds_c(k)
i0(k)
```

---

## 11. New-window summary

The project currently has a strong D4 full-cycle soft-label calibration benchmark:

```text
ModelFin_107A = ModelFin_106 + anode cs_a/theta_a correction
ModelFin_106  = ModelFin_105 + linear-cycle common-mode gauge
ModelFin_105  = ModelFin_104 + potential-only data anchor
ModelFin_104  = ModelFin_103-style model with reduced positive radial freedom
```

Current best full-cycle soft-label metrics are excellent, but because 107A uses full-cycle calibration, the next scientific step is **not** to claim final predictive performance. The next step is to build **ModelFin_108** with embedded correction and held-out cycle validation.
