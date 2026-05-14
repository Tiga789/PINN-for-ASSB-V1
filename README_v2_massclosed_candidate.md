# ASSB cycle5-522 v2 mass-closed-candidate soft-label generator

This package contains the D4 candidate generator for:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate
```

## Files

```text
integration_spm/generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py
scripts/run_generate_v2_massclosed_candidate.ps1
README_v2_massclosed_candidate.md
```

## How to use

Extract this package into the project root:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

Then run:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\run_generate_v2_massclosed_candidate.ps1
```

Or run the Python file directly:

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\integration_spm\generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py `
  --source_solution_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate" `
  --cycle_from 5 `
  --cycle_to 522 `
  --overwrite
```

## What the generator does

1. Loads the existing v1 continuous soft label solution from:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1\solution.npz
```

2. Loads project parameters/OCP/fixed-B settings through the existing solver.

3. Repairs the positive-electrode `cs_c` mean concentration by forcing the spherical average to follow:

```text
d<c_c>/dt = - I(t) / (eps_s_c * F * V_c)
```

4. Applies a uniform shift over the positive radial grid at each time point, preserving radial shape while fixing `cbar_c`.

5. Recomputes `theta_c`, `Uocp_c`, `eta_c`, `phie`, and `phis_c` after the repair.

6. Writes a new candidate `solution.npz`, summary JSON, voltage metrics, and mass-closure audit files.

## Output files

```text
solution.npz
soft_label_summary.json
record_profile_summary.json
metrics_voltage_fixedB_by_cycle.csv
mass_closure_audit_global.json
mass_closure_audit_by_cycle.csv
mass_closure_audit_timeseries.csv
```

## Candidate warning

This is a candidate dataset. Do not promote it to the main training target until the generated audit files and voltage/OCP/theta consistency are checked.
