# D14-P5 XJTU P2Dlite Soft-label Neural-network Smoke

## Position

D14-P5 follows D14-P4B-v3. P4B-v3 generated P2Dlite model-consistent soft labels
for eight selected XJTU profiles:

```text
Batch-1 / 2C:        battery-1, battery-3
Batch-3 / R2.5:      battery-6, battery-7
Batch-4 / R3:        battery-2, battery-7
Batch-5 / random:    battery-7
Batch-6 / GEO:       battery-3
```

P5 is the first neural-network smoke that reads those soft labels and learns to
predict:

```text
theta_a(t, r), theta_c(t, r), phie(t), phis_c(t)
```

It does **not** regenerate soft labels and does **not** generate SOH.

## Task definition

The default P5 task is:

```text
I(t), V(t), T(t), step_type, batch/protocol metadata
  -> P2Dlite soft-label states
```

This is voltage-current-informed internal-state inference, not pure forward
voltage prediction. Using `voltage_exp` as an input is intentional in this smoke.

## Default split

```text
train:
  Batch-1 battery-1
  Batch-3 battery-6
  Batch-4 battery-2
  Batch-5 battery-7

val:
  Batch-1 battery-3
  Batch-3 battery-7

test:
  Batch-4 battery-7
  Batch-6 battery-3
```

Batch-6 battery-3 is therefore a held-out GEO test profile in the default smoke.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p5_xjtu_p2dlite_nn_smoke.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -SoftlabelRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3" `
  -PriorFile "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5_p2dlite_nn_smoke" `
  -Epochs 120 `
  -BatchSize 2048 `
  -AllowWarn
```

## Expected runtime

This is a smoke run, not a final training run. It samples each profile to keep
memory and runtime bounded. If runtime is high, reduce:

```text
-Epochs 40
-BatchSize 1024
```

## Boundaries

- No P2Dlite soft-label generation.
- No SOH generation.
- No full-P2D internal-state truth claim.
- No modification to GV1 mainline files:
  - `gv1/model.py`
  - `gv1/output_transform.py`
  - `gv1/losses.py`
  - `gv1/trainer.py`
