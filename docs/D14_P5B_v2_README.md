# D14-P5B-v2 Stable GTX1080Ti Closed-set Precision Package

## Why v2 is needed

Your P5B run reached:

```text
device=NVIDIA GeForce GTX 1080 Ti amp=True gpu_resident=True compiled=True
points=326166 batch_size=65536 steps_per_epoch=5
```

Then training exited before writing `best.pt`, `training_summary.json`, or
`loss_history.csv`.

The most likely failure is `torch.compile` / Triton / Inductor runtime behavior
on GTX 1080 Ti (Pascal). GTX 1080 Ti also has no tensor cores, so AMP does not
give the usual modern-GPU speedup and can introduce instability.

## What v2 changes

```text
torch_compile = false
amp = false
gpu_resident_tensors = true
batch_size = 65536
hidden_dim = 896
```

The goal is still to keep GPU utilization high through large dense batches and
GPU-resident tensors, but without the compile/AMP failure mode.

The PowerShell runner now stops immediately if training fails, instead of
continuing to eval/verify and producing misleading missing-output messages.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p5b_8cell_closedset_precision.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -SoftlabelRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision_v2" `
  -Epochs 500 `
  -BatchSize 65536 `
  -AllowWarn
```

If CUDA OOM occurs:

```text
-BatchSize 32768
```

If memory is still available and GPU utilization is low:

```text
-BatchSize 131072
```

## What to upload for checking

Upload the whole output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision_v2
```

If training fails again, upload at least:

```text
D14_P5B_TRAIN_stderr.log
D14_P5B_TRAIN_stdout.log
D14_P5B_CLOSEDSET_PRECISION_console.log
ModelFin_D14_P5B_8cell_closedset_precision\training_failed.json
```
