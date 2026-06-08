# D14-P5B XJTU P2Dlite 8-cell Closed-set Precision Benchmark

## Position

D14-P5B is **not** a data expansion step. It uses the same eight batteries from
P4B-v3 and trains a higher-capacity neural network in a closed-set calibration
mode.

This means:

```text
All 8 profiles are used for training.
The same 8 profiles are used for evaluation.
```

The goal is to test whether the neural-network side can reproduce the P2Dlite
soft labels at precision comparable to the earlier ASSB closed-set benchmarks.
It is not a held-out generalization claim.

## GPU utilization strategy

Compared with P5, P5B is designed to drive the GPU harder:

```text
1. larger MLP: hidden_dim=768, num_layers=6
2. richer Fourier time/charge features
3. profile one-hot conditioning
4. GPU-resident tensors by default
5. large batch size: 65536
6. AMP mixed precision enabled by default
7. torch.compile enabled when supported
```

If you see CUDA out-of-memory, rerun with:

```text
-BatchSize 32768
```

If GPU utilization is still low and memory is available, rerun with:

```text
-BatchSize 131072
```

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p5b_8cell_closedset_precision.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -SoftlabelRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision" `
  -Epochs 500 `
  -BatchSize 65536 `
  -AllowWarn
```

For a faster first smoke:

```powershell
-Epochs 120 `
-BatchSize 65536 `
```

## Expected target

Because this is closed-set calibration, the desired level is stricter than P5:

```text
mean phis_c MAE < 0.010–0.015 V ideally
mean phie MAE   < 0.010–0.015
mean theta MAE  < 0.015–0.020
```

If P5B cannot reach this level, the next action should be model/feature/loss
improvement, not adding more batteries.
