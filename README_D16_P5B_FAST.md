# D16-P5B FAST training patch

This package keeps the original train6/eval49 logic and data boundary, but replaces the trainer with a GPU-resident tensor training loop.

It still reads only observed `t/I/V` during training. It does not read or directly supervise `theta/cs/phie/phis_c` soft-label arrays.

## Why this patch exists

The first trainer used a CPU `DataLoader` and validated every epoch. On small MLP models this causes low GPU utilization. The fast trainer:

- loads sampled train/val feature tensors once;
- moves them to GPU once;
- uses GPU-resident random/permutation batch indexing;
- validates every `ValEvery` epochs instead of every epoch;
- keeps the same checkpoint schema `model/best_with_state.pt`.

## Smoke train

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5b_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

## Formal training + eval

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5b_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is still low and utilization remains low, try:

```powershell
-BatchSize 262144
```

If memory errors occur, fall back to:

```powershell
-BatchSize 65536
```

Outputs are written to:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5b_train6_eval49_observation_physics_FAST
```
