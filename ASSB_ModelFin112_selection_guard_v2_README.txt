ASSB ModelFin112 selection/guard v2
====================================

本包只修 SOH head 的 train/val-only checkpoint selection，不改 107A 四状态 core。

修改文件：
  scripts/train_assb111_soh_head.py

新增文件：
  scripts/run_assb112_guarded_seed_sweep_parallel.ps1
  scripts/summarize_assb112_guarded_seed_sweep.py
  input_assb112_phase1_guarded_parallel

核心变化：
  1. visible_guard 从 train_R2 + train_MAE + val_MAE 升级为：
     train_R2, train_MAE, val_MAE, val_R2, val_corr, val_bias_abs,
     val_tail_bias_abs, val_slope_mae, val_range_ratio, visible monotonic penalty。
  2. 训练历史和 checkpoint selection 仍不写入/不使用 test metrics。
  3. 新增 dtype/cuda_matmul_precision/num_threads 参数。
  4. 新增 PowerShell 并行 sweep，默认 MaxParallel=4，用 4 个 seed 进程并行提高 GPU 显存占用和整体吞吐。

推荐流程：
  1. 覆盖修改文件和新增文件。
  2. 运行并行 sweep：
     .\scripts\run_assb112_guarded_seed_sweep_parallel.ps1 -MaxParallel 4
  3. 汇总结果：
     & D:\Anaconda\envs\torchgpu\python.exe .\scripts\summarize_assb112_guarded_seed_sweep.py `
       --model_prefix .\ModelFin_112_guardedSOH_seed `
       --seeds 7,42,2026,3407,7890 `
       --output_dir .\EvalFin_112_guarded_soh_sweep_v2

若 CUDA OOM：
  把 -MaxParallel 4 改成 -MaxParallel 2。

若希望进一步提速且接受数值轻微变化：
  加 -UseFloat32FastMode。正式结果建议先用 float64。
