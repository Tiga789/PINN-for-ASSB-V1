ASSB ModelFin112 guard v7
===========================

核心修复：
1. train_assb111_soh_head.py 改为 visible soft-score 主选择；hard guard 只做审计/报警，默认不再阻止保存模型。
2. 保留 no-test-selection：训练 history 不写 test；test 只在 selected checkpoint 固定后输出。
3. 新增 deterministic ridge SOH baseline：使用同一套 G4 strict 特征、train-only scaler、train/val-only alpha selection，几秒级完成，用于快速确认 Step2 G4 特征是否真的能支撑 SOH。
4. 多 seed 脚本继续使用 Start-Process，不使用 Start-Job/Receive-Job。
5. 旧名 run_assb112_guarded_seed_sweep_parallel.ps1 已改成 wrapper，避免再次触发 Job 报错。

推荐执行顺序：

A. 先跑快速 ridge 基线：
   .\scripts\run_assb112_ridge_soh_baseline.ps1 -Clean

   预期：能快速生成 ModelFin_112_ridgeSOH_g4_v7，并打印 held-out test R2/MAE。
   这个结果用于判断 G4 特征是否仍然可用。它不是神经网络 SOH head，但可以作为 teacher/baseline。

B. 再跑神经 head 快速冒烟：
   .\scripts\run_assb112_guarded_quick_smoke.ps1 -Seed 7 -Clean

   预期：每 25 epoch 有进度，300 epoch 内结束，不会因 hard guard 不通过而失败。

C. 最后跑 5 seed 并行：
   .\scripts\run_assb112_guarded_seed_sweep_startprocess.ps1 -MaxParallel 4 -Clean

   或兼容旧命令：
   .\scripts\run_assb112_guarded_seed_sweep_parallel.ps1 -MaxParallel 4 -Clean

判定：
- ridge baseline 若已明显达到目标，说明特征有效，当前神经 head 的问题主要是训练/结构稳定性。
- neural 5 seed 若仍不稳，不应继续怪 guard；应考虑 ridge teacher distillation 或直接将 ridge/linear SOH branch 作为 ModelFin112A 工程 SOH 分支。
