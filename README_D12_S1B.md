# D12-S1B preservation-tightened train-inside P2D-like correction package

## 背景

D12-S1 6-profile 40ks 已经跑通 18/18，但两个 P2D-like 候选都没有 promotion：
它们能降低 `low_target` / `low_target_le_2p75`，但 `normal_target_gt_3p20` 被明显拉坏，
导致 `global_ok=False`。S1B 因此只做一件事：保留低压修正能力，同时加强 normal-region preservation。

## 与 D12-S1 的区别

新增/修改点：

1. `gv1/d12_s1_p2d_transform.py`
   - 新增 `p2d_low_gate_power` 与 `p2d_pred_low_gate_power`，用于收紧低压 gate。
   - 新增 `p2d_normal_suppression_center_V / width / power`，用 prediction-side baseline 抑制 normal 区域 correction 泄漏。

2. `gv1/d12_s1_p2d_losses.py`
   - 新增 `p2d_normal_bias_preservation`，惩罚 normal 区域 correction 的平均偏置。
   - 新增 `p2d_normal_shift_guard`，惩罚 normal 区域超过小阈值的逐点 shift。

3. `scripts/gv1_scorecard_d12_s1b.py`
   - 输出文件改为 `D12_S1B_*`。
   - promotion 增加 `normal_ok`，并将 global MAE 允许退化从 10 mV 收紧到 5 mV。

4. 新增运行脚本：
   - `scripts/gv1_run_d12_s1b_6profile_40ks.ps1`
   - `scripts/gv1_run_d12_s1b_6profile_200ks.ps1`

## 40ks 默认模式

```text
baseline_d951
d12s1b_p2d_preserve_light
d12s1b_p2d_preserve_mid
d12s1b_p2d_preserve_guarded
```

三个 S1B 候选是 preservation sweep：

- `light`: correction 最弱，normal suppression 最强，优先看 global 是否能守住；
- `mid`: 中等 correction + 较强 normal preservation，是主观察候选；
- `guarded`: correction 略强，但 normal bias/shift guard 也最强。

## 运行顺序

先做 2 epoch smoke：

```powershell
.\scripts\gv1_run_d12_s1b_6profile_40ks.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -Epochs 2 `
  -BatchSize 512 `
  -MaxTimePoints 512 `
  -PredictionTimePoints 256 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

smoke 通过后再跑正式 40ks：

```powershell
.\scripts\gv1_run_d12_s1b_6profile_40ks.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -Epochs 1200 `
  -BatchSize 2048 `
  -MaxTimePoints 4096 `
  -PredictionTimePoints 2048 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

## 40ks promotion 标准

查看：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1b_p2d_preservation_6x40ks_scorecard\D12_S1B_candidate_decisions.csv
```

只有某个候选满足：

```text
low_ok=True
deep_ok=True
global_ok=True
normal_ok=True
corr_ok=True
rest_ok=True
high_ok=True
promote_to_200ks=True
```

才允许进入 200ks。

S1B 的关键目标不是追求更大 low-target 改善，而是：

```text
low_target 改善 >= 20 mV
low_target_le_2p75 改善 >= 20 mV
global MAE 退化 <= 5 mV
normal_target_gt_3p20 退化 <= 5 mV
corr 下降 <= 0.005
rest/high 不明显恶化
```

## 200ks gate

没有 40ks promotion 时不要跑 200ks。若有候选 promotion，再运行：

```powershell
.\scripts\gv1_run_d12_s1b_6profile_200ks.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -Epochs 1800 `
  -BatchSize 2048 `
  -MaxTimePoints 8192 `
  -PredictionTimePoints 4096 `
  -Seed 42 `
  -Device "auto" `
  -Modes baseline_d951,d12s1b_p2d_preserve_mid `
  -Confirm200ks `
  -Clean
```

## 边界

- 不覆盖 D9.6/D9.5.1 主线。
- 不启用 metadata_on。
- 不启用 hard clamp。
- battery-8 仍 excluded。
- 40ks 无 promotion 时不跑 200ks。
