# D12-S1 train-inside P2D-like localized correction package

## 目标

D12-S1 不是替换 D9.6/D9.5.1 主线，而是在独立文件中新增一个训练内的
localized P2D-like transport-deficit branch，用于验证低压段 `low_target` / `low_target_le_2p75`
是否能改善，同时通过 normal/rest/high preservation 避免 D11-S8/S9 那种全局曲线破坏。

## 新增文件

```text
gv1/d12_s1_p2d_model.py
  - 新增 raw_p2d_deficit channel

gv1/d12_s1_p2d_transform.py
  - 继承 D9.5.1 output transform
  - 新增 discharge + low-voltage + protocol/high-rate gated P2D-like downward deficit

gv1/d12_s1_p2d_losses.py
  - 继承 D9.5.1 loss
  - 新增 lowtarget/deep coverage + normal/rest/high preservation + correction L2

gv1/d12_s1_p2d_trainer.py
  - 使用 D12-S1 model/transform/loss 的独立 trainer

scripts/gv1_train_d12_s1_p2d_local.py
  - D12-S1 单 profile 训练入口

scripts/gv1_scorecard_d12_s1.py
  - 直接从 prediction.npz 生成 run/segment/mode/decision scorecard

scripts/gv1_run_d12_s1_6profile_40ks.ps1
  - 6-profile 40ks 默认验证脚本：2C/R2.5/R3 各 2 个，排除 battery-8

scripts/gv1_run_d12_s1_6profile_200ks.ps1
  - 6-profile 200ks confirmation，必须显式传入 -Confirm200ks
```

## 默认 6-profile 40ks

默认 profile：

```text
Batch-1_2C_battery-1
Batch-1_2C_battery-2
Batch-3_R2.5_battery-1
Batch-3_R2.5_battery-2
Batch-4_R3_battery-1
Batch-4_R3_battery-2
```

默认 mode：

```text
baseline_d951
  - 调用现有 scripts/gv1_train_conditioned_pinn.py

d12s1_p2d_local_mild
  - 较弱 P2D-like branch，较强 preservation

d12s1_p2d_protocol_guarded
  - protocol/high-rate gated branch，D12-S1 主候选
```

可额外手动加入：

```powershell
-Modes baseline_d951,d12s1_p2d_local_mild,d12s1_p2d_protocol_guarded,d12s1_p2d_lowtarget_focus
```

## 通过标准

`scripts/gv1_scorecard_d12_s1.py` 会生成 `D12_S1_candidate_decisions.csv`。
候选进入 200ks 的标准：

```text
low_target MAE 至少下降 20 mV
low_target_le_2p75 MAE 至少下降 20 mV（若该 profile 有深低压样本）
global MAE 不明显上升，默认不得上升超过 10 mV
corr 不明显下降，默认不得下降超过 0.005
rest/high-target 不明显恶化，默认不得上升超过 20 mV
```

## 推荐运行顺序

1. 覆盖/新增本包文件到项目根目录。
2. 运行 6-profile 40ks。
3. 检查 `D12_S1_candidate_decisions.csv`。
4. 只有有候选 `promote_to_200ks=True`，才运行 6-profile 200ks。
5. 仍不要运行 23-profile，也不要 unflag battery-8。

## 关键边界

- 本包不修改 `main.py`、`util/*`、`integration_spm/*`。
- 本包不覆盖 `gv1/model.py`、`gv1/output_transform.py`、`gv1/losses.py`、`gv1/trainer.py`。
- `metadata_on` 不参与本实验。
- `enable_voltage_hard_clamp=false`。
- battery-8 继续 excluded。
