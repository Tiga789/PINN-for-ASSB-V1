# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进不同电池与不同充放电策略下的泛化能力。当前最新工作阶段为 **ASSB-D11 / GV1 low-target correction 诊断阶段**。

## 当前总状态

### ASSB 基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- `SOH` 来自 `ModelFin_112_deterministicSOH_ridge_g4`。
- 这是一个 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它证明当前 ASSB 本电池五目标工程封装成立，但不代表跨电池规格泛化。

### GV1 / XJTU 当前主线

当前 GV1 主线仍为：

```text
D9.6 / D9.5.1 trend-first warmup rare-regime conditioned effective-SPM-style voltage surrogate
```

当前主线证据：

```text
D10-P1 23-profile 200ks excluding / flagging B1_2C battery-8
pass / borderline / fail / read_error = 23 / 0 / 0 / 0
mean_MAE_V ≈ 0.0659
mean_corr ≈ 0.9521
```

重要边界：

- `B1_2C battery-8` 继续作为 late-2C discharge boundary / outlier 被 flag/exclude。
- 不采用 D9.6.1 / D9.6.2 / D9.6.3。
- 不启用 hard voltage clamp。
- 不把 D11 的任何 low-target / P2D-like candidate 升级为主线。

## D11 主要结论

D11 完成了：

```text
D12-S3 clean 23-profile 40ks metadata ablation
D13 segment/protocol diagnosis
D11-S4 low-voltage tail smoke
D11-S5A/S5B/S5C low-target gate/sign/amplitude diagnosis
D11-S6 low-target floor / model-capacity audit
D11-S7 low-voltage escape redesign
D11-S8 P2D-like transport deficit post-hoc correction
D11-S9 trainable localized P2D-like correction
```

总体结论：

```text
D11 没有产生新主线 candidate。
GV1 主线仍保持 D9.6 / D9.5.1。
metadata_on 停止作为主线候选。
low_target / low_target_le_2p75 是当前最大误差瓶颈。
P2D-like 方向有价值，但必须嵌入训练过程，而不能继续做粗 post-hoc correction。
```

## D11 关键实验结果

### D12-S3 metadata ablation

```text
run_count = 69 / 69
strict_completed_metrics_ok = 69
verdict = d12_s3_all_runs_completed_metrics_ok
```

| mode | n | mean_MAE_V | mean_corr | mean_bias_V |
|---|---:|---:|---:|---:|
| off | 23 | 0.07854 | 0.93166 | -0.01650 |
| zero | 23 | 0.07512 | 0.92872 | -0.00049 |
| on | 23 | 0.10353 | 0.92958 | -0.05089 |

结论：

```text
metadata_on 明显劣于 metadata_off，不应替代 D9.6/D9.5.1 主线。
```

### D13 segment/protocol diagnosis

```text
run_count = 92
segment_row_count = 851
verdict = d13_diagnosis_completed
```

主要发现：

```text
最坏误差集中在 low_target / low_target_le_2p75。
rest_I_zero 也有误差，但优先级低于 low-target。
metadata_on 的退步在 R2.5/R3 和部分 2C profile 更明显。
```

### D11-S4 low-voltage tail smoke

```text
run_count = 18 / 18
verdict = d11_s4_all_runs_completed_metrics_ok
```

| mode | mean_MAE_V | mean_corr |
|---|---:|---:|
| baseline_d951 | 0.07299 | 0.93580 |
| lowtail_mild | 0.07056 | 0.93613 |
| lowtail_strong_safe | 0.06799 | 0.93668 |

后续分段表显示：

```text
global MAE 改善，但 low_target / low_target_le_2p75 反而变差。
不能进入 200ks confirmation。
```

### D11-S5A / S5B / S5C

S5A：

```text
run_count = 24 / 24
global MAE 改善
low_target / low_target_le_2p75 变差
lowtarget_gate_probe 与 lowtarget_downward_mild 完全重复
```

S5B：

```text
diagnostic only
low gate 已在 low-target 强激活
low-target 点上预测 100% 高于真实值
候选模式削弱了向下 correction
```

S5C：

```text
run_count = 24 / 24
global MAE 从 0.07299 降到 0.05676
但 low_ok = False
无候选 promotion
```

### D11-S6 floor / model-capacity audit

```text
prediction_count = 24
loaded_run_count = 24
verdict = d11_s6_audit_completed
```

主要发现：

```text
真实 low-target 约 2.5–2.9 V。
模型低压区预测仍常停在约 3.39 V 或更高。
怀疑 output-transform / model-capacity barrier。
```

### D11-S7 low-voltage escape

```text
run_count = 24 / 24
verdict = d11_s7_all_runs_completed_metrics_ok
promoted_candidates = []
```

结果：

- `medium` / `mild`: low_ok=True，但 global_ok=False。
- `strong_guarded`: global_ok=True，但 low_ok=False。

结论：

```text
低压逃逸分支没有形成可升级 candidate。
```

### D11-S8 P2D-like transport deficit correction

```text
run_count = 30 / 30
verdict = d11_s8_scorecard_completed
promoted_candidates = []
```

结果：

- 所有 P2D-like 后处理候选 `low_ok=True`。
- 但所有候选 `global_ok=False`，多数 `corr_ok=False`。
- 说明 P2D-like transport deficit 方向能改善低压段，但粗后处理会破坏全局电压曲线。

### D11-S9 trainable localized P2D-like correction

```text
run_count = 30 / 30
verdict = d11_s9_scorecard_completed
promoted_candidates = []
```

| mode | global MAE | corr | 结论 |
|---|---:|---:|---|
| baseline_copy | 0.07299 | 0.93580 | 参考 |
| p2dtrain_local_mild | 0.27948 | 0.91153 | low_target 改善，但全局严重恶化 |
| p2dtrain_local_medium | 0.39012 | 0.90277 | 同上 |
| p2dtrain_local_guarded | 0.41842 | 0.89858 | 同上 |
| p2dtrain_deeplow_focus | 0.44237 | 0.89622 | 同上 |

结论：

```text
trainable localized post-hoc head 可以显著降低 low_target，
但会严重破坏全局 MAE/corr。
不能进入 200ks，也不能替代主线。
```

## 当前禁止事项

不要执行：

```text
1. 不要把 metadata_on 升级为主线。
2. 不要取消 battery-8 flag。
3. 不要进入 23-profile 或 200ks low-target expansion。
4. 不要把 D11-S4/S5/S7/S8/S9 的任一候选作为主线。
5. 不要继续扩大 post-hoc correction 幅度。
6. 不要恢复 D9.6.1 / D9.6.2 / D9.6.3。
7. 不要默认启用 D11-S7 low-voltage escape patch。
```

## D12 下一步建议

D12 新窗口应从这里继续：

```text
D12-S1: protocol-specific / train-inside P2D-like localized correction design
```

推荐方向：

```text
1. 不是 post-hoc correction，而是把 P2D-like transport deficit head 放进训练过程。
2. 激活条件使用 low gate + discharge gate + SOC/capacity proxy + protocol/high-rate gate。
3. 引入 normal-region preservation loss：
   - all/global MAE
   - rest_I_zero
   - charge/discharge
   - high_target_ge_4p10
4. 以 low_target / low_target_le_2p75 为第一选择标准。
5. 仍然只做 6-profile 40ks smoke。
6. battery-8 继续 excluded。
7. metadata_on disabled。
8. hard clamp disabled。
```

通过标准：

```text
low_target MAE 下降 >= 20 mV
low_target_le_2p75 MAE 下降 >= 20 mV
global MAE 不明显上升
corr 不明显下降
rest_I_zero 不恶化
high_target 不引入 overshoot
```

只有 D12-S1 小窗口通过后，才考虑：

```text
D12-S2: 6-profile 200ks confirmation
```

## 关键缓存目录

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_s3_metadata_ablation_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d13_segment_protocol_diagnosis
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11_s5b_lowtarget_gate_sign_analysis
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11_s6_lowtarget_floor_capacity_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11_s9_trainable_p2dlike_correction_scorecard
```

## 新窗口接续摘要

```text
这是 QJW-2 / PINN-for-ASSB-V1 项目。ASSB 本体基线仍为 ModelFin_112_deterministic_wrapper。GV1 主线仍为 D9.6/D9.5.1，battery-8 flagged/excluded。D11 完成 metadata_on 证伪、segment/protocol 诊断和多轮 low-target/P2D-like correction 试验。所有 D11 correction candidates 均未 promotion。下一步 D12 应设计 train-inside protocol-specific / P2D-like localized correction，不要继续 post-hoc correction 或直接进入 200ks。
```
