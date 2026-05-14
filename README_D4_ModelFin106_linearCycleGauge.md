# D4 / ModelFin_106 linear-cycle common-mode gauge package

## 核心定义

`ModelFin_106 = ModelFin_105 + linear_cycle_mean common-mode gauge correction`。

这不是重新训练一个新的 PINN 权重，而是把当前最优的 `ModelFin_105 + linear_cycle_mean common-mode correction` 固化成一个可复现的 `ModelFin_106` 模型目录：

```text
common_mode_error(cycle) = 0.5 * [(phie_pred - phie_true) + (phis_c_pred - phis_c_true)]
fitted_bias(cycle)      = slope * cycle_id + intercept
offset_to_add(cycle)    = -fitted_bias(cycle)

phie_pred_106   = phie_pred_105   + offset_to_add(cycle_id)
phis_c_pred_106 = phis_c_pred_105 + offset_to_add(cycle_id)
```

这个修正只动 `phie_pred` 和 `phis_c_pred` 的公共电势基准；不改变 `theta_a/theta_c/cs_a/cs_c`，也不改变 `phis_c - phie` 差分。

## 为什么 106 不重新训练

105 的结果显示：

```text
phie / phis_c：主要是 common-mode 负偏置
phis_c - phie：差分误差已经很小
theta_c / cs_c：已经保持 104 的优秀水平
```

方法对比显示 `linear_cycle_mean` correction 明显优于常数偏移，并且能把 common-mode MAE 降到 mV 级。因此 106 先做最小、可解释、可复现的两参数 gauge 层，而不是继续盲目微调神经网络。

## 解压位置

把压缩包解压到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

解压后应出现：

```text
PINN-for-ASSB-V1\
  build_ModelFin106_from_ModelFin105_linearCycleGauge.py
  apply_ModelFin106_linear_cycle_gauge.py
  evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py
  README_D4_ModelFin106_linearCycleGauge.md

  scripts\
    run_build_ModelFin106_linearCycleGauge.ps1
    check_ModelFin106_linearGauge_config.ps1
    run_eval_ModelFin106_v2_massclosed_cycle5_100_linearGauge.ps1
    run_apply_ModelFin106_gauge_only.ps1
    run_all_ModelFin106_linearCycleGauge_cycle5_100.ps1
    run_train_ModelFin106_v2_massclosed_linearCycleGauge.ps1
```

## 前置条件

需要本地已有：

```text
ModelFin_105\best.pt
ModelFin_105\config.json
EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only\eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz
```

如果 105 评估目录不存在，先运行：

```powershell
.\scripts\run_eval_ModelFin105_v2_massclosed_cycle5_100.ps1
```

## 推荐运行顺序

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

.\scripts\run_build_ModelFin106_linearCycleGauge.ps1
.\scripts\check_ModelFin106_linearGauge_config.ps1
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_100_linearGauge.ps1
```

也可以一键运行：

```powershell
.\scripts\run_all_ModelFin106_linearCycleGauge_cycle5_100.ps1
```

## 输出目录

构建模型目录：

```text
ModelFin_106\
  best.pt                         # 从 ModelFin_105 复制
  config.json                     # 从 ModelFin_105 复制并加 wrapper 元数据
  gauge_config.json               # 线性 cycle common-mode gauge 参数
  MODEL_CARD_ModelFin106_linearCycleGauge.md
```

评估目录：

```text
EvalFin_106_cycles5_100_v2_massclosed_candidate_linearCycleGauge_softlabel_only\
  metrics_global_corrected.json
  metrics_by_cycle_corrected.csv
  metrics_global_before_raw.json
  metrics_by_cycle_before_raw.csv
  gauge_application_summary.json
  potential_common_mode_diagnostic_before_after.json
  eval_sampled_arrays_ModelFin106_linearGauge_corrected.npz
  plots_linear_gauge_corrected\
```

另外会有 raw 评估目录：

```text
EvalFin_106_cycles5_100_v2_massclosed_candidate_linearGauge_raw_softlabel_only\
```

raw 结果应基本等同于 105，因为 106 的 neural checkpoint 继承自 105；最终看 corrected 目录。

## 预期成功指标

应接近你之前的 `ModelFin_105 + linear_cycle_mean` 结果：

```text
phis_c MAE ≈ 0.00725 V
phie   MAE ≈ 0.00151 V
theta_c MAE ≈ 0.00566
cs_c    MAE ≈ 0.29308
common_mode MAE ≈ 0.00365 V
differential MAE ≈ 0.00723 V
```

如果 corrected 指标接近上述结果，说明 D4 当前最优方案已成功固化为 ModelFin_106。

## 注意事项

1. `run_train_ModelFin106_v2_massclosed_linearCycleGauge.ps1` 只是为了命名习惯保留的 wrapper；它不会做 SGD/L-BFGS 训练，只会构建 106 gauge 模型目录。
2. `ModelFin_106` 复制了 `ModelFin_105/best.pt`，因此如果以后删除 105，106 仍能独立做 raw evaluation。
3. 这一步的物理含义是公共电势 gauge 校准，不是新的 SOH/aging 模型。后续若扩展到 cycle5-200/522，再判断 linear-cycle gauge 是否仍能外推。
