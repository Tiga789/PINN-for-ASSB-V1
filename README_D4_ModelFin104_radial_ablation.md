# D4 ModelFin_104 radial ablation + training package

解压位置：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

## 文件放置位置

```text
项目根目录/
  evaluate_assb_radial_ablation_from_eval_npz.py
  evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py
  input_assb_cycles5to522_v2_massclosed_ID104_radial010
  README_D4_ModelFin104_radial_ablation.md
  scripts/
    run_ablation_ModelFin103_v2_massclosed_cycle5_100.ps1
    run_train_ModelFin104_v2_massclosed_cycle5_20.ps1
    run_eval_ModelFin104_v2_massclosed_cycle5_100.ps1
    check_ModelFin104_config.ps1
```

## 1. 先做评估端 ablation，不训练

目的：验证 ModelFin_103 在 v2 massclosed 标签上的 cs_c/theta_c 误差是否主要来自正极径向偏差。

运行：

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\run_ablation_ModelFin103_v2_massclosed_cycle5_100.ps1
```

输出：

```text
EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only\radial_ablation_cs_c\
  ablation_summary.json
  ablation_radial_scale_global.csv
  ablation_radial_scale_by_cycle.csv
  plots\per_cycle_mae_cs_c_radial_ablation.png
  plots\per_cycle_mae_theta_c_radial_ablation.png
```

解释：

```text
scale = 1.0  原始 ModelFin_103 正极径向偏差
scale = 0.0  保留预测 cbar，只强制 cs_c 径向均匀
scale = 0.1  保留 10% 正极径向偏差
```

如果 scale=0 或 scale=0.1 显著优于 scale=1，说明 D4 的主问题已经从 mass closure 转到正极 radial ansatz。

## 2. 训练 ModelFin_104

ModelFin_104 只做一个关键改动：

```text
CBAR_BASELINE_DEVIATION_FRACTION_C : 0.10
```

其余保持 ModelFin_103/ID101 主结构：

```text
training slice: cycle5-20
soft labels:    ..\assb_soft_labels_cycle5_522_v2_massclosed_candidate
A radial:       zero-mean, fraction_A=0.15
C radial:       zero-mean, fraction_C=0.10
potential:      current-aware baseline enabled
loss:           physics-only, data loss off
```

运行：

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\run_train_ModelFin104_v2_massclosed_cycle5_20.ps1
```

训练后检查 config：

```powershell
.\scripts\check_ModelFin104_config.ps1
```

重点确认：

```text
ID = 104
ASSB_SOFT_LABEL_DIR -> v2_massclosed_candidate
ASSB_CYCLE_FROM = 5
ASSB_CYCLE_TO = 20
CBAR_BASELINE_DEVIATION_FRACTION_C = 0.10
USE_ZERO_MEAN_RADIAL_DEVIATION_C = True
```

## 3. 评估 ModelFin_104 on cycle5-100

运行：

```powershell
.\scripts\run_eval_ModelFin104_v2_massclosed_cycle5_100.ps1
```

输出：

```text
EvalFin_104_cycles5_100_v2_massclosed_candidate_cRadial010_softlabel_only\
  metrics_global.json
  metrics_by_cycle.csv
  debug_model_and_data.json
  eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz
  plots_softlabel_only\*.png
```

通过/失败初判：

```text
优先看 theta_c/cs_c 是否比 ModelFin_103-v2 的 MAE=0.0405 / 2.098 明显下降。
同时确认 phis_c 不要从 0.0228 V 退化到过大。
cs_a/theta_a 理论上不应被本次正极径向修改显著影响。
```
