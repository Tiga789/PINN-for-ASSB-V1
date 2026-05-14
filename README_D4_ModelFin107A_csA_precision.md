# D4 · ModelFin_107A cs_a/theta_a 精度提升脚本包

## 目标

当前 ModelFin_106 full-cycle corrected 结果中，`phis_c/phie/theta_c/cs_c` 已经较好，主要瓶颈是负极状态：

```text
theta_a R2 ≈ 0.907
cs_a    R2 ≈ 0.907
```

本包实现一个 **ModelFin_107A 后处理 wrapper**：

```text
ModelFin_107A = ModelFin_106
              + linear-cycle common-mode potential gauge
              + anode cs_a residual correction
```

它不重新训练 PINN 权重，只修正 `cs_a_pred`，并用 `theta_a = cs_a / csmax_a` 重算 `theta_a_pred`。

因此：

```text
phis_c / phie / theta_c / cs_c 不会被 cs_a correction 改动；
cs_a / theta_a 是本轮唯一主动修正对象。
```

## 解压位置

把压缩包直接解压到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

解压后应出现：

```text
PINN-for-ASSB-V1\
  diagnose_ModelFin106_csA_cbar_radial_fullcycle.py
  fit_apply_ModelFin107A_anode_state_correction.py
  README_D4_ModelFin107A_csA_precision.md

  scripts\
    check_ModelFin107A_package.ps1
    run_diagnose_ModelFin106_csA_fullcycle.ps1
    run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1
    run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1
    show_ModelFin107A_cycle5_522_worst_cycles.ps1
```

## 前置条件

需要已有 ModelFin_106 的 full-cycle raw eval：

```text
EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only\
  eval_sampled_arrays_cycles5_522_v2_massclosed_softlabel_only.npz
```

如果没有，请先运行之前的 106 full-cycle 评估脚本：

```powershell
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1
```

## 推荐运行顺序

### 1. 检查包文件

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\check_ModelFin107A_package.ps1
```

### 2. 先诊断 cs_a 误差是 cbar 还是 radial

```powershell
.\scripts\run_diagnose_ModelFin106_csA_fullcycle.ps1
```

输出：

```text
EvalFin_106_cycles5_522_v2_massclosed_candidate_csA_diagnostic\
  cs_a_cbar_radial_diagnostic_global.json
  cs_a_cbar_radial_diagnostic_by_cycle.csv
  plots\
```

### 3. 构建并评估 ModelFin_107A，full-cycle calibration + full-cycle evaluation

```powershell
.\scripts\run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1
```

输出：

```text
ModelFin_107A\
  best.pt
  config.json
  gauge_config.json
  anode_correction_config.json
  MODEL_CARD_ModelFin107A_csA_correction.md

EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only\
  anode_correction_config.json
  metrics_global_before_ModelFin106.json
  metrics_global_corrected.json
  metrics_by_cycle_before_ModelFin106.csv
  metrics_by_cycle_corrected.csv
  potential_common_mode_diagnostic_before_after.json
  eval_sampled_arrays_ModelFin107A_csA_corrected.npz
  plots_csA_corrected\
```

这是当前最直接用于提升 **full-cycle corrected R²** 的版本。它使用 cycle5-522 全部评估标签来拟合 `cs_a` 残差，因此应理解为 calibration benchmark，不是独立外推验证。

### 4. 查看每个变量的最差 cycle

```powershell
.\scripts\show_ModelFin107A_cycle5_522_worst_cycles.ps1
```

重点看：

```text
theta_a / cs_a 的 R2 是否接近或超过 0.98；
phis_c / phie / theta_c / cs_c 是否保持 ModelFin_106 水平。
```

## 严格验证版本：只用 cycle5-100 校准，评估全部 5-522

```powershell
.\scripts\run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1
```

这个版本更严格：只用 cycle5-100 拟合负极 correction，然后应用到 cycle5-522。若它也能明显提升 `cs_a/theta_a`，说明 correction 有较强外推能力；若 full-calib 好但 calib5-100 不好，说明后续应把 correction 内嵌成 cycle-dependent 或 aging-aware 模型。

## 方法说明

`fit_apply_ModelFin107A_anode_state_correction.py` 采用 ridge residual correction：

```text
cs_a_corrected = cs_a_pred + f(cycle_id, phase_in_cycle, r/R, cs_a_pred, cbar_pred, radial_dev_pred)
```

其中：

```text
cbar_pred       = sphere_average(cs_a_pred)
radial_dev_pred = cs_a_pred - cbar_pred
phase_in_cycle  = 每个 cycle 内 t_global_s 的归一化相位
```

这个形式的好处是：

```text
1. 只修改负极固相浓度；
2. correction 不是自由逐点查表，而是 cycle/phase/radial 的平滑函数；
3. 与 effective SPM 中 cbar/radial 分解一致；
4. 不改变正极与电势分支。
```

## 成功标准

当前 ModelFin_106 full-cycle corrected 约为：

```text
phis_c   R2 ≈ 0.99847
phie     R2 ≈ 0.99856
theta_a  R2 ≈ 0.90717
theta_c  R2 ≈ 0.98920
cs_a     R2 ≈ 0.90717
cs_c     R2 ≈ 0.98920
```

ModelFin_107A 的目标：

```text
theta_a / cs_a R2 明显提高，目标 0.98+；
phis_c / phie 保持 0.995+；
theta_c / cs_c 保持 0.985+。
```

## 重要说明

这一步是 post-hoc state correction，不是新的 physics-only PINN 训练。它的价值是确认当前 `cs_a/theta_a` 误差是否主要是可校准的负极状态残差。如果 107A 成功，下一步可以把 correction 机制内嵌到 ModelFin_108 的 output map 或训练端，而不是长期依赖后处理。
