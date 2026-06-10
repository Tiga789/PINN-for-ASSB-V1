# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM / P2Dlite 思路，先完成 NMC811||Li-In 全固态电池（ASSB）本体建模，再推进 XJTU / Kokam / 多路径泛化。当前最新工作阶段为 **ASSB-D15 / XJTU P2Dlite-RG 55-cell soft-label 全覆盖阶段**。

## 当前总状态

### ASSB 基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- 四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- SOH 来自 `ModelFin_112_deterministicSOH_ridge_g4`。
- 它是 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它证明 ASSB 本电池五目标工程封装成立，但不代表跨电池规格泛化。

### XJTU 电压基线

XJTU voltage replay 当前基线仍为：

```text
D12-S1K low_plus_transition_fade_to_baseline voltage-wrapper
D9.6 / D9.5.1 training mainline
```

D15 未覆盖或重写 D12-S1K，也没有改变 battery-8 flagged/outlier 边界。

### D15 / P2Dlite-RG 当前状态

D15 已完成 XJTU P2Dlite-RG soft-label 55-cell 全覆盖。最终统一目录：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
```

最终 consolidation summary：

```text
expected_cell_count = 55
actual_cell_count   = 55
missing_count       = 0
extra_count         = 0
duplicate_warning_count = 0
total_size_GB       = 52.742
status              = PASS
```

Batch 覆盖情况：

| Batch | Cell count | D15 coverage |
|---|---:|---|
| Batch-1 | 8 / 8 | complete |
| Batch-2 | 15 / 15 | complete |
| Batch-3 | 8 / 8 | complete |
| Batch-4 | 8 / 8 | complete |
| Batch-5 | 8 / 8 | complete |
| Batch-6 | 8 / 8 | complete |

## D15 做了什么

D15 从 D14 暴露的 `cs_a / cs_c` 径向梯度过弱问题出发，完成了以下链路：

```text
D15-P0  8-cell P2Dlite-RG generator + radial audit
D15-P1  8-cell closed-set NN smoke
D15-P2  8-cell precision benchmark（hard gates PASS, final REVIEW）
D15-P3  Batch-2 3-cell applicability validation
D15-P3B Batch-2 theta projection repair
D15-P3C Batch-2 15-cell expansion
D15-P4A / P4A-fix remaining 32-cell replay readiness audit
D15-P4B Batch-1/3/4 remaining-ready 18-cell generation
D15-P4C Batch-5/6 remaining 14-cell replay profile completion
D15-P4D Batch-5/6 remaining 14-cell generation + targeted Batch-5_battery-8 fix
D15 ALL55 final consolidation
```

## 当前允许声明的结论

可以声明：

- XJTU 55/55 cells 已具备 P2Dlite-RG model-consistent soft-label folders。
- P2Dlite-RG generator 和 radial audit 已覆盖 XJTU 六个 batch。
- Batch-2 3C charge / 1C discharge stress-test 已完成 15-cell generator/radial audit 和 projection-repaired NN benchmark。
- Batch-5/6 random-walk / GEO remaining cells 已补齐 replay profiles 并完成 soft-label generation。
- D15 labels 可以作为后续 Aether-P2Dlite / GV1 NN 训练与 transfer evaluation 的统一 soft-label 数据集。

不能声明：

- `cs_a / cs_c / phie / phis_c` 是实验直接测得的真实内部状态。
- D15 已证明 held-out cell 泛化。
- D15-P2 是 clean PASS。
- Batch-2 raw NN 不加 theta projection 已 clean PASS。
- P2Dlite-RG 是 full P2D solver。

## 重要路径

```text
# Final ALL55 soft labels
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL

# Final manifest / summary
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL/D15_ALL55_SOFTLABEL_MANIFEST.csv
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL/D15_ALL55_SOFTLABEL_SUMMARY.json

# Replay profiles to keep
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profiles
E:/XJTU battery dataset/_gv1_cache/xjtu_batch2_replay_profiles_d15p3
E:/XJTU battery dataset/_gv1_cache/xjtu_batch56_remaining14_replay_profiles_d15p4c

# Old P2Dlite v1 baseline to keep for comparison
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_v1_p4b_multicell_v3
```

## D15 工程经验

1. 大规模 P2Dlite-RG soft-label generation 当前是 NumPy/CPU 后端，GPU 不会自动参与。
2. 简单提高 `Workers` 不能保证高 CPU 利用率；当内存接近上限时会触发换页和假死。
3. 后续大规模生成必须支持 `resume / skip completed / per-cell log / resource monitor`。
4. 交付 zip 必须检查：不含 `__pycache__`、不含 `*.pyc`、不含无关 `gv1/__init__.py`。
5. C 盘 SSD staging 可用于加速，但最终数据必须归档回 `E:/XJTU battery dataset/_gv1_cache`。

## D16 下一步建议

下一步不要直接训练新的 55-cell 模型。推荐先做：

```text
D16-P5A：ALL55 existing-model transfer evaluation
```

目标是评估已有 D15-P2 / Batch-2 projection-repaired 模型在 ALL55 上的迁移能力。评估维度至少包括：

```text
seen vs unseen cells
Batch-1 / Batch-2 / Batch-3 / Batch-4 / Batch-5 / Batch-6
raw theta vs projected theta
phis_c / phie / theta_a / theta_c / gradient metrics
current transition / high-target / low-target / rest segments
```

只有当 existing-model transfer 明显不足时，再进入：

```text
D16-P5B：ALL55 unified NN training benchmark
```

D16 若要证明泛化，必须进一步设计 held-out cell / held-out protocol split，不能只依赖 ALL55 closed-set training。

## 当前边界说明

D15 的核心成果是 **XJTU 55-cell P2Dlite-RG soft-label coverage**，不是实验内部状态真值验证。若论文或 README 需要表达，应使用：

```text
model-consistent internal-state soft labels
radial-gradient-aware P2Dlite labels
P2Dlite-RG generator / audit / NN reproduction chain
```

避免使用：

```text
真实 cs_a/cs_c 真值
full-P2D ground truth
held-out 泛化已证明
```

## Project status handoff

进入 D16 窗口时，请先阅读 `ASSB-D15.docx`。D16 应从 ALL55 existing-model transfer evaluation 开始，不要覆盖 ASSB baseline、D12-S1K voltage baseline 或 D15_ALL55_FINAL。
