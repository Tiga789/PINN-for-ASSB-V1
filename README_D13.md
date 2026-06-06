# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进不同电池体系与不同充放电策略下的泛化能力。当前最新工作阶段为 **ASSB-D13 / XJTU 泛化定位与 D14 工作标准阶段**。

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
- 这是一个 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它证明当前 ASSB 本电池五目标工程封装成立，但不代表跨电池规格泛化。

### XJTU / GV1 当前结果

XJTU 当前推荐的非 outlier 电压泛化结果仍为：

```text
D12-S1K low_plus_transition_fade_to_baseline wrapper
```

当前边界：

- D9.6 / D9.5.1 仍是 GV1 训练主线。
- D12-S1K 是 voltage-wrapper / diagnostic result，不是 full P2D solver。
- 非 `Batch-1_2C_battery-8` 的 23 个 profile 上，transition-fade wrapper 在 200ks confirmation 中达到 mean_MAE≈0.05462 V、mean_corr≈0.97327。
- `Batch-1_2C_battery-8` 继续 flagged，不纳入主线 promotion，可作为 stress-test / target-probe。
- XJTU 当前用于真实公开液态电池 measured-current replay、多 cell、多 protocol、低压尾段和 outlier 处理验证；不用于直接宣称 `cs_a/cs_c/phie/phis_c` 为真实内部状态。

## D13 新增结论

D13 没有执行新的代码覆盖、训练或 scorecard promotion。本阶段完成的是项目故事线、证据边界和 D14 标准的统一。

### 1. 项目任务定义

当前更准确的任务表述是：

```text
I(t), V(t), T(t), cycle/step, CellSpec
→ physics-constrained internal-state inference
→ model-consistent cs_a / cs_c / phie / phis_c / SOH
```

而不是简单的：

```text
I(t) only → V(t) prediction
```

若 V(t) 在推理阶段作为输入，则任务应定义为 inverse inference / internal-state estimation，而不是 pure forward voltage prediction。

### 2. measured-current replay 边界

D13 再次确认：

```text
未提供 I(t) ≠ 非恒流
非恒流工况 + 实测 I(t) = 可以按 measured-current replay 处理
```

当前项目第一阶段把 `I(t)` 作为必要输入。只要实验 record 中有实测电流，CC、CV、CC-CV、CP、脉冲、随机游走等都可以先作为已知边界轨迹处理；controller-solved 模式另列后续扩展。

### 3. Aether-PINN 命名体系

D13 建议的论文/框架命名：

```text
Aether-PINN     # 总框架
Aether-SPM      # ASSB effective-SPM path
Aether-P2Dlite  # 液态电池 P2D-inspired / P2D-like path
Aether-P2D      # 完整 P2D path，可后续扩展
AetherSpec      # CellSpec / ChemistrySpec / resolved spec 系统
AetherLabels    # physics-consistent / XRD-constrained soft labels
```

工程代码中仍可保留 `gv1/`、`specv2/`、`xjtu/` 等明确路径名；Aether 主要用于论文和项目叙事。

## 证据链边界

D13 明确三类实验各自承担不同结论：

| 实验对象 | 主要作用 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| ZHB ASSB + XRD | 固态电池内部状态可信度 | ASSB effective-SPM path 的 soft labels 可被 XRD 部分约束，尤其正极平均锂化状态 | XRD 不能直接证明完整径向分布、`phie/phis_c` 真值 |
| Kokam / JPS paper-replica benchmark | 液态电池 P2D-consistent 内部状态对标 | 未来可公平比较 Aether-PINN 与 Li-style 2GLSTM / LSTM 的状态精度与速度 | 原作者未开源，不能声称 exact hidden labels reproduction |
| XJTU real-data benchmark | 真实公开液态电池多工况泛化 | measured-current replay、多 cell、多 protocol、低压尾段、outlier policy | 不能单独证明内部浓度/电势为真实可信 |

## XRD 可信度定位

`基于XDR的软标签可信度验证.docx` 给出的核心定位：

- XRD 可以较可靠地验证正极 NMC811 的平均锂化状态。
- 对 Li-In/In 负极，XRD 更适合验证相组成、Li/In 库存区间和是否处于 In/InLi 主平台。
- XRD 不应被写成完整 `c_s(r,t)`、`phie`、`phis_c` 的直接真值来源。

推荐论文表述：

```text
XRD-constrained / experimentally constrained / model-consistent internal states
```

不推荐写成：

```text
full internal-state ground truth
```

## Kokam / JPS 对比对象

D13 研究的高水平对比对象：

```text
Li et al., “Physics-informed neural networks for electrode-level state estimation in lithium-ion batteries,” Journal of Power Sources, 2021.
```

关键锚点：

```text
Cell: Kokam SLPB75106100, 7.5 Ah, NMC-graphite pouch cell
Input: x = [I, V, T]^T
Output: y = [cs, css, ce, Φs, Φe]^T
Physics model: P2D electrochemical-thermal model + FVM/EVM
Experimental validation: 25 °C Binder chamber, 1C/2C/2.5C charge/discharge multi-pulse, measured V and surface T
Data generation: 15 dynamic driving cycles × 5 temperatures = 75 data groups
Test profiles: SLCB, WLTP1, WLTP2, WLTP3
```

D13 结论：未来可自建 paper-replica Kokam-P2D-EVM generator，但当前 D14 暂不继续对比实验。

## D14 下一步建议

D14 应继续在 XJTU 数据上体现真实液态电池泛化能力，建议按以下顺序：

1. **冻结 D12-S1K 结果**  
   保留 transition-fade / low-only scorecards，禁止覆盖 D9.6/D9.5.1 主线。

2. **建立 XJTU real-data generalization scorecard**  
   包含 global、segment、protocol、cell-held-out、outlier/stress-test 指标。

3. **完善 envelope / outlier policy**  
   把 battery-8 和未来异常 profile 按预定义规则分为 in-envelope、flagged、out-of-envelope、stress-test。

4. **设计 protocol-held-out 与 cell-held-out split**  
   证明模型不是记住单一电池或单一充放电策略。

5. **考虑扩展 Batch-5 / Batch-6 审计**  
   Batch-5 random walk 和 Batch-6 GEO 可显著增强多工况证据，但必须先按 D8 流程完成 time/cycle/step/replay audit。

6. **低压修正只做 safe residual expert**  
   若低压尾段仍是瓶颈，只允许 bounded、low-only / transition-fade、preservation loss 的 P2D-inspired residual branch；不得重走 S1F/S1G strong high-safe/hard-clamp 路线。

7. **保持 ASSB no-regression**  
   任何 shared encoder / router / SpecV2 修改都不能破坏 ModelFin_112 / ASSB path。

## 当前 README 边界声明

- `ModelFin_112_deterministic_wrapper` 是 ASSB 当前五目标工程基线，不是跨电池泛化模型。
- `D12-S1K transition_fade` 是 XJTU 当前非 outlier voltage-wrapper 推荐结果，不是 full P2D solver。
- XJTU 成功说明真实液态电池 measured-current replay 与多工况外部响应验证有效；不说明内部状态真值。
- ASSB 内部状态可信度应结合 XRD 方案表达为 `XRD-constrained model-consistent states`。
- Kokam/JPS paper-replica benchmark 是未来高水平对比实验方向，D14 当前重点仍是 XJTU real-data generalization。
