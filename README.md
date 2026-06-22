# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES、effective SPM 与 P2Dlite-RG，先完成 NMC811||Li-In 全固态电池（ASSB）工程基线，再推进 XJTU 55-cell model-consistent internal-state surrogate。当前最新阶段为 **ASSB-D18 / FORMAL55-DEPLOY**。

## 当前总状态

### ASSB 本体基线

ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

- `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A`。
- SOH 来自 deterministic ridge head。
- 这是 ASSB 本电池的 engineering wrapper，不是端到端联合网络，也不是跨电池规格泛化模型。

### XJTU D18 当前工程模型

当前冻结工程版本为：

```text
FORMAL55-DEPLOY
```

准确定位：

```text
55-cell closed-set
+ Step2/P2Dlite-RG-assisted
+ protocol-routed
+ per-cell calibrated
+ model-consistent internal-state engineering surrogate
```

它不是：

```text
实验真实内部状态 ground truth predictor
跨 cell generalization model
raw I/V/T-only independent solver
```

## D18 已完成的主要工作

1. 放弃不可辨识的 no-state-label 唯一反演目标，将任务改为复现固定 Step2/P2Dlite-RG teacher。
2. 冻结 Step2、55-cell manifest、split、source priority 与 hash。
3. 建立 33 fit / 6 internal / 7 validation / 8 frozen-test / 1 flagged 的角色体系。
4. 从 zero-residual scaffold 逐步构建 protocol-specific concentration/radial residual。
5. 冻结 `phie/phis_c` 与 cbar 主轨迹，theta 由 cs 推导。
6. 形成 3C/R2.5/R3/random_walk learned specialists、2C conservative base、GEO conservative/reference route。
7. 为 55 cells 生成 per-cell adapter 或 parent passthrough。
8. 构建紧凑 `FORMAL55-DEPLOY` bundle、runtime、adapter registry 与 confidence ledger。
9. 完成 55/55 operational audit 和全 cycle streaming performance audit。
10. 交付 JSON-driven selected-cycle inference、soft-label audit 与交互式 3D 绘图工具。

## 当前关键结果

### Deploy bundle

| 项目 | 结果 |
|---|---:|
| cells | 55/55 |
| adapters | 55 |
| parent artifacts | 40 |
| bundle files | 118 |
| bundle size | 6.37 MiB |
| HIGH confidence | 45 |
| MEDIUM confidence | 4 |
| LOW confidence | 6 |

### 55/55 bounded operational audit

```text
55/55 cells PASS
605/605 runtime checks PASS
118/118 bundle hashes PASS
40/40 parent hashes PASS
finite / theta bounds / zero-mean / cbar reconstruction PASS
```

该 audit 每 cell 只抽取一个 cycle、最多 128 点，只证明运行完整性，不是性能结论。

### 全 cycle streaming performance audit

审计覆盖：

```text
55 cells
26,606 / 26,606 cycles
每 cycle 最多 64 个确定性事件感知点
```

结果：

| 指标 | 结果 |
|---|---:|
| cell-balanced primary mean R² | 0.9163 |
| cell-balanced q10 of cycle-q10 R² | 0.6359 |
| complete predictive cycles | 26,244 |
| negative complete cycles | 242 (0.922%) |
| R² ≥ 0.7 cycle fraction | 88.13% |
| suspicious exact metrics | 0 |
| verdict | ACCEPT_WITH_LIMITATIONS |

协议级：

| Protocol | Mean R² | Cycle q10 | R²≥0.7 cycle fraction |
|---|---:|---:|---:|
| R3 | 0.9799 | 0.9384 | 100.0% |
| 3C | 0.9745 | 0.9415 | 97.91% |
| R2.5 | 0.9518 | 0.8601 | 99.98% |
| 2C | 0.9369 | 0.8531 | 100.0% |
| random_walk | 0.8985 | 0.5085 | 91.76% |
| GEO | 0.7053 | 0.0923 | 60.0% |

注意：这是**全部 cycles + 每 cycle 最多64点**的事件感知审计，不是全部 1 Hz 原始点逐点审计。

## 当前主要限制

- GEO / Batch-6 是主要径向细节短板，尤其 battery-5 / battery-6。
- random_walk battery-8 是明显 outlier/weak cell。
- 3C battery-3 有极少数低方差灾难 cycle，整体主体仍较好。
- `phie/phis_c` 当前为 Step2/reference-only，不报告 learned predictive R²。
- 总 `cs` R² 可能被准确 `cbar` 主轨迹抬高，必须同时查看：

```text
delta_cs R²
surface-minus-mean R²
surface-center gradient R²
radial-energy R²
```

- XJTU soft labels 是 P2Dlite-RG model-consistent teacher，不是实验 ground truth。
- 当前是 closed-set per-cell calibrated deployment，不支持跨 cell 泛化主张。

## 关键路径

### 数据与模型

```text
项目根：
C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1

ALL55 P2Dlite-RG soft labels：
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL

FORMAL55-DEPLOY：
C:/Users/Tiga_QJW/Desktop/XJTUstation/D18/Formal-A/Deploy_build/
D18_DEPLOY_BUILD_OUTPUT/D18_FORMAL55_DEPLOY_BUILD_20260621_215009/
MODELFIN_D18_FORMAL55_DEPLOY

All-cycle audit：
E:/XJTU battery dataset/_gv1_cache/d18_formal55_allcycle_streaming_audit/
D18_FORMAL55_ALLCYCLE_STREAMING_AUDIT_20260622_004323
```

## Selected-cycle 按需推理与 3D 绘图

工具目录：

```text
formal55_selected_cycle_tool
```

任务 JSON：

```text
formal55_selected_cycle_tool/configs/selected_cycle_request.json
```

示例：

```json
{
  "selection": {
    "batch": 2,
    "battery": 5,
    "cycles": "35-37"
  }
}
```

运行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\formal55_selected_cycle_tool\scripts\run_formal55_selected_cycle.ps1 `
  -RequestJson ".\formal55_selected_cycle_tool\configs\selected_cycle_request.json"
```

执行顺序：

```text
读取目标 cycle 之前的历史
→ 生成冻结 baseline / parent route
→ 目标 cycles 全原始时间点推理
→ 预测固定在内存
→ 再读取 soft-label target
→ 计算全局与逐 cycle 指标
→ 打开 cs_a/cs_c prediction/reference 四张可旋转3D图
```

默认不保存大型预测 NPZ。指标使用目标 cycle 全部时间点；只有 3D surface 为绘图流畅可降采样。

### 当前图形解释示例

Batch-6 GEO battery-5 cycles 45-47：

```text
cathode total cs_c R² ≈ 0.9824
NMAE ≈ 0.0326
NRMSE ≈ 0.0418
delta_cs R² ≈ 0.2692
```

说明平均库存与总 cycle 趋势复现较好，但径向偏差/gradient 仍弱。不能只看总浓度曲面。

## 固定工程规则

1. 新阶段使用独立目录，默认不覆盖旧文件。
2. 大缓存和输出放 E 盘，不把 50GB 级文件写入项目根目录。
3. 修改阶段先做 5–15 分钟 smoke；无高把握不得直接安排全量重审计。
4. 包交付前至少执行 `py_compile`、`--help`、参数静态检查和合成端到端测试。
5. UID 必须按 canonical batch/protocol/battery 精确匹配，禁止 substring。
6. 预测必须先于 target state 读取；`R²≈1 + MAE≈0` 的 learned 输出触发泄露审计。
7. 总状态指标与径向专属指标必须同时报告。
8. reference-only、fallback、learned route 必须分开统计。
9. 必须明确审计范围：cells、cycles、points/cycle、是否全原始点、是否 closed-set。

## D19 下一步

优先级：

1. 升级 selected-cycle 工具：新增 `delta_cs` prediction/reference/error surface、surface-center gradient、radial energy 图。
2. 从 all-cycle CSV 自动选择 GEO battery-5/6、random_walk battery-8、3C battery-3 的 worst cycles，做全原始点 targeted audit。
3. 只有 targeted full-point audit 确认后，才训练局部 GEO/radial specialist 或修改 per-cell adapter。
4. 生成 clean release：删除 pycache，重建 SHA256 manifest，冻结 D18 scorecard。
5. 可选：将冻结 Step2 runtime 接入 raw replay，减少对外部 baseline channels 的依赖。

## 科学边界

允许表述：

> FORMAL55-DEPLOY 是一个 55-cell closed-set、Step2/P2Dlite-RG-assisted、per-cell calibrated engineering surrogate。在覆盖全部 26,606 cycles、每 cycle 最多64个事件感知点的审计中，cell-balanced primary mean R²≈0.916，整体结论为 ACCEPT_WITH_LIMITATIONS。

禁止表述：

```text
55 cells 所有原始1Hz采样点都达到 R²=0.916
跨 cell 泛化已证明
XJTU 内部状态是实验真值
phie/phis_c learned prediction 已通过
```
