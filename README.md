# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM / P2Dlite 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的工程基线，再推进 XJTU 55-cell 多 protocol 数据上的 P2Dlite-RG soft-label generator 与 PINN / neural surrogate。

当前最新工作阶段为 **ASSB-D17**。本 README 用于 D18 新窗口和 GitHub 首页接续，重点记录 D17 的已成立成果、未成立边界和下一步工作。

## 当前总状态

### ASSB 基线

ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- 四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- SOH 来自 `ModelFin_112_deterministicSOH_ridge_g4`。
- 这是 engineering wrapper / unified package，不是端到端联合训练的单个神经网络，也不是跨电池规格泛化证明。

### XJTU / D15 soft-label 数据

D15 已完成 XJTU 55/55 cells 的 P2Dlite-RG model-consistent soft-label 数据集：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
```

数据边界：这些是 P2Dlite-RG model-consistent soft labels，不是实验直接测得的真实内部状态。D15 证明 soft-label 数据集生成、径向审计和 closed-set/repair evidence 成立，但不自动证明 held-out cell 或 full-cycle surrogate 泛化。

### D17-G4 sampled-window surrogate

D17-G 主线最终形成：

```text
D17-G4_GENERATOR_SURROGATE_CANDIDATE
```

它是一个 **physics-informed / generator-distilled neural surrogate**，用于快速复现 P2Dlite-RG soft-label generator 的输出。

训练与审计协议：

```text
train-cell soft labels: used for supervised training
validation soft labels: report-only
frozen-test soft labels: report-only; not used for training or checkpoint selection
checkpoint selection: fit-train + train-internal heldout only
```

G4 sampled-window 结果：

```text
status = PASS
final_candidate_ready = true
frozen_test_mean_r2 ≈ 0.99790
frozen_test_min_r2 ≈ 0.97935
speed_status = PASS
samples_per_second ≈ 95,192
```

重要边界：G4 的训练/审计使用每个 profile 的 sampled-grid，典型为 `39 × 512` train points 和 `7 × 512` validation points。它不能写成 55 cells × all cycles × full-time-grid 成功。

## D17 关键结论

### 1. D17-P no-state-label inverse PINN 路线暂停

D17-P 尝试把 generator 的电化学机制写入 PINN，训练端不直接使用 `cs/theta/phie/phis` soft labels。该路线完成 P0/P1/P2/P3/P4 smoke 与 state audit，但最终结论是：

```text
terminal voltage consistency does not guarantee generator-consistent internal states
```

P4/P4mini 表明 `theta_c/cs_c/phie` 与 generator state 严重不对齐，因此 D17-P 不作为当前主线 promotion。

### 2. D17-G generator-distilled sampled-window surrogate 成立

D17-G 从 G0 generator equivalence audit 开始，逐步修复 phie/gauge、profile-id memorization、protocol/branch heldout、P4D/random_walk coverage，最终通过 G3/G4 sampled-window frozen-test audit。

保留主成果：

```text
D17-G4 sampled-window P2Dlite-RG generator surrogate
```

不要误写成：

```text
full-cycle arbitrary-cycle surrogate
strict no-state-label PINN
experimental ground-truth internal-state estimator
```

### 3. P4D generator provenance 已恢复

D17 后期一度怀疑 Batch-6 GEO / P4D soft-label provenance 不完整。最终通过原始 D15-P4D 脚本 scratch 重放验证：

```text
scripts/d15_p4d_full_generate_one_rg_softlabel.py
configs/d15_p4d_full_remaining14_config.json
E:/XJTU battery dataset/_gv1_cache/xjtu_batch56_remaining14_replay_profiles_d15p4c/xjtu_batch56_remaining14_replay_profile_manifest.csv
```

`Batch-6_battery-2` 和 `Batch-6_battery-5` 的 scratch `solution_softlabels.npz` SHA256 与 ALL55 final 完全一致。结论：D15-P4D generator 可 exact replay；此前 G6.3/G6.5 失败是手写简化 formula replay 不等价。

### 4. full-cycle arbitrary-cycle surrogate 当前未成立

G6F selected-cycle on-demand inference 表明：

- G3 saved 512-point prediction 本身高精度；
- G6F 在同一 exact-grid 上 sanity PASS；
- 但 dense selected cycles，例如 Batch-2 battery-3 cycles 1-4 / 36-38，指标明显失败。

G7-S0 full-cycle sampling audit PASS，但 G7-S1 small full-cycle smoke 失败：fit-train 很高，internal-heldout/validation 很差。S1E 进一步显示超过一半低 R2 失败项不能靠简单 constant inventory shift 或 phie gauge shift 解释。

因此：

```text
Do not enter G7-S2.
Do not run another long training from the current S1 design.
D18 must redesign the full-cycle/cycle-aware surrogate architecture.
```

## 当前不可误用的内容

不要使用以下内容作为 promotion 依据：

```text
D17-G6.1 full_cycle_coverage_repair candidate      # g6_ready=false
D17-G6.2 / G6.2L simplified P4D patch              # formula direction incorrect
D17-G6.3 formula forensics                          # diagnostic only
D17-G6.5 exact provenance replay                    # superseded by true D15-P4D script replay
G7-S1 small full-cycle smoke                        # failed; selected_cycle_check_ready=false
```

不要声称：

```text
D17-G can predict arbitrary cycles with high accuracy.
D17-G is a strict no-state-label PINN.
XJTU soft labels are experimental ground-truth internal states.
```

## 推荐 D18 工作路线

D18 应作为 full-cycle arbitrary-cycle surrogate 的新阶段，而不是继续 G7-S2。

建议顺序：

```text
D18-P0: freeze D17 artifacts and write new D18 manifest
D18-S0: design full-cycle/cycle-aware architecture
D18-S1: array-level latent diagnostic on failed dense cycles
D18-S2: small full-cycle smoke with cycle/protocol/branch stratification
D18-S3: introduce sequence/operator or cycle-aware model family
D18-S4: only if S2/S3 pass, run 39-train / 7-validation mini expansion
D18-S5: cycle-wise streaming full audit, metrics-only by default
```

D18 model design should consider:

- full-profile encoder, not selected-cycle-only summary;
- cycle index / normalized cycle / cumulative Ah / EFC / aging-like latent;
- deterministic inventory baseline + learned residual;
- phie gauge / current-aware profile latent;
- branch-specific adapters for RG and P4D;
- selected-cycle dense audit and cycle-wise streaming audit as mandatory gates.

## 常用路径

```text
Project root:
C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1

GitHub:
https://github.com/Tiga789/PINN-for-ASSB-V1

ALL55 soft labels:
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL

D17-G outputs:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g

D17-P outputs:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild

D17 split manifest:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json

D17-G4/G21 sampled-window candidate:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair

G7-S0 sampling audit:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit

G7-S1 failed smoke:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s1_small_fullcycle_smoke

G7-S1E diagnostic:
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s1e_profile_latent_explainability
```

## 新窗口接续提示

在 D18 新窗口中，先阅读：

```text
ASSB-D17_项目进度复盘总结_20260617.docx
README.md
```

然后查看本地项目与 Git 状态。不要直接进入训练。先确认 D17 成果和失败边界，再设计 full-cycle/cycle-aware surrogate。
