# D18-S0 Cycle-aware Full-profile Operator Architecture

## 1. 设计目的

D17-G4 在固定 sampled-grid 上成立，但 dense selected-cycle 与 full-cycle arbitrary-cycle 失败。D18-S0 不再把每个时间点视为互相独立样本，而把状态写成：

```text
full-profile cycle history
        +
causal within-cycle history
        +
branch semantics (RG / P4D)
        +
deterministic inventory/potential baseline
        ↓
cycle-aware operator decoder
        ↓
cs_a, cs_c, theta_a, theta_c, phie, phis_c
```

## 2. 两层时序编码

### 2.1 Cycle-history encoder

每个 cycle 先形成一个 summary token：

```text
normalized cycle index
cycle duration
charge/rest/discharge duration
charge/discharge/absolute Ah
cumulative absolute Ah
EFC proxy
mean/max current
start/end/min/max voltage
mean/max temperature
previous-cycle terminal latent
protocol / branch metadata
```

这些 token 按 cycle 顺序进入 GRU/TCN，得到 `h_cycle[k]`。它承担 aging、inventory phase 和 history state。

### 2.2 Within-cycle encoder

当前 cycle 内按时间输入：

```text
t_in_cycle
I(t), dI/dt
V(t), dV/dt
T(t)
step type
signed/absolute Ah in cycle
normalized cycle position
```

首版 scaffold 使用 causal GRU。后续 S3 可以对比 causal TCN、Transformer 或 DeepONet branch encoder。

## 3. Hard output structure

### 3.1 Concentration

```text
cs_j(t,r) = cbar_j(t) + delta_cs_j(t,r)
```

- `cbar_j(t)` 由 current integration / generator-consistent inventory baseline 提供；
- `delta_cs_j` 用少量 radial basis 表示；
- 每个 radial basis 满足 spherical-volume weighted zero mean；
- residual amplitude 有界；
- `theta_j` 从 `cs_j` 通过已解析的线性/窗口映射推导，不独立预测。

### 3.2 Potential

```text
phie   = phie_baseline   + shared_gauge + differential_residual_e
phis_c = phis_c_baseline + shared_gauge + differential_residual_s
```

这将 common-mode gauge 和 differential electrochemical response 分开，避免仅靠 terminal voltage 掩盖 phie 的绝对基准漂移。

## 4. RG/P4D 分支

共享 history encoder，但 decoder residual adapter 分开：

```text
shared history features
        ├── RG adapter
        └── P4D adapter
```

P4D/GEO labels 的复现必须调用原始 D15-P4D generator 语义；不得用手写简化公式替代。

## 5. Scaffold 输入/输出

实现文件：`gv1/d18_cycleaware/model_scaffold.py`

输入：

```text
cycle_features      [B, C, F_cycle]
local_features      [B, T, F_local]
cycle_index         [B, T]
cbar                [B, T, 2]
potential_baseline  [B, T, 2]
branch_id           [B]
theta_offset        [B, 2]
theta_scale         [B, 2]
```

输出：

```text
cs_a, theta_a  [B, T, R_a]
cs_c, theta_c  [B, T, R_c]
phie, phis_c   [B, T, 1]
cycle/local/fused latent
potential gauge
radial residuals
```

## 6. S0 只验证什么

S0 synthetic forward 检查：

- tensor shape；
- finite output；
- radial residual zero-volume-mean；
- theta-from-cs relation；
- RG/P4D adapter tensor contract。

S0 不证明：

- 模型已经训练；
- full-cycle 精度；
- 55-cell 泛化；
- soft labels 是实验真值。

## 7. S2 训练前必须完成

S1 报告必须回答：

1. constant/affine latent 是否足够；
2. cycle drift / lag 是否显著；
3. residual 是否低秩；
4. cycle boundary 是否断裂；
5. phie 是否主要为 gauge error；
6. RG 与 P4D 是否需要不同 decoder；
7. teacher/array alignment 是否存在异常。
