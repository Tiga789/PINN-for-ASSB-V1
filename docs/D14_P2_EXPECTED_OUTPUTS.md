
# D14-P2 Expected Output Interpretation

## PASS

满足以下条件：

- P0/P1 没有 FAIL；WARN 可接受。
- 能读取 D10/D12 既有 scorecard 或从 P0 index 读取到可用 aggregate fallback。
- `Batch-1 / 2C / battery-8` 没有出现在 mainline scorecard 行中。
- 生成 global/protocol/cell/candidate/outlier 表。

## WARN

常见可接受 WARN：

- P0/P1 继承 WARN，例如 hard clamp 入口存在但默认关闭。
- 缺少 D13 segment diagnosis 目录，但 D12-S1K segment metrics 已足够。
- 只能从 P0 index 读取 aggregate fallback，而不能读取详细 CSV。

## FAIL

需要停止并修复：

- P0 或 P1 是 FAIL。
- mainline scorecard 中包含 `Batch-1 / 2C / battery-8`。
- strict evidence 模式下找不到 D10-P1 或 D12-S1K scorecard。
- run metrics 完全为空。
