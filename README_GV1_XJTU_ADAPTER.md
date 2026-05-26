# GV1 XJTU 数据集适配层 v1

本压缩包只新增 XJTU 数据集适配层，不修改旧 ASSB 主线文件。

## 包含文件

```text
gv1/adapters/__init__.py
gv1/adapters/xjtu_adapter.py
gv1/adapters/xjtu_protocols.py
gv1/adapters/xjtu_cell_spec_defaults.py
gv1/adapters/xjtu_soh_targets.py
tests/test_xjtu_adapter_metadata.py
README_GV1_XJTU_ADAPTER.md
```

## 设计原则

- XJTU 适配层只负责数据集元数据、Batch 工况标签、默认 CellSpec 锚点和 SOH 标签生成。
- 真正的数据读取仍调用 `gv1/io` 通用读取层。
- 后续脚本不写死某个 battery；目标对象通过 manifest / dataset index 选择。
- 对 Batch-1、Batch-3、Batch-4 的不同策略，只标记 observed protocol；求解仍采用 measured-current replay。

## 最小用法

```python
from gv1.adapters.xjtu_adapter import XJTUAdapter

adapter = XJTUAdapter(r"E:/XJTU battery dataset", default_temperature_C=25.0)
result = adapter.read(r"E:/XJTU battery dataset/Batch-1/2C_battery-1.mat")
df = result.dataframe
print(df.columns)
```

如果 `.mat` 文件结构特殊，仍可向通用读取层传入 `mat_table_path`：

```python
result = adapter.read(path, mat_table_path="root.data")
```

## SOH 标签

Batch-4 包含部分放电循环，因此不要把所有 operating cycles 都当作完整容量标签。`xjtu_soh_targets.py` 会标记：

```text
is_full_discharge
is_partial_discharge
is_capacity_label_candidate
```

SOH 只应从完整放电或容量测试循环计算。
