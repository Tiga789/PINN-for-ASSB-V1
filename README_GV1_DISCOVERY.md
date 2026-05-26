# GV1 通用数据集发现与遍历层 v1

本压缩包只包含“2. 通用数据集发现与遍历”相关的新增文件，不修改旧主线文件。
它依赖前一个“通用数据读取层”的目录布局，但不要求读取 `.mat` 内容；本层只负责发现文件、解析 path-level 元数据、建立 dataset index。

## 包含文件

```text
gv1/data_discovery.py
gv1/dataset_index.py
gv1/cell_id_parser.py
gv1/protocol_parser.py
gv1/standard_table.py
scripts/gv1_build_dataset_index.py
tests/test_gv1_dataset_discovery.py
README_GV1_DISCOVERY.md
```

## 设计目标

- 不写死 XJTU、Batch-1 或某一只电池；这些都通过命令行或 manifest 控制。
- 支持递归发现 `.mat / .csv / .parquet` 文件。
- 从路径中自动解析常见的 `batch_id / battery_id / cycle_id_hint / protocol_hint`。
- 输出统一 `dataset_index.csv` 和 `dataset_index.jsonl`，后续标准表构建、soft-label 生成、训练、评估都从这个 index 开始。

## XJTU Batch-1/3/4 示例

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_build_dataset_index.py `
  --dataset_root "E:\XJTU battery dataset" `
  --dataset_id XJTU `
  --include_batches Batch-1 Batch-3 Batch-4 `
  --patterns "*.mat" `
  --output_dir ".\manifests\xjtu_batch134_index"
```

预期输出：

```text
manifests/xjtu_batch134_index/dataset_index.csv
manifests/xjtu_batch134_index/dataset_index.jsonl
manifests/xjtu_batch134_index/dataset_index_summary.json
```

## 可选协议映射

如果希望给 Batch 添加更明确的工况标签，可以提供 JSON：

```json
{
  "Batch-1": {
    "protocol_id": "xjtu_batch1_fixed_cccv_full_discharge",
    "observed_control_mode": "cccv_record",
    "notes": "fixed CC-CV charge, full discharge"
  },
  "Batch-3": {
    "protocol_id": "xjtu_batch3_variable_discharge_full",
    "observed_control_mode": "cccv_record",
    "notes": "variable discharge C-rate, full discharge"
  },
  "Batch-4": {
    "protocol_id": "xjtu_batch4_variable_discharge_partial_with_capacity_tests",
    "observed_control_mode": "cccv_record",
    "is_partial_discharge_hint": true,
    "notes": "partial discharge with periodic capacity tests"
  }
}
```

然后运行：

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_build_dataset_index.py `
  --dataset_root "E:\XJTU battery dataset" `
  --dataset_id XJTU `
  --include_batches Batch-1 Batch-3 Batch-4 `
  --patterns "*.mat" `
  --protocol_mapping ".\manifests\xjtu_batch134_protocol_mapping.json" `
  --output_dir ".\manifests\xjtu_batch134_index"
```

## 说明

本层不读取 `.mat` 文件内部字段，不判断电流符号，也不生成 soft labels。它只做数据集发现与遍历。下一层会基于 `dataset_index.csv` 调用 `gv1/io/auto_reader.py` 读取每个文件并生成标准表。
