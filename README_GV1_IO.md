# GV1 通用数据读取层

本压缩包只新增 `gv1/io/*` 与一个检查脚本 `scripts/gv1_inspect_data_file.py`，不修改旧的 ASSB 主线文件。

## 目标

将 `.mat`、`.csv`、`.parquet` 三类电池时序文件统一读取为标准表：

```text
dataset_id, batch_id, battery_id, cycle_id, step_id, step_type,
time_s, current_A, voltage_V, temperature_C, capacity_Ah,
energy_Wh, source_file, source_format
```

默认输出电流符号约定为：

```text
current_A > 0 表示充电
current_A < 0 表示放电
current_A = 0 表示静置
```

## 推荐放置位置

将压缩包解压到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

它只会新增：

```text
gv1/io/*.py
scripts/gv1_inspect_data_file.py
README_GV1_IO.md
```

## 检查 CSV 示例

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_inspect_data_file.py `
  --input "E:\XJTU battery dataset\Batch-3\R2.5_battery-1_cycle_0001.csv" `
  --dataset_id XJTU `
  --batch_id Batch-3 `
  --battery_id battery-1 `
  --cycle_id 1 `
  --default_temperature_C 25 `
  --preview_csv ".\CacheGV1\preview.csv"
```

## 检查 MAT 示例

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_inspect_data_file.py `
  --input "E:\XJTU battery dataset\Batch-1\2C_battery-1.mat" `
  --dataset_id XJTU `
  --batch_id Batch-1 `
  --battery_id battery-1 `
  --default_temperature_C 25 `
  --output ".\CacheGV1\Batch-1_2C_battery-1.standard.parquet"
```

`.mat` 读取策略：

1. 优先 `scipy.io.loadmat`；
2. 若遇到 MATLAB v7.3/HDF5 格式，自动尝试 `h5py`；
3. 自动扫描嵌套结构中的表格候选，优先选择含 `time/current/voltage/capacity/temperature` 字段最多的表。

如 `.mat` 嵌套路径很特殊，可用：

```powershell
--mat_table_path root.some_nested_table
```

## 依赖

基础依赖：

```text
numpy
pandas
scipy
```

可选依赖：

```text
h5py      # MATLAB v7.3 .mat
pyarrow   # parquet 缓存读写
```

## 注意

本读取层只做数据读取、字段映射、单位归一化和缓存，不生成 soft label，不训练模型。后续的 dataset index、XJTU adapter、soft-label generator、trainer/evaluator 会在下一批文件中接入。
