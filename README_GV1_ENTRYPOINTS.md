# GV1 软标签 / 训练 / 评估入口

本压缩包只新增 GV1 高层入口脚本和轻量 pipeline 工具，不修改旧 ASSB 主线文件。

## 包含内容

```text
gv1/pipeline/__init__.py
gv1/pipeline/manifest.py
gv1/pipeline/npz_utils.py
gv1/pipeline/metrics.py
gv1/pipeline/data_loader.py
scripts/gv1_generate_softlabels.py
scripts/gv1_train.py
scripts/gv1_eval.py
```

## 入口 1：生成 measured-current replay profile / soft-label 输入包

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_generate_softlabels.py `
  --dataset_index_csv .\manifests\xjtu_batch134_index\dataset_index.csv `
  --dataset_root "E:\XJTU battery dataset" `
  --adapter xjtu `
  --output_dir .\DataGV1\xjtu_batch134_replay_smoke `
  --max_files 1 `
  --default_temperature_C 25 `
  --write_standard_csv
```

输出：

```text
profile_manifest.csv                         # 每个 source_file / cell 一个 profile
profiles/<profile_id>/solution_replay_profile.npz
profiles/<profile_id>/profile_summary.json
profiles/<profile_id>/replay_audit.json
solution_replay_profile.npz                  # 第一个 profile 的便捷副本，用于 smoke
standard_table.parquet
standard_table.csv                           # 可选
cycle_integrals.csv
replay_audit.json
profile_summary.json
resolved_manifest.json
```

说明：当前脚本生成的是 measured-current replay 输入包和 capacity-normalized baseline，尚不是最终
fully resolved effective-SPM soft labels。真正 PINN soft-label solver 接入后，仍沿用这个输出结构。

## 入口 2：训练入口 scaffold

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_train.py `
  --prepared_dir .\DataGV1\xjtu_batch134_replay_smoke `
  --output_dir .\ModelGV1_xjtu_smoke
```

当前版本只验证输入并生成训练计划，避免在 PINN 网络/loss 文件未交付前误启动训练。

## 入口 3：评估入口

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_eval.py `
  --solution_npz .\DataGV1\xjtu_batch134_replay_smoke\solution_replay_profile.npz `
  --output_dir .\EvalGV1_xjtu_smoke
```

若后续有预测文件，可加：

```powershell
  --prediction_npz .\ModelGV1_xjtu_smoke\predictions.npz
```
