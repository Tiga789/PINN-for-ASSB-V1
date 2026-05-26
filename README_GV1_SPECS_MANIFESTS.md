# GV1 规格文件与实验清单 v1

本压缩包只新增规格文件、实验配置和训练/评估 manifest，不修改旧 ASSB 主线文件。

## 包含文件

```text
cell_specs/templates/effective_spm_li_ion_default.yaml
cell_specs/xjtu_ncm523_graphite_assumed.yaml
experiment_specs/xjtu_batch1_measured_replay.yaml
experiment_specs/xjtu_batch3_measured_replay.yaml
experiment_specs/xjtu_batch4_measured_replay.yaml
manifests/gv1_xjtu_batch134_index.yaml
manifests/gv1_xjtu_batch134_smoke.yaml
manifests/gv1_xjtu_batch134_train.yaml
manifests/gv1_xjtu_batch134_eval.yaml
README_GV1_SPECS_MANIFESTS.md
```

## 使用边界

- XJTU 温度默认 25 ℃。
- 正极材料按数据集说明设为 NCM523：LiNi0.5Co0.2Mn0.3O2。
- 负极暂设为 `graphite_assumed`，这是工程假设，不是说明文件明确给出的字段。
- 几何信息缺失时，第一版采用 `capacity_normalized_effective_spm`。
- 所有 Batch 均按 `measured_current_profile` 处理；不求解恒压/恒功率控制器电流。

## 建议执行顺序

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\scripts\gv1_build_dataset_index.py `
  --dataset_root "E:\XJTU battery dataset" `
  --dataset_id XJTU `
  --include_batches Batch-1 Batch-3 Batch-4 `
  --patterns "*.mat" `
  --output_dir ".\manifests\xjtu_batch134_index"
```

然后先运行 smoke manifest，再进入正式 train/eval manifest。
