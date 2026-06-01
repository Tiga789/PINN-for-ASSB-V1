# D12-S3 clean 23-profile 40ks metadata ablation package v2

本包用于重新执行或复查 **D12-S3 clean 23-profile 40ks metadata ablation**。

目标不是替代 D9.6 / D9.5.1 主线，而是审计 metadata 输入是否在 23-profile、40ks 条件下有效。当前项目结论仍应遵守：`metadata_on` 不能直接替代 D9.6/D9.5.1，`B1_2C battery-8` 继续作为 late-2C discharge boundary / outlier 排除。

## 1. 本包内容

解压到项目根目录后，应得到：

```text
scripts/gv1_d12_s3_prepare_23profile_strict_commands.py
scripts/gv1_d12_s3_scorecard_from_predictions.py
scripts/run_gv1_d12_s3_prepare_commands.ps1
scripts/run_gv1_d12_s3_collect_scorecard.ps1
scripts/run_gv1_d12_s3_preflight_check.ps1
scripts/run_gv1_d12_s3_full_workflow_optional.ps1
README_D12_S3_CLEAN_23PROFILE.md
RUN_ORDER_D12_S3.txt
```

其中：

- `gv1_d12_s3_prepare_23profile_strict_commands.py`：生成 off / zero / on 三组 23-profile 严格 40ks 运行脚本。
- `gv1_d12_s3_scorecard_from_predictions.py`：从各 run 的 `prediction.npz` 直接计算 scorecard。
- `run_gv1_d12_s3_prepare_commands.ps1`：一键生成 3 组 generated run scripts。
- `run_gv1_d12_s3_collect_scorecard.ps1`：一键收集 scorecard。
- `run_gv1_d12_s3_preflight_check.ps1`：运行前/生成后预检，防止混入旧 40000 epoch / 200ks 参数。
- `run_gv1_d12_s3_full_workflow_optional.ps1`：可选总控脚本，默认只 prepare，不直接启动 69 runs。

## 2. 默认实验设置

```text
profiles = 23, excluding / flagging B1_2C battery-8
modes = metadata_off / metadata_zero / metadata_on
time_window_s = 40000
epochs = 100
max_time_points = 1024
batch_size = 512
prediction_time_points = 1024
prediction_radial_points = 32
seed = 42
```

预期总运行数：

```text
23 profiles × 3 modes = 69 runs
```

## 3. 前置要求

项目根目录默认：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

Python 默认：

```text
D:\Anaconda\envs\torchgpu\python.exe
```

XJTU 缓存默认：

```text
E:\XJTU battery dataset\_gv1_cache
```

D12 metadata runtime wrapper 应已经存在：

```text
scripts/gv1_train_conditioned_pinn_d12_metadata_runtime.py
```

如果这个文件不存在，说明 D12 runtime patch 没有落实，不能运行 D12-S3。

## 4. 推荐运行顺序

进入项目根目录：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4.1 源码预检

```powershell
.\scripts\run_gv1_d12_s3_preflight_check.ps1
```

通过后应看到：

```text
D12-S3 preflight PASS.
```

### 4.2 生成 D12-S3 三组运行脚本

```powershell
.\scripts\run_gv1_d12_s3_prepare_commands.ps1
```

生成目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_commands
```

### 4.3 生成脚本预检

```powershell
.\scripts\run_gv1_d12_s3_preflight_check.ps1 -AfterPrepare
```

这一步会拒绝下面这些旧危险参数：

```text
epochs 40000
time_window_s 200000
max_time_points 8192
batch_size 2048
_200ks
```

### 4.4 分组运行 69 个任务

建议第一次分开跑，便于定位：

```powershell
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_commands\run_d12_s3_metadata_off_23profile.generated.ps1"

& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_commands\run_d12_s3_metadata_zero_23profile.generated.ps1"

& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_commands\run_d12_s3_metadata_on_23profile.generated.ps1"
```

也可以一次性运行三组：

```powershell
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_commands\run_d12_s3_all_modes_23profile.generated.ps1"
```

### 4.5 收集 scorecard

```powershell
.\scripts\run_gv1_d12_s3_collect_scorecard.ps1
```

输出目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_scorecard
```

重点文件：

```text
d12_s3_scorecard_summary.json
d12_s3_scorecard.csv
d12_s3_mode_summary.csv
d12_s3_protocol_summary.csv
d12_s3_segment_metrics.csv
D12_S3_SCORECARD_RECOMMENDATION.md
```

## 5. 可选总控脚本

只 prepare + preflight，不启动训练：

```powershell
.\scripts\run_gv1_d12_s3_full_workflow_optional.ps1
```

prepare + preflight + 直接运行 69 个任务 + 收集 scorecard：

```powershell
.\scripts\run_gv1_d12_s3_full_workflow_optional.ps1 -RunAblation
```

只收集已有结果：

```powershell
.\scripts\run_gv1_d12_s3_full_workflow_optional.ps1 -CollectOnly
```

## 6. 通过标准

`d12_s3_scorecard_summary.json` 中应满足：

```text
run_count = 69
expected_run_count = 69
counts.strict_completed_metrics_ok = 69
verdict = d12_s3_all_runs_completed_metrics_ok
```

然后比较三组：

```text
metadata_off mean_MAE_V
metadata_zero mean_MAE_V
metadata_on mean_MAE_V
```

如果 `metadata_on` 仍弱于 `metadata_off`，应停止 metadata_on 替代主线的路线，回到 D9.6 / D9.5.1 主线做 segment / protocol 分析。

## 7. 注意事项

- 不要解除 `B1_2C battery-8` 的 flagged/excluded 状态。
- 不要将 metadata_on 直接升格为主线。
- 不要运行旧 D12 runtime 中的 40000 epoch / 200ks 脚本。
- 不要覆盖 D9.6 / D9.5.1 主线代码。
- 本包只新增 D12-S3 辅助脚本，不应修改 `gv1/model.py`、`gv1/output_transform.py`、`gv1/losses.py`、`gv1/trainer.py` 的主线实现。
