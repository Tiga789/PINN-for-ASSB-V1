# QJW-2 / PINN-for-ASSB-V1 clean clone reproducibility check

这个简本包只做 **clean clone 静态复现检查**，不会修改你的主项目，不会启动训练，也不会改 GitHub 仓库。

## 包内文件

```text
run_clean_clone_repro_check.ps1   # PowerShell 主脚本
check_repo_after_clone.py         # Python 静态检查脚本
README_CLEAN_CLONE_REPRO_CHECK.md # 本说明
```

## 默认检查内容

1. 从 GitHub 重新 clone：

```text
https://github.com/Tiga789/PINN-for-ASSB-V1.git
```

2. 检查关键文件是否存在：

```text
README.md
gv1/model.py
gv1/output_transform.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
```

3. 对 `gv1/` 和 `scripts/` 执行 `python -m compileall -q`。

4. 检查 README 是否包含当前主线边界关键词：

```text
ASSB-D10
ModelFin_112_deterministic_wrapper
D9.6
D9.5.1
battery-8
metadata_on
D12-S2
D12-S3
```

5. 搜索常见风险信号，例如：

```text
enable_voltage_hard_clamp=True
epochs=40000
time_window_s=200000
```

这些风险信号默认作为 warning，而不是自动判定失败，因为还需要人工看上下文。

## 推荐运行方式

在 PowerShell 中进入本目录，然后运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_clean_clone_repro_check.ps1 -ForceRemoveExisting
```

如你的 Python 路径不同，使用：

```powershell
.\run_clean_clone_repro_check.ps1 `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -ForceRemoveExisting
```

## 输出位置

默认会生成：

```text
C:\Users\<你>\Desktop\PINN-for-ASSB-V1_cleancheck
C:\Users\<你>\Desktop\_qjw_clean_clone_reports\clean_clone_YYYYMMDD_HHMMSS
```

报告文件：

```text
clean_clone_check_report.json
clean_clone_check_report.md
01_git_clone.log
02_git_head.log
03_git_status.log
04_python_version.log
05_static_checker.log
```

## 通过标准

优先看 JSON 或 MD 报告中的：

```text
overall_status = PASS
hard_failures = 0
```

warning 不一定代表失败，但需要人工检查。如果出现以下情况，应先停下来修复：

```text
required path missing
compileall returncode != 0
README 主线关键词缺失
```

## 注意

- 这不是训练脚本。
- 这不是 D12-S3 包。
- 这一步只确认 GitHub clean clone 是否具备继续实验的最低复现条件。
- D12-S3 clean 23-profile 40ks ablation 应在 clean clone 检查通过后再做。
