# GV1 时间轴修复补丁 v3

只覆盖新增文件：`gv1/io/field_mapper.py`。

修复内容：

1. 不再把 `system_time` 自动映射为 `time_s`，避免 XJTU `.mat` 拼接后 `system_time` 跨 subrecord 回跳。
2. 优先保留 `raw__system_time` 用于审计，同时用 `relative_time_min` + `raw__mat_subrecord_index` 重建全局单调 `time_s`。
3. 修复成功后日志应出现：

```text
time_s rebuilt as monotonic global time from raw__mat_subrecord_index + local elapsed time (GV1 time fix v3)
valid_standard_table = true
validation_problems = []
```

覆盖后先运行：

```powershell
Select-String -Path .\gv1\io\field_mapper.py -Pattern "GV1 time fix v3|system_time" -Context 0,2
```
