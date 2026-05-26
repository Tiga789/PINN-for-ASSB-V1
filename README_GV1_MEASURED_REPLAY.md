# GV1 measured-current replay 数据构建层

本压缩包只新增 measured-current replay 数据构建工具，不修改旧 ASSB 主线文件。

## 作用

把已经标准化的数据表转换成 GV1 后续软标签、训练和评估入口统一使用的 `ReplayProfile`：

```text
t_global_s
I_profile
voltage_exp
cycle_id
step_id / step_type
temperature_C
Q_charge_Ah / Q_discharge_Ah / throughput_Ah
E_charge_Wh / E_discharge_Wh
```

## 核心假设

对于恒流、分段恒流、恒功率、恒压、CC-CV 或动态工况，只要文件中已有实测 `I(t)`，本层都按
`measured_current_profile` 处理。它不求解控制器，只回放实测电流。

## 主要文件

```text
gv1/measured_replay/profile_builder.py
gv1/measured_replay/current_interpolator.py
gv1/measured_replay/capacity_integrator.py
gv1/measured_replay/step_classifier.py
gv1/measured_replay/replay_audit.py
```

## 用法示例

```python
from gv1.io.auto_reader import read_battery_file
from gv1.io.reader_base import ReadOptions
from gv1.measured_replay.profile_builder import build_replay_profile, save_replay_profile_npz

result = read_battery_file('some_cell.mat', ReadOptions(default_temperature_C=25.0))
profile = build_replay_profile(result.dataframe)
save_replay_profile_npz(profile, 'DataGV1/example/solution_replay_profile.npz')
```
