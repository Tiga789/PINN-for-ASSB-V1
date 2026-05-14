# ModelFin_104 input parser fix

问题：旧版 `input_assb_cycles5to522_v2_massclosed_ID104_radial010` 被错误打成一整行，prettyPlot/parser.py 使用 `line.split(":")` 时，一行内出现多个冒号，导致：

```text
ValueError: too many values to unpack (expected 2)
```

修复：覆盖项目根目录下的 input 文件，使每一行只保留一个 `key : value`。

放置位置：

```text
PINN-for-ASSB-V1\input_assb_cycles5to522_v2_massclosed_ID104_radial010
PINN-for-ASSB-V1\scripts\check_input104_parser_format.ps1
```

覆盖后运行：

```powershell
.\scripts\check_input104_parser_format.ps1
.\scriptsun_train_ModelFin104_v2_massclosed_cycle5_20.ps1
```
