# FORMAL55 selected-cycle inference, soft-label audit and interactive 3D plotting

## Purpose

This tool performs on-demand inference for a small cycle range of one of the 55 closed-set XJTU cells. It does **not** generate a 55-cell prediction archive.

Execution order is fixed:

1. Resolve the exact canonical cell UID from `batch + battery`.
2. Load the full observable/frozen-baseline profile so that all history before the requested cycles is available.
3. Predict every original time point of the requested cycles and keep predictions in RAM.
4. Only after prediction is complete, stream the selected `cs_a/cs_c` soft-label rows for evaluation.
5. Compute full-point aggregate and per-cycle metrics.
6. Draw four interactive Matplotlib 3D windows: anode/cathode prediction and reference.

The tool is a Step2/P2Dlite-RG-assisted closed-set engineering utility. Soft labels are model-consistent references, not experimental ground truth.

## Install location

Extract the package into the project root so the directory becomes:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\formal55_selected_cycle_tool
```

No existing project files are overwritten.

## Edit the JSON request

Edit:

```text
formal55_selected_cycle_tool\configs\selected_cycle_request.json
```

Example selection:

```json
"selection": {
  "batch": 2,
  "battery": 5,
  "cycles": "35-37"
}
```

Batch/protocol mapping is fixed:

```text
Batch 1 = 2C
Batch 2 = 3C
Batch 3 = R2.5
Batch 4 = R3
Batch 5 = random_walk
Batch 6 = GEO
```

`inference_points` is intentionally not a sampling option: requested cycles always use all original time points. `plot.max_time_points` affects visualization only, never metric calculation.

## Run

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\formal55_selected_cycle_tool\scripts\run_formal55_selected_cycle.ps1 `
  -RequestJson ".\formal55_selected_cycle_tool\configs\selected_cycle_request.json"
```

To save figures and metrics without opening interactive windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\formal55_selected_cycle_tool\scripts\run_formal55_selected_cycle.ps1 `
  -RequestJson ".\formal55_selected_cycle_tool\configs\selected_cycle_request.json" `
  -NoShow
```

Default output directory:

```text
<project root>\formal55_selected_cycle_outputs\<request_name>_<timestamp>
```

## Outputs

```text
RUN_STATUS.json
request_resolved.json
metrics_global.json
metrics_global_and_by_cycle.csv
physical_and_leakage_audit.json
selected_cycle_index_ledger.csv
suspicious_exact_metrics.csv
plots\*.png                     (when save_png=true)
selected-cycle NPZ              (only when save_selected_npz=true)
```

Metrics are computed from every original selected time point and every radial point. The 3D surface is downsampled only for display.

Reported metrics include:

```text
cs R², MAE, RMSE, NMAE, NRMSE, bias, correlation
theta R² and normalized errors
delta_cs R²
surface-minus-mean R²
surface-center-gradient R²
radial-energy R²
```

The prediction/reference surfaces for each electrode use identical axes, z-limits and color normalization, while retaining the requested distinct colormaps.

## History handling

The tool loads the complete observable/frozen-baseline profile. Cumulative charge, cycle age, lagged features and the Step2 baseline are therefore constructed using the profile history before the first requested cycle. Only the requested cycles are emitted, evaluated and plotted.

## Storage behavior

By default no prediction NPZ is saved. Metrics and PNG files are small. Set:

```json
"output": {
  "save_selected_npz": true,
  "save_truth_in_npz": false
}
```

only when a small reusable selected-cycle prediction file is needed.
