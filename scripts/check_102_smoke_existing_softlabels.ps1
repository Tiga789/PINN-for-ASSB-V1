$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$soft = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1"
$solution = Join-Path $soft "solution.npz"
if (!(Test-Path $solution)) { throw "Missing existing soft label solution: $solution" }

$required = @(
  ".\util\spm_assb_train_discharge.py",
  ".\util\_rescale.py",
  ".\util\_losses.py",
  ".\util\init_pinn.py",
  ".\evaluate_assb_pinn_vs_softlabels.py",
  ".\input_assb_cycles5to522_v4_continuous_ID102"
)
foreach ($f in $required) { if (!(Test-Path $f)) { throw "Missing required file: $f" } }

Select-String -Path .\util\spm_assb_train_discharge.py -Pattern "ASSB_CYCLE_FROM|ASSB_CYCLE_TO|cycle filter|t_global_s|time_scale_s|assb_soft_lable_cycle5-522_v1" | Out-Host
Select-String -Path .\evaluate_assb_pinn_vs_softlabels.py -Pattern "cycle_from|cycle_to|_filter_solution_cycles|soft labels only" | Out-Host
Select-String -Path .\input_assb_cycles5to522_v4_continuous_ID102 -Pattern "assb_soft_lable_cycle5-522_v1|ASSB_CYCLE_FROM|ASSB_CYCLE_TO|existing_full_solution_cycle5_20_smoke" | Out-Host

$py = @"
from pathlib import Path
import numpy as np
p = Path(r'C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1\solution.npz')
with np.load(p, allow_pickle=False) as z:
    cyc = z['cycle_id'] if 'cycle_id' in z.files else z['cycle']
    m = (cyc >= 5) & (cyc <= 20)
    tkey = 't_global_s' if 't_global_s' in z.files else ('t' if 't' in z.files else 'time_s')
    t = z[tkey][m]
    print({'solution': str(p), 'selected_cycles': [int(cyc[m].min()), int(cyc[m].max())], 'selected_points': int(m.sum()), 't_selected_start': float(t[0]), 't_selected_end_minus_start': float(t[-1]-t[0])})
"@
D:\Anaconda\envs\torchgpu\python.exe -c $py

Write-Host "Check finished: this patch uses the existing full solution.npz and filters cycle 5-20. No tools directory is required."
