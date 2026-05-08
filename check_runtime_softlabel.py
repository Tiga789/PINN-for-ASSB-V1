import os, json, numpy as np
from pathlib import Path

d = Path(os.environ["ASSB_SOFT_LABEL_DIR"])
print("ASSB_SOFT_LABEL_DIR =", d)
print("exists =", d.exists())

sol = d / "solution.npz"
summary = d / "soft_label_summary.json"

print("solution exists =", sol.exists())
print("summary exists =", summary.exists())

z = np.load(sol, allow_pickle=True)
print("array count =", len(z.files))
print("first keys =", z.files[:30])

for k in [
    "t_global_s", "t", "time_s",
    "I_profile", "cycle_id",
    "cs_a", "cs_c", "phis_c", "phie",
    "j_a", "j_c"
]:
    if k in z.files:
        arr = z[k]
        print(k, arr.shape, arr.dtype)

if summary.exists():
    js = json.loads(summary.read_text(encoding="utf-8"))
    for k in [
        "time_scale_s",
        "t_end_s",
        "cycle_min",
        "cycle_max",
        "theta_c_bottom",
        "theta_c_top",
        "R_ohm_eff",
        "voltage_alignment_offset",
        "csanmax",
        "cscmax",
    ]:
        if k in js:
            print(k, "=", js[k])
