import argparse
import json
import math
import numpy as np
from pathlib import Path

def find_key(files, candidates):
    s = set(files)
    for k in candidates:
        if k in s:
            return k
    return None

def orient_time_radial(arr, n_time):
    a = np.asarray(arr)
    if a.ndim == 1:
        if a.shape[0] != n_time:
            raise ValueError(f"1D length {a.shape[0]} != n_time {n_time}")
        return a.reshape(n_time, 1)
    if a.ndim == 2:
        if a.shape[0] == n_time:
            return a
        if a.shape[1] == n_time:
            return a.T
    raise ValueError(f"cannot orient shape={a.shape} for n_time={n_time}")

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if y.size == 0:
        return {"n": 0}
    err = p - y
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rng = float(np.max(y) - np.min(y))
    return {
        "n": int(y.size),
        "r2": float(1.0 - sse / max(sst, 1e-30)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "nmae": float(np.mean(np.abs(err)) / max(rng, 1e-30)),
        "nrmse": float(np.sqrt(np.mean(err ** 2)) / max(rng, 1e-30)),
        "bias": float(np.mean(err)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
    }

def cycle_ranges(cycles):
    vals = sorted(set(int(x) for x in cycles if np.isfinite(x)))
    if not vals:
        return ""
    ranges = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = v
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)

ap = argparse.ArgumentParser()
ap.add_argument("--pred_npz", required=True)
ap.add_argument("--targets", nargs="+", default=["cs_a", "cs_c", "phie", "phis_c"])
args = ap.parse_args()

z = np.load(args.pred_npz, allow_pickle=True)
files = list(z.files)

t_key = find_key(files, ["t_global_s", "time_s", "t_s", "t"])
cycle_key = find_key(files, ["cycle_id", "cycles", "cycle"])
t = np.asarray(z[t_key]).reshape(-1) if t_key else np.arange(0)
cycles = np.asarray(z[cycle_key]).reshape(-1) if cycle_key else np.array([])

out = {
    "pred_npz": args.pred_npz,
    "keys": files,
    "time_key": t_key,
    "cycle_key": cycle_key,
    "n_time": int(t.size),
    "time_range": [float(np.min(t)), float(np.max(t))] if t.size else None,
    "available_cycles": cycle_ranges(cycles) if cycles.size else "",
    "target_metrics": {},
    "missing": {},
}

n_time = int(t.size)
for target in args.targets:
    pred_key = find_key(files, [
        f"{target}_pred",
        f"pred_{target}",
        f"{target}_p",
        target,
    ])
    true_key = find_key(files, [
        f"{target}_true_report_only",
        f"{target}_true",
        f"true_{target}",
        f"{target}_target",
        f"{target}_soft",
    ])
    if pred_key is None or true_key is None:
        out["missing"][target] = {
            "pred_key": pred_key,
            "true_key": true_key,
        }
        continue
    yp = orient_time_radial(z[pred_key], n_time)
    yt = orient_time_radial(z[true_key], n_time)
    out["target_metrics"][target] = {
        "pred_key": pred_key,
        "true_key": true_key,
        "shape_pred": list(yp.shape),
        "shape_true": list(yt.shape),
        **metrics(yt, yp),
    }

z.close()
print(json.dumps(out, indent=2, ensure_ascii=False))
