#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply ModelFin_106 linear-cycle common-mode potential gauge to a full cycle5-522 raw eval directory."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


def jfloat(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = ["variable", "cycle_id", "n", "mae", "rmse", "max_abs_error", "bias_mean", "corr", "r2", "nmae", "std_ratio_pred_over_label"]
    keys: List[str] = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def find_npz(eval_dir: Path) -> Path:
    for name in ["eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz", "eval_sampled_arrays_cycles5_100_softlabel_only.npz"]:
        p = eval_dir / name
        if p.exists():
            return p
    hits = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not hits:
        raise FileNotFoundError(f"No eval_sampled_arrays*.npz found in {eval_dir}")
    return hits[0]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def corr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size < 2 or np.std(y) <= 0 or np.std(p) <= 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "max_abs_error": None, "bias_mean": None, "corr": None, "r2": None, "nmae": None, "std_ratio_pred_over_label": None}
    e = p - y
    ss = float(np.sum((y - np.mean(y)) ** 2))
    rng = float(np.max(y) - np.min(y))
    ys = float(np.std(y)); ps = float(np.std(p))
    return {
        "n": int(y.size),
        "mae": jfloat(np.mean(np.abs(e))),
        "rmse": jfloat(np.sqrt(np.mean(e * e))),
        "max_abs_error": jfloat(np.max(np.abs(e))),
        "bias_mean": jfloat(np.mean(e)),
        "corr": jfloat(corr(y, p)),
        "r2": jfloat(float("nan") if ss <= 0 else 1.0 - np.sum(e * e) / ss),
        "nmae": jfloat(float("nan") if rng <= 0 else np.mean(np.abs(e)) / rng),
        "std_ratio_pred_over_label": jfloat(float("nan") if ys <= 0 else ps / ys),
    }


def repeat_cycle_ids(cycle_time: np.ndarray, flat_len: int, nr: int) -> np.ndarray:
    c = np.asarray(cycle_time).reshape(-1)
    if c.size == flat_len:
        return c
    if c.size * nr == flat_len:
        return np.repeat(c, nr)
    raise ValueError(f"Cannot align cycle IDs: len={c.size}, flat_len={flat_len}, nr={nr}")


def variables(arr: Mapping[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cid_p = np.asarray(arr["cycle_id_potential"]).reshape(-1)
    out["phis_c"] = (arr["phis_c_true"].reshape(-1), arr["phis_c_pred"].reshape(-1), cid_p)
    out["phie"] = (arr["phie_true"].reshape(-1), arr["phie_pred"].reshape(-1), cid_p)
    cid_cs = np.asarray(arr["cycle_id_cs"]).reshape(-1)
    nr_a = int(np.asarray(arr["r_a"]).reshape(-1).size)
    nr_c = int(np.asarray(arr["r_c"]).reshape(-1).size)
    for var, nr in [("theta_a", nr_a), ("theta_c", nr_c), ("cs_a", nr_a), ("cs_c", nr_c)]:
        yt = arr[f"{var}_true"].reshape(-1)
        yp = arr[f"{var}_pred"].reshape(-1)
        out[var] = (yt, yp, repeat_cycle_ids(cid_cs, yt.size, nr))
    return out


def compute_metrics(arr: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    glob: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for var, (yt, yp, cid) in variables(arr).items():
        allmask = (cid >= cycle_from) & (cid <= cycle_to)
        glob[var] = metrics(yt[allmask], yp[allmask])
        for c in range(cycle_from, cycle_to + 1):
            m = allmask & (cid == c)
            if not np.any(m):
                continue
            row = {"variable": var, "cycle_id": int(c)}
            row.update(metrics(yt[m], yp[m]))
            rows.append(row)
    return glob, rows


def common_diag(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Dict[str, Any]:
    cid = before["cycle_id_potential"].reshape(-1)
    m = (cid >= cycle_from) & (cid <= cycle_to)

    def errs(arr: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phie = arr["phie_pred"].reshape(-1).astype(float) - arr["phie_true"].reshape(-1).astype(float)
        phis = arr["phis_c_pred"].reshape(-1).astype(float) - arr["phis_c_true"].reshape(-1).astype(float)
        cm = 0.5 * (phie + phis)
        diff = phis - phie
        return phie, phis, cm, diff

    def stat(x: np.ndarray) -> Dict[str, Any]:
        x = x[m]
        return {"n": int(x.size), "mae": jfloat(np.mean(np.abs(x))), "rmse": jfloat(np.sqrt(np.mean(x*x))), "bias_mean": jfloat(np.mean(x)), "std": jfloat(np.std(x))}

    labels = ["phie_error", "phis_c_error", "common_mode_error", "differential_phis_minus_phie_error"]
    b = errs(before); a = errs(after)
    return {"before": {k: stat(v) for k, v in zip(labels, b)}, "after": {k: stat(v) for k, v in zip(labels, a)}}


def common_by_cycle(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], offset: np.ndarray, cycle_from: int, cycle_to: int) -> List[Dict[str, Any]]:
    cid = before["cycle_id_potential"].reshape(-1)
    phie_b = before["phie_pred"].reshape(-1).astype(float) - before["phie_true"].reshape(-1).astype(float)
    phis_b = before["phis_c_pred"].reshape(-1).astype(float) - before["phis_c_true"].reshape(-1).astype(float)
    phie_a = after["phie_pred"].reshape(-1).astype(float) - after["phie_true"].reshape(-1).astype(float)
    phis_a = after["phis_c_pred"].reshape(-1).astype(float) - after["phis_c_true"].reshape(-1).astype(float)
    cm_b = 0.5 * (phie_b + phis_b); cm_a = 0.5 * (phie_a + phis_a)
    diff_b = phis_b - phie_b; diff_a = phis_a - phie_a
    rows = []
    for c in range(cycle_from, cycle_to + 1):
        m = cid == c
        if not np.any(m):
            continue
        rows.append({
            "cycle_id": int(c), "n": int(np.sum(m)),
            "common_mode_error_mean_before": jfloat(np.mean(cm_b[m])),
            "common_mode_error_mae_before": jfloat(np.mean(np.abs(cm_b[m]))),
            "common_mode_error_mean_after": jfloat(np.mean(cm_a[m])),
            "common_mode_error_mae_after": jfloat(np.mean(np.abs(cm_a[m]))),
            "differential_error_mae_before": jfloat(np.mean(np.abs(diff_b[m]))),
            "differential_error_mae_after": jfloat(np.mean(np.abs(diff_a[m]))),
            "offset_to_add_mean_V": jfloat(np.mean(offset[m])),
        })
    return rows


def try_plots(outdir: Path, rows_raw: List[Dict[str, Any]], rows_corr: List[Dict[str, Any]], common_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = outdir / "plots_linearCycleGauge"
    pdir.mkdir(parents=True, exist_ok=True)
    for var in ["phis_c", "phie", "theta_c", "cs_c"]:
        rb = [r for r in rows_raw if r["variable"] == var]
        ra = [r for r in rows_corr if r["variable"] == var]
        if not rb or not ra:
            continue
        xb = [r["cycle_id"] for r in rb]; yb = [r["mae"] for r in rb]
        xa = [r["cycle_id"] for r in ra]; ya = [r["mae"] for r in ra]
        plt.figure(figsize=(8, 4.5))
        plt.plot(xb, yb, label="raw")
        plt.plot(xa, ya, label="corrected")
        plt.xlabel("cycle_id"); plt.ylabel(f"{var} MAE")
        plt.title(f"ModelFin_106 {var} per-cycle MAE")
        plt.legend(); plt.tight_layout()
        plt.savefig(pdir / f"per_cycle_mae_{var}_raw_vs_corrected.png", dpi=160)
        plt.close()
    if common_rows:
        x = [r["cycle_id"] for r in common_rows]
        b = [r["common_mode_error_mean_before"] for r in common_rows]
        a = [r["common_mode_error_mean_after"] for r in common_rows]
        off = [r["offset_to_add_mean_V"] for r in common_rows]
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, b, label="common error raw")
        plt.plot(x, a, label="common error corrected")
        plt.plot(x, off, label="offset added")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("cycle_id"); plt.ylabel("V")
        plt.legend(); plt.tight_layout()
        plt.savefig(pdir / "per_cycle_common_mode_error_and_offset.png", dpi=160)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="ModelFin_106")
    ap.add_argument("--eval_dir", default="EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only")
    ap.add_argument("--output_dir", default="EvalFin_106_cycles5_522_v2_massclosed_candidate_linearCycleGauge_softlabel_only")
    ap.add_argument("--cycle_from", type=int, default=5)
    ap.add_argument("--cycle_to", type=int, default=522)
    ap.add_argument("--save_npz", action="store_true")
    args = ap.parse_args()

    gauge = read_json(Path(args.model_dir) / "gauge_config.json")
    slope = float(gauge["linear_bias_slope_V_per_cycle"])
    intercept = float(gauge["linear_bias_intercept_V"])
    arr = load_npz(find_npz(Path(args.eval_dir)))
    cid = arr["cycle_id_potential"].reshape(-1).astype(float)
    offset = -(slope * cid + intercept)
    corr_arr = {k: np.array(v, copy=True) for k, v in arr.items()}
    for key in ["phie_pred", "phis_c_pred"]:
        shape = corr_arr[key].shape
        corr_arr[key] = (corr_arr[key].reshape(-1).astype(float) + offset).reshape(shape).astype(corr_arr[key].dtype, copy=False)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    glob_raw, rows_raw = compute_metrics(arr, args.cycle_from, args.cycle_to)
    glob_corr, rows_corr = compute_metrics(corr_arr, args.cycle_from, args.cycle_to)
    diag = common_diag(arr, corr_arr, args.cycle_from, args.cycle_to)
    common_rows = common_by_cycle(arr, corr_arr, offset, args.cycle_from, args.cycle_to)
    summary = {
        "script_version": "ASSB_ModelFin106_linearCycleGauge_apply_cycle5_522_v1",
        "model_dir": args.model_dir,
        "raw_eval_dir": args.eval_dir,
        "gauge_config": gauge,
        "metrics_global_raw": glob_raw,
        "metrics_global_corrected": glob_corr,
        "potential_common_mode_diagnostic": diag,
    }
    write_json(out / "model106_linear_cycle_gauge_summary.json", summary)
    write_json(out / "metrics_global_raw.json", glob_raw)
    write_json(out / "metrics_global_corrected.json", glob_corr)
    write_json(out / "potential_common_mode_diagnostic_before_after.json", diag)
    write_csv(out / "metrics_by_cycle_raw.csv", rows_raw)
    write_csv(out / "metrics_by_cycle_corrected.csv", rows_corr)
    write_csv(out / "potential_common_mode_by_cycle_before_after.csv", common_rows)
    if args.save_npz:
        corr_arr["potential_common_mode_offset_to_add"] = offset.astype(np.float32)
        corr_arr["phie_pred_before_linearCycleGauge"] = arr["phie_pred"]
        corr_arr["phis_c_pred_before_linearCycleGauge"] = arr["phis_c_pred"]
        np.savez_compressed(out / "eval_sampled_arrays_ModelFin106_linearCycleGauge_corrected.npz", **corr_arr)
    try_plots(out, rows_raw, rows_corr, common_rows)
    print(json.dumps({
        "output_dir": str(out),
        "phis_c_mae_raw": glob_raw["phis_c"]["mae"],
        "phis_c_mae_corrected": glob_corr["phis_c"]["mae"],
        "phie_mae_raw": glob_raw["phie"]["mae"],
        "phie_mae_corrected": glob_corr["phie"]["mae"],
        "theta_c_mae_corrected": glob_corr["theta_c"]["mae"],
        "cs_c_mae_corrected": glob_corr["cs_c"]["mae"],
        "common_mode_mae_raw": diag["before"]["common_mode_error"]["mae"],
        "common_mode_mae_corrected": diag["after"]["common_mode_error"]["mae"],
        "differential_mae_corrected": diag["after"]["differential_phis_minus_phie_error"]["mae"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
