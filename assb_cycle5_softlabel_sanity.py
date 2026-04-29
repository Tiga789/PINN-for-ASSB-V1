#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity-check ASSB cycle5 v3 soft labels before physics-only PINN training.

Recommended location: project root
Example:
  D:\\Anaconda\\envs\\torchgpu\\python.exe assb_cycle5_softlabel_sanity.py ^
    --soft_label_dir Data\\assb_soft_labels_cycle5_v3 ^
    --output_dir Eval_cycle5_softlabel_sanity

This script does not import the PINN code and does not train anything.
It checks that the soft-label files are internally consistent and that phis_c
roughly matches the experimental voltage stored in solution.npz if available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


REQUIRED_NPZ = ["data_phie.npz", "data_phis_c.npz", "data_cs_a.npz", "data_cs_c.npz"]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"Could not parse {path.name}: {exc}"}


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _find_key(data: Dict[str, np.ndarray], candidates: Iterable[str]) -> Optional[str]:
    lower = {k.lower(): k for k in data.keys()}
    for cand in candidates:
        if cand in data:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "maxabs": float("nan"), "bias_mean": float("nan"), "corr": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "maxabs": float(np.max(np.abs(err))),
        "bias_mean": float(np.mean(err)),
        "corr": corr,
        "y_true_min": float(np.min(yt)),
        "y_true_max": float(np.max(yt)),
        "y_pred_min": float(np.min(yp)),
        "y_pred_max": float(np.max(yp)),
    }


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {"n": int(x.size), "min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x))}


def _dataset_report(name: str, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"keys": list(data.keys())}
    for key in ["x_train", "y_train", "x_params_train"]:
        if key in data:
            arr = np.asarray(data[key])
            rep[f"{key}_shape"] = list(arr.shape)
            if key == "x_train":
                x = _as_2d(arr)
                rep["t_stats"] = _summary_stats(x[:, 0])
                if x.shape[1] > 1:
                    rep["r_stats"] = _summary_stats(x[:, 1])
            elif key == "y_train":
                rep["y_stats"] = _summary_stats(arr)
    if "x_train" in data and "y_train" in data:
        x = _as_2d(data["x_train"])
        y = _as_2d(data["y_train"])
        rep["n_x_equals_n_y"] = bool(x.shape[0] == y.shape[0])
        rep["finite_y_fraction"] = float(np.isfinite(y).mean())
    return rep


def _solution_voltage_comparison(phis_data: Dict[str, np.ndarray], solution: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    tkey = _find_key(solution, ["t", "t_s", "time_s", "time", "t_eval"])
    vkey = _find_key(solution, ["V_exp", "voltage_exp", "voltage_V", "V_record", "V_meas", "voltage_measured", "voltage"])
    if tkey is None or vkey is None:
        return {"available": False, "reason": "Could not find compatible time/voltage keys in solution.npz", "solution_keys": list(solution.keys())}, None
    if "x_train" not in phis_data or "y_train" not in phis_data:
        return {"available": False, "reason": "data_phis_c.npz lacks x_train/y_train"}, None
    t_label = _as_2d(phis_data["x_train"])[:, 0].reshape(-1)
    v_label = _as_2d(phis_data["y_train"]).reshape(-1)
    t_exp = np.asarray(solution[tkey], dtype=np.float64).reshape(-1)
    v_exp = np.asarray(solution[vkey], dtype=np.float64).reshape(-1)
    mask = np.isfinite(t_exp) & np.isfinite(v_exp)
    t_exp = t_exp[mask]
    v_exp = v_exp[mask]
    if t_exp.size < 2:
        return {"available": False, "reason": "Not enough finite voltage samples in solution.npz"}, None
    order = np.argsort(t_exp)
    t_exp = t_exp[order]
    v_exp = v_exp[order]
    # Remove duplicate times for np.interp.
    t_unique, idx_unique = np.unique(t_exp, return_index=True)
    v_unique = v_exp[idx_unique]
    v_interp = np.interp(t_label, t_unique, v_unique)
    rep = {
        "available": True,
        "time_key": tkey,
        "voltage_key": vkey,
        "metrics_phis_c_vs_experiment": _metrics(v_interp, v_label),
    }
    return rep, (t_label, v_label, v_interp)


def _make_plots(out_dir: Path, datasets: Dict[str, Dict[str, np.ndarray]], solution: Dict[str, np.ndarray], voltage_cmp: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib import failed; skip plots. {exc}")
        return

    # Voltage label vs experimental voltage.
    if voltage_cmp is not None:
        t_label, v_label, v_interp = voltage_cmp
        plt.figure(figsize=(10, 4.8))
        plt.plot(t_label, v_label, label="soft label phis_c")
        plt.plot(t_label, v_interp, label="experiment voltage interpolated", alpha=0.75)
        plt.xlabel("time / s")
        plt.ylabel("voltage / V")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "softlabel_phis_c_vs_experiment.png", dpi=180)
        plt.close()

    # Current profile if available.
    if solution:
        tkey = _find_key(solution, ["t", "t_s", "time_s", "time", "t_eval"])
        ikey = _find_key(solution, ["I_profile", "current_A", "current", "I", "I_A"])
        if tkey is not None and ikey is not None:
            t = np.asarray(solution[tkey]).reshape(-1)
            current = np.asarray(solution[ikey]).reshape(-1)
            if t.size == current.size and t.size > 1:
                plt.figure(figsize=(10, 4.0))
                plt.plot(t, current)
                plt.xlabel("time / s")
                plt.ylabel("current / A")
                plt.tight_layout()
                plt.savefig(out_dir / "solution_current_profile.png", dpi=180)
                plt.close()

    # Surface concentration time series from soft labels.
    for var in ["cs_a", "cs_c"]:
        data = datasets.get(var)
        if not data or "x_train" not in data or "y_train" not in data:
            continue
        x = _as_2d(data["x_train"])
        if x.shape[1] < 2:
            continue
        y = _as_2d(data["y_train"]).reshape(-1)
        r = x[:, 1]
        rmax = np.nanmax(r)
        mask = np.isclose(r, rmax, rtol=1e-7, atol=max(1e-12, abs(rmax) * 1e-7))
        if mask.sum() < 2:
            continue
        t = x[mask, 0]
        yy = y[mask]
        order = np.argsort(t)
        plt.figure(figsize=(10, 4.8))
        plt.plot(t[order], yy[order], label=f"{var} soft label surface")
        plt.xlabel("time / s")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"softlabel_surface_{var}.png", dpi=180)
        plt.close()


def _write_text_report(report: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("ASSB cycle5 soft-label sanity report")
    lines.append("=" * 56)
    lines.append(f"soft_label_dir: {report.get('soft_label_dir')}")
    lines.append("")
    lines.append("[required files]")
    for k, v in report.get("required_files", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("[datasets]")
    for name, rep in report.get("datasets", {}).items():
        lines.append(f"  {name}:")
        for key in ["keys", "x_train_shape", "y_train_shape", "x_params_train_shape", "n_x_equals_n_y", "finite_y_fraction", "t_stats", "r_stats", "y_stats"]:
            if key in rep:
                lines.append(f"    {key}: {rep[key]}")
    lines.append("")
    lines.append("[solution voltage comparison]")
    lines.append(json.dumps(report.get("solution_voltage_comparison", {}), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("[summary json]")
    summ = report.get("soft_label_summary", {})
    for key in ["cycle", "cycle_id", "cycle_from", "merge_cycles", "R_ohm_eff", "voltage_alignment_offset", "voltage_alignment_offset_V", "theta_c_bottom", "theta_c_top", "csanmax", "cscamax"]:
        if key in summ:
            lines.append(f"  {key}: {summ[key]}")
    if "_error" in summ:
        lines.append(f"  error: {summ['_error']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check ASSB cycle5 soft labels")
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3")
    parser.add_argument("--output_dir", default="Eval_cycle5_softlabel_sanity")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    soft_dir = Path(args.soft_label_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {name: str((soft_dir / name).exists()) for name in REQUIRED_NPZ + ["solution.npz", "soft_label_summary.json"]}
    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    report: Dict[str, Any] = {
        "soft_label_dir": str(soft_dir),
        "required_files": required,
        "datasets": {},
    }

    for name in REQUIRED_NPZ:
        stem = name.replace("data_", "").replace(".npz", "")
        data = _load_npz(soft_dir / name)
        datasets[stem] = data
        report["datasets"][stem] = _dataset_report(stem, data)

    solution = _load_npz(soft_dir / "solution.npz") if (soft_dir / "solution.npz").exists() else {}
    summary = _load_json(soft_dir / "soft_label_summary.json")
    report["soft_label_summary"] = summary
    report["solution_keys"] = list(solution.keys()) if solution else []

    voltage_report, voltage_cmp = _solution_voltage_comparison(datasets["phis_c"], solution) if solution else ({"available": False, "reason": "solution.npz not found"}, None)
    report["solution_voltage_comparison"] = voltage_report

    (out_dir / "softlabel_sanity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_text_report(report, out_dir / "softlabel_sanity_report.txt")
    if not args.no_plots:
        _make_plots(out_dir, datasets, solution, voltage_cmp)

    print("Soft-label sanity check finished.")
    print(f"Report TXT : {out_dir / 'softlabel_sanity_report.txt'}")
    print(f"Report JSON: {out_dir / 'softlabel_sanity_report.json'}")
    cmp_rep = report.get("solution_voltage_comparison", {})
    if cmp_rep.get("available"):
        m = cmp_rep["metrics_phis_c_vs_experiment"]
        print(
            "phis_c vs experiment: "
            f"MAE={m['mae']:.6g}, RMSE={m['rmse']:.6g}, MAX={m['maxabs']:.6g}, corr={m['corr']:.6g}"
        )
    else:
        print(f"No voltage comparison: {cmp_rep.get('reason')}")


if __name__ == "__main__":
    main()
