#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GV1 D9.7 battery-8 regime diagnosis plotter.

This diagnostic script reads d97_battery8_diagnosis_summary.json and plots
voltage/residual/regime views from the prediction.npz files referenced in it.
It does not train or modify any model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Use a non-interactive backend so the script works from PowerShell without GUI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_name(text: str, max_len: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text[:max_len] if len(text) > max_len else text


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = np.ravel(arr)
    return arr


def _pick_npz_array(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[np.ndarray]:
    keys = set(npz.files)
    for k in candidates:
        if k in keys:
            return _as_float_array(npz[k])
    # Case-insensitive fallback.
    lower_map = {k.lower(): k for k in npz.files}
    for k in candidates:
        kk = lower_map.get(k.lower())
        if kk is not None:
            return _as_float_array(npz[kk])
    return None


def _finite_common(*arrays: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    valid_arrays = [a for a in arrays if a is not None]
    if not valid_arrays:
        return tuple()
    n = min(len(a) for a in valid_arrays)
    mask = np.ones(n, dtype=bool)
    for a in valid_arrays:
        mask &= np.isfinite(a[:n])
    out = []
    for a in arrays:
        if a is None:
            out.append(None)
        else:
            out.append(a[:n][mask])
    return tuple(out)


def _metrics(y: np.ndarray, yp: np.ndarray) -> Dict[str, Any]:
    y, yp = _finite_common(y, yp)[:2]
    if y is None or yp is None or len(y) == 0:
        return {"n": 0}
    err = yp - y
    corr = None
    if len(y) > 2 and np.nanstd(y) > 0 and np.nanstd(yp) > 0:
        corr = float(np.corrcoef(y, yp)[0, 1])
    return {
        "n": int(len(y)),
        "mae_V": float(np.mean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.mean(err ** 2))),
        "bias_V": float(np.mean(err)),
        "corr": corr,
        "voltage_exp_minmax": [float(np.min(y)), float(np.max(y))],
        "voltage_pred_minmax": [float(np.min(yp)), float(np.max(yp))],
        "pred_upper_frac_ge_4p269": float(np.mean(yp >= 4.269)),
        "pred_overshoot_frac_gt_4p35": float(np.mean(yp > 4.35)),
        "pred_low_voltage_frac_le_2p75": float(np.mean(yp <= 2.75)),
    }


def _load_prediction(pred_path: Path) -> Dict[str, Optional[np.ndarray]]:
    with np.load(pred_path, allow_pickle=False) as d:
        t = _pick_npz_array(d, ["t_global_s", "time_s", "t_s", "time", "t"])
        y = _pick_npz_array(d, ["voltage_exp", "voltage_V", "voltage", "V_exp", "y"])
        yp = _pick_npz_array(d, ["voltage_exp_pred", "voltage_pred", "V_pred", "voltage_model", "y_pred", "phis_c_pred"])
        I = _pick_npz_array(d, ["I_profile", "current_A", "I_A", "current", "I"])
        T = _pick_npz_array(d, ["temperature_C", "T_C", "temperature", "temp_C"])
    if t is None and y is not None:
        t = np.arange(len(y), dtype=float)
    t, y, yp, I, T = _finite_common(t, y, yp, I, T)
    return {"t": t, "voltage_exp": y, "voltage_pred": yp, "current": I, "temperature": T}


def _downsample_idx(n: int, max_points: int = 6000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).round().astype(int)


def _plot_voltage(label: str, data: Dict[str, Optional[np.ndarray]], out: Path) -> None:
    t = data["t"]
    y = data["voltage_exp"]
    yp = data["voltage_pred"]
    if t is None or y is None or yp is None or len(y) == 0:
        return
    idx = _downsample_idx(len(y))
    th = t[idx] / 3600.0
    plt.figure(figsize=(11, 5.5))
    plt.plot(th, y[idx], label="exp", linewidth=1.2)
    plt.plot(th, yp[idx], label="pred", linewidth=1.0)
    plt.axhline(4.269, linestyle="--", linewidth=0.8)
    plt.axhline(4.35, linestyle=":", linewidth=0.8)
    plt.axhline(2.75, linestyle="--", linewidth=0.8)
    plt.xlabel("time / h")
    plt.ylabel("voltage / V")
    plt.title(f"Voltage overlay: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_residual(label: str, data: Dict[str, Optional[np.ndarray]], out: Path) -> None:
    t = data["t"]
    y = data["voltage_exp"]
    yp = data["voltage_pred"]
    if t is None or y is None or yp is None or len(y) == 0:
        return
    idx = _downsample_idx(len(y))
    th = t[idx] / 3600.0
    err = yp - y
    plt.figure(figsize=(11, 4.8))
    plt.plot(th, err[idx], linewidth=0.9)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(0.1, linestyle=":", linewidth=0.8)
    plt.axhline(-0.1, linestyle=":", linewidth=0.8)
    plt.xlabel("time / h")
    plt.ylabel("pred - exp / V")
    plt.title(f"Voltage residual: {label}")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_scatter(label: str, data: Dict[str, Optional[np.ndarray]], out: Path) -> None:
    y = data["voltage_exp"]
    yp = data["voltage_pred"]
    if y is None or yp is None or len(y) == 0:
        return
    idx = _downsample_idx(len(y), max_points=5000)
    lo = float(min(np.min(y), np.min(yp)))
    hi = float(max(np.max(y), np.max(yp)))
    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(y[idx], yp[idx], s=5, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], linewidth=1.0)
    plt.axhline(4.269, linestyle="--", linewidth=0.8)
    plt.axhline(4.35, linestyle=":", linewidth=0.8)
    plt.axvline(2.75, linestyle="--", linewidth=0.8)
    plt.xlabel("exp voltage / V")
    plt.ylabel("pred voltage / V")
    plt.title(f"Pred vs exp: {label}")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_current_temp(label: str, data: Dict[str, Optional[np.ndarray]], out: Path) -> None:
    t = data["t"]
    I = data["current"]
    T = data["temperature"]
    if t is None or (I is None and T is None):
        return
    n = len(t)
    idx = _downsample_idx(n)
    th = t[idx] / 3600.0
    plt.figure(figsize=(11, 4.8))
    if I is not None:
        plt.plot(th, I[idx], label="current / A", linewidth=0.9)
    if T is not None:
        # Put temperature on same figure after centering to make regime transitions visible.
        Tz = T[idx] - float(np.nanmean(T[idx]))
        plt.plot(th, Tz, label="temperature - mean / C", linewidth=0.9)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("time / h")
    plt.title(f"Current and temperature regimes: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_summary_bars(rows: List[Dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    labels = [_safe_name(r.get("label", f"run{i}"), 36) for i, r in enumerate(rows)]
    mae = [float(r.get("mae_V", np.nan)) for r in rows]
    corr = [float(r.get("corr", np.nan)) if r.get("corr") is not None else np.nan for r in rows]
    x = np.arange(len(rows))
    plt.figure(figsize=(max(10, len(rows) * 1.4), 5.2))
    plt.bar(x, mae)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("MAE / V")
    plt.title("D9.7 battery-8 candidate MAE")
    plt.tight_layout()
    plt.savefig(out.with_name(out.stem + "_mae.png"), dpi=160)
    plt.close()
    plt.figure(figsize=(max(10, len(rows) * 1.4), 5.2))
    plt.bar(x, corr)
    plt.axhline(0.90, linestyle="--", linewidth=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("corr")
    plt.title("D9.7 battery-8 candidate correlation")
    plt.tight_layout()
    plt.savefig(out.with_name(out.stem + "_corr.png"), dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot D9.7 battery-8 regime diagnosis from prediction.npz files.")
    parser.add_argument("--summary_json", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\d97_battery8_diagnosis_summary.json")
    parser.add_argument("--output_dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots")
    parser.add_argument("--max_runs", type=int, default=20)
    parser.add_argument("--only", default="", help="Optional substring filter for prediction label.")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preds = summary.get("prediction_summaries", [])
    if args.only:
        preds = [p for p in preds if args.only.lower() in str(p.get("label", "")).lower()]
    preds = preds[: max(0, args.max_runs)]

    manifest: Dict[str, Any] = {
        "ok": True,
        "summary_json": str(summary_path),
        "output_dir": str(out_dir),
        "n_requested": len(preds),
        "plots": [],
        "rows": [],
        "missing": [],
    }

    rows_for_csv: List[Dict[str, Any]] = []
    for pred in preds:
        label = str(pred.get("label") or pred.get("run") or "unknown")
        pred_path = Path(str(pred.get("prediction_npz", "")))
        if not pred_path.exists():
            manifest["missing"].append({"label": label, "prediction_npz": str(pred_path)})
            continue
        data = _load_prediction(pred_path)
        y = data["voltage_exp"]
        yp = data["voltage_pred"]
        if y is None or yp is None:
            manifest["missing"].append({"label": label, "prediction_npz": str(pred_path), "reason": "missing voltage arrays"})
            continue
        met = _metrics(y, yp)
        row = {"label": label, "prediction_npz": str(pred_path), **met}
        rows_for_csv.append(row)
        safe = _safe_name(label)
        files = {
            "voltage_overlay": out_dir / f"{safe}__voltage_overlay.png",
            "residual": out_dir / f"{safe}__residual.png",
            "pred_vs_exp": out_dir / f"{safe}__pred_vs_exp.png",
            "current_temp": out_dir / f"{safe}__current_temp.png",
        }
        _plot_voltage(label, data, files["voltage_overlay"])
        _plot_residual(label, data, files["residual"])
        _plot_scatter(label, data, files["pred_vs_exp"])
        _plot_current_temp(label, data, files["current_temp"])
        manifest["plots"].append({"label": label, **{k: str(v) for k, v in files.items()}})
        manifest["rows"].append(row)

    if rows_for_csv:
        csv_path = out_dir / "d97_candidate_metrics_table.csv"
        fieldnames = list(rows_for_csv[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        manifest["csv"] = str(csv_path)
        _plot_summary_bars(rows_for_csv, out_dir / "d97_candidate_summary.png")

    manifest_path = out_dir / "d97_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "n_plotted": len(manifest["plots"]), "missing": manifest["missing"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
