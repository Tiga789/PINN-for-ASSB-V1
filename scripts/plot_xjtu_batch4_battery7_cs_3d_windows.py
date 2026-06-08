#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_xjtu_batch4_battery7_cs_3d_windows.py

Purpose
-------
Draw 3D concentration-gradient surfaces for XJTU Batch-4 / R3 / battery-7,
using the P5B closed-set NN predictions and P4B-v3 soft-label cycle metadata.

Default target:
    cell_uid = 0023_battery-7_R3_battery-7
    batch    = Batch-4
    protocol = R3

Outputs:
    3 windows x 2 states = 6 figures
      cs_a: window 1/2/3
      cs_c: window 1/2/3

Each window spans 4 unique cycles. The title includes:
    - cell / batch / protocol
    - state name
    - cycle interval/list
    - global prediction metrics: R², NRMSE, MAE
    - window prediction metrics: R², NRMSE, MAE

Default input paths assume:
    P5B root:
      E:/XJTU battery dataset/_gv1_cache/xjtu_d14_p5b_8cell_closedset_precision_v2

    P4B-v3 soft-label root:
      E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_v1_p4b_multicell_v3

Run in PyCharm:
    1. Put this script anywhere inside the project, e.g. scripts/.
    2. Open it in PyCharm.
    3. Run directly, or edit DEFAULT_* paths below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# =========================
# Default local paths
# =========================

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")

DEFAULT_P5B_ROOT = Path(r"E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision_v2")

DEFAULT_P4B_SOFTLABEL_ROOT = Path(r"E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3")

DEFAULT_CELL_UID = "0023_battery-7_R3_battery-7"

DEFAULT_OUTPUT_DIR = DEFAULT_P5B_ROOT / "EvalFin_D14_P5B_8cell_closedset_precision" / "plots_3d_Batch4_battery7_cs_windows"


# =========================
# General helpers
# =========================

def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _format_metric(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.{digits}f}"
    return f"{x:.{digits}g}"


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 1e-30:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    d = yt = y_true[mask]
    d = y_pred[mask] - yt
    return float(np.sqrt(np.mean(d * d)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mask = np.isfinite(y_true)
    if mask.sum() < 3:
        return float("nan")
    denom = float(np.nanmax(y_true[mask]) - np.nanmin(y_true[mask]))
    if denom <= 1e-30:
        denom = max(abs(float(np.nanmean(y_true[mask]))), 1e-30)
    return _rmse(y_true, y_pred) / denom


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": _r2_score(y_true, y_pred),
        "nrmse": _nrmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
    }


def metric_text(prefix: str, m: Dict[str, float]) -> str:
    return (
        f"{prefix}: R²={_format_metric(m['r2'])}, "
        f"NRMSE={_format_metric(m['nrmse'])}, "
        f"MAE={_format_metric(m['mae'])}"
    )


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# =========================
# Input discovery
# =========================

def resolve_prediction_npz(p5b_root: Path, cell_uid: str) -> Path:
    pred = (
        p5b_root
        / "EvalFin_D14_P5B_8cell_closedset_precision"
        / "predictions"
        / cell_uid
        / "prediction_sampled.npz"
    )
    if pred.exists():
        return pred

    # Fallback search
    candidates = list((p5b_root / "EvalFin_D14_P5B_8cell_closedset_precision").rglob("prediction_sampled.npz"))
    for c in candidates:
        if cell_uid in str(c):
            return c
    raise FileNotFoundError(f"Cannot find prediction_sampled.npz for {cell_uid} under {p5b_root}")


def resolve_softlabel_npz(p4b_softlabel_root: Path, cell_uid: str) -> Path:
    soft = p4b_softlabel_root / "profiles" / cell_uid / "solution_softlabels.npz"
    if soft.exists():
        return soft

    # Fallback search
    candidates = list(p4b_softlabel_root.rglob("solution_softlabels.npz"))
    for c in candidates:
        if cell_uid in str(c):
            return c
    raise FileNotFoundError(f"Cannot find solution_softlabels.npz for {cell_uid} under {p4b_softlabel_root}")


def load_metric_row(p5b_root: Path, cell_uid: str) -> Dict[str, str]:
    path = p5b_root / "EvalFin_D14_P5B_8cell_closedset_precision" / "metrics_by_profile.csv"
    rows = _read_csv_rows(path)
    for r in rows:
        if r.get("cell_uid", "") == cell_uid:
            return r
    return {}


# =========================
# Cycle matching
# =========================

def nearest_indices(sorted_ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return nearest indices in sorted_ref for each query value."""
    sorted_ref = np.asarray(sorted_ref, dtype=float)
    query = np.asarray(query, dtype=float)
    pos = np.searchsorted(sorted_ref, query, side="left")
    pos0 = np.clip(pos - 1, 0, len(sorted_ref) - 1)
    pos1 = np.clip(pos, 0, len(sorted_ref) - 1)
    d0 = np.abs(sorted_ref[pos0] - query)
    d1 = np.abs(sorted_ref[pos1] - query)
    return np.where(d1 < d0, pos1, pos0)


def attach_cycle_id_to_prediction(pred_npz: np.lib.npyio.NpzFile, soft_npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
    """Attach cycle_id to sampled prediction time points using nearest t_global_s."""
    pred_t = np.asarray(pred_npz["t_global_s"], dtype=float)
    soft_t = np.asarray(soft_npz["t_global_s"], dtype=float)
    soft_cycle = np.asarray(soft_npz["cycle_id"]).reshape(-1)

    if len(soft_t) != len(soft_cycle):
        raise ValueError(f"soft_t length {len(soft_t)} != cycle_id length {len(soft_cycle)}")

    idx = nearest_indices(soft_t, pred_t)
    pred_cycle = soft_cycle[idx]
    match_dt = np.abs(soft_t[idx] - pred_t)
    return pred_cycle, match_dt


def parse_cycle_windows(s: str) -> Optional[List[List[int]]]:
    """Parse windows like '1-4,5-8,9-12' or '1,2,3,4;10,11,12,13'."""
    if not s:
        return None
    windows: List[List[int]] = []
    chunks = re.split(r"[;|]", s)
    if len(chunks) == 1 and "," in s and "-" in s:
        chunks = s.split(",")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            windows.append(list(range(a, b + 1)))
        else:
            vals = [int(x) for x in re.split(r"[,\s]+", chunk) if x.strip()]
            windows.append(vals)
    return windows if windows else None


def choose_auto_windows(cycle_id: np.ndarray, window_len: int = 4, n_windows: int = 3) -> List[List[int]]:
    cycles = np.asarray(cycle_id)
    cycles = cycles[np.isfinite(cycles.astype(float))]
    unique = sorted(set(int(x) for x in cycles))
    if len(unique) == 0:
        raise ValueError("No valid cycles found.")
    if len(unique) <= window_len:
        return [unique]

    starts = [0, max(0, (len(unique) - window_len) // 2), max(0, len(unique) - window_len)]
    windows = []
    seen = set()
    for st in starts:
        w = unique[st:st + window_len]
        key = tuple(w)
        if key not in seen and len(w) > 0:
            windows.append(w)
            seen.add(key)
    # If dedup leaves fewer than n_windows, fill from non-overlapping starts.
    st = 0
    while len(windows) < n_windows and st <= len(unique) - window_len:
        w = unique[st:st + window_len]
        key = tuple(w)
        if key not in seen:
            windows.append(w)
            seen.add(key)
        st += window_len
    return windows[:n_windows]


# =========================
# Surface plotting
# =========================

def downsample_time_for_surface(mask_indices: np.ndarray, max_time_points: int) -> np.ndarray:
    if len(mask_indices) <= max_time_points:
        return mask_indices
    take = np.linspace(0, len(mask_indices) - 1, max_time_points).round().astype(int)
    return mask_indices[take]


def get_state_arrays(pred: np.lib.npyio.NpzFile, state: str) -> Tuple[np.ndarray, np.ndarray]:
    pred_key = f"{state}_pred"
    true_key = f"{state}_true"
    if pred_key not in pred.files or true_key not in pred.files:
        raise KeyError(f"Prediction NPZ must contain {pred_key} and {true_key}. Existing keys: {pred.files}")
    return np.asarray(pred[true_key], dtype=float), np.asarray(pred[pred_key], dtype=float)


def plot_state_window(
    *,
    pred: np.lib.npyio.NpzFile,
    cycle_id: np.ndarray,
    window_cycles: List[int],
    state: str,
    cell_uid: str,
    batch: str,
    protocol: str,
    output_dir: Path,
    max_time_points: int = 260,
    surface_source: str = "pred",
    overlay_true_wire: bool = True,
    cmap: str = "viridis",
    elev: float = 28,
    azim: float = -126,
    dpi: int = 220,
    save_pdf: bool = False,
):
    t = np.asarray(pred["t_global_s"], dtype=float)
    I = np.asarray(pred["I_profile"], dtype=float) if "I_profile" in pred.files else None
    V = np.asarray(pred["voltage_exp"], dtype=float) if "voltage_exp" in pred.files else None

    y_true, y_pred = get_state_arrays(pred, state)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(f"{state}_true/pred must be 2D (time, r), got {y_true.shape}, {y_pred.shape}")

    mask = np.isin(cycle_id.astype(int), np.asarray(window_cycles, dtype=int))
    idx_all = np.where(mask)[0]
    if len(idx_all) < 10:
        raise ValueError(f"Too few points for {state} cycles={window_cycles}: {len(idx_all)}")

    idx = downsample_time_for_surface(idx_all, max_time_points)
    z_true = y_true[idx, :]
    z_pred = y_pred[idx, :]
    z_plot = z_pred if surface_source == "pred" else z_true

    t_hours = (t[idx] - t[idx][0]) / 3600.0
    r_norm = np.linspace(0.0, 1.0, z_plot.shape[1])
    T_mesh, R_mesh = np.meshgrid(t_hours, r_norm, indexing="ij")

    global_metrics = metric_pack(y_true, y_pred)
    window_metrics = metric_pack(z_true, z_pred)

    z_min = float(np.nanmin([np.nanmin(z_true), np.nanmin(z_pred)]))
    z_max = float(np.nanmax([np.nanmax(z_true), np.nanmax(z_pred)]))
    z_pad = 0.02 * max(abs(z_max - z_min), 1.0)

    fig = plt.figure(figsize=(14.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=z_min, vmax=z_max)
    surf = ax.plot_surface(
        T_mesh,
        R_mesh,
        z_plot,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        rcount=min(max_time_points, z_plot.shape[0]),
        ccount=z_plot.shape[1],
        alpha=0.92,
    )

    if overlay_true_wire:
        # True target as a sparse wireframe so prediction/target mismatch can be spotted.
        wire_stride_t = max(1, z_true.shape[0] // 34)
        ax.plot_wireframe(
            T_mesh,
            R_mesh,
            z_true,
            rstride=wire_stride_t,
            cstride=2,
            linewidth=0.35,
            alpha=0.35,
        )

    ax.set_xlabel("window-local time / h", labelpad=10)
    ax.set_ylabel("normalized radius r/R", labelpad=10)
    ax.set_zlabel(f"{state} concentration / mol m$^{{-3}}$", labelpad=10)

    ax.set_zlim(z_min - z_pad, z_max + z_pad)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.65, 0.75, 0.82))

    cycle_label = f"{min(window_cycles)}-{max(window_cycles)}" if len(window_cycles) > 1 else str(window_cycles[0])
    title = (
        f"{batch} / {protocol} / battery-7 | {state} | cycles {cycle_label}  "
        f"(cycle list: {window_cycles})\n"
        f"{metric_text('Global', global_metrics)} | {metric_text('Window', window_metrics)}"
    )
    ax.set_title(title, pad=22, fontsize=11)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.68, aspect=18, pad=0.08)
    cbar.set_label(f"{state} concentration / mol m$^{{-3}}$")

    # Add a small text note for current/voltage range in this window.
    note_lines = []
    if I is not None:
        note_lines.append(f"I range: {np.nanmin(I[idx_all]):.3g} to {np.nanmax(I[idx_all]):.3g} A")
    if V is not None:
        note_lines.append(f"V range: {np.nanmin(V[idx_all]):.4g} to {np.nanmax(V[idx_all]):.4g} V")
    if note_lines:
        fig.text(0.02, 0.02, " | ".join(note_lines), fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname_base = sanitize_filename(f"{cell_uid}_{state}_cycles_{cycle_label}_3d")
    png_path = output_dir / f"{fname_base}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{fname_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "state": state,
        "cycle_label": cycle_label,
        "cycles": window_cycles,
        "png": str(png_path),
        "global_r2": global_metrics["r2"],
        "global_nrmse": global_metrics["nrmse"],
        "global_mae": global_metrics["mae"],
        "window_r2": window_metrics["r2"],
        "window_nrmse": window_metrics["nrmse"],
        "window_mae": window_metrics["mae"],
        "n_points_window": int(len(idx_all)),
        "n_points_plotted": int(len(idx)),
    }


def write_summary_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p5b_root", default=str(DEFAULT_P5B_ROOT), help="D14-P5B output root.")
    parser.add_argument("--p4b_softlabel_root", default=str(DEFAULT_P4B_SOFTLABEL_ROOT), help="D14-P4B-v3 soft-label root.")
    parser.add_argument("--cell_uid", default=DEFAULT_CELL_UID)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cycle_windows", default="", help="Optional windows, e.g. '1-4,5-8,9-12' or '1 2 3 4; 10 11 12 13'.")
    parser.add_argument("--window_len", type=int, default=4)
    parser.add_argument("--n_windows", type=int, default=3)
    parser.add_argument("--max_time_points", type=int, default=260)
    parser.add_argument("--surface_source", choices=["pred", "true"], default="pred")
    parser.add_argument("--no_true_wire", action="store_true", help="Disable target wireframe overlay.")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving. For PyCharm this may open windows.")

    args = parser.parse_args()

    p5b_root = Path(args.p5b_root)
    p4b_root = Path(args.p4b_softlabel_root)
    output_dir = Path(args.output_dir)

    pred_path = resolve_prediction_npz(p5b_root, args.cell_uid)
    soft_path = resolve_softlabel_npz(p4b_root, args.cell_uid)
    metrics_row = load_metric_row(p5b_root, args.cell_uid)

    print(f"[INFO] prediction NPZ: {pred_path}")
    print(f"[INFO] soft-label NPZ: {soft_path}")
    print(f"[INFO] output dir: {output_dir}")

    pred = np.load(pred_path, allow_pickle=True)
    soft = np.load(soft_path, allow_pickle=True)

    pred_cycle, match_dt = attach_cycle_id_to_prediction(pred, soft)
    if np.nanmax(match_dt) > 5.0:
        print(f"[WARN] max nearest time mismatch = {np.nanmax(match_dt):.3f} s. Check t_global_s alignment.")

    windows = parse_cycle_windows(args.cycle_windows)
    if windows is None:
        windows = choose_auto_windows(pred_cycle, window_len=args.window_len, n_windows=args.n_windows)

    # Keep exactly n_windows if possible.
    windows = windows[:args.n_windows]
    print(f"[INFO] selected cycle windows: {windows}")

    batch = metrics_row.get("batch", "Batch-4") or "Batch-4"
    protocol = metrics_row.get("protocol", "R3") or "R3"

    rows = []
    for window in windows:
        for state in ["cs_a", "cs_c"]:
            row = plot_state_window(
                pred=pred,
                cycle_id=pred_cycle,
                window_cycles=window,
                state=state,
                cell_uid=args.cell_uid,
                batch=batch,
                protocol=protocol,
                output_dir=output_dir,
                max_time_points=args.max_time_points,
                surface_source=args.surface_source,
                overlay_true_wire=not args.no_true_wire,
                cmap=args.cmap,
                dpi=args.dpi,
                save_pdf=args.save_pdf,
            )
            rows.append(row)
            print(f"[OK] {state} cycles={row['cycle_label']} -> {row['png']}")

    summary_csv = output_dir / f"{sanitize_filename(args.cell_uid)}_3d_cs_window_metrics_summary.csv"
    write_summary_csv(summary_csv, rows)

    summary_json = output_dir / f"{sanitize_filename(args.cell_uid)}_3d_cs_window_metrics_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cell_uid": args.cell_uid,
                "prediction_npz": str(pred_path),
                "softlabel_npz": str(soft_path),
                "output_dir": str(output_dir),
                "cycle_windows": windows,
                "figures": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] Generated {len(rows)} figures.")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] Summary JSON: {summary_json}")

    if args.show:
        # Re-open generated images is intentionally not done; figures were closed after saving.
        print("[INFO] --show requested, but figures are saved and closed to avoid memory growth.")


if __name__ == "__main__":
    main()
