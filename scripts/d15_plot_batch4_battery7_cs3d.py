# -*- coding: utf-8 -*-
"""
D15 interactive 3D concentration-surface plots for XJTU Batch-4 R3 battery-7.

Purpose
-------
Draw 6 draggable Matplotlib 3D popup figures in PyCharm:
  - cs_a predicted surface, 3 windows, each window contains 4 cycles.
  - cs_c predicted surface, 3 windows, each window contains 4 cycles.

The figure title reports the full-profile prediction accuracy and the local
window accuracy (R², NRMSE, MAE) plus the cycle interval.

Default data source
-------------------
Soft labels:
  E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
D15-P1 model/eval directory:
  E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p1_rg_closedset_nn_smoke
Profile key:
  Batch-4_R3_battery-7

Run in PyCharm terminal from the project root:
  python scripts/d15_plot_batch4_battery7_cs3d.py

Optional manual cycle windows:
  python scripts/d15_plot_batch4_battery7_cs3d.py --cycle-window 20:23 --cycle-window 120:123 --cycle-window 240:243

Notes
-----
- This script does not modify any project file or cache.
- It predicts from the D15-P1 checkpoint by default. Use --surface-source true
  if you only want to visualize the RG soft-label surfaces without NN prediction.
- For draggable 3D windows in PyCharm, use a GUI backend such as Qt5Agg/TkAgg
  and disable non-interactive Scientific inline plotting if needed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _project_root_from_script() -> Path:
    # Expected location: <project_root>/scripts/d15_plot_batch4_battery7_cs3d.py
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()


PROJECT_ROOT = _project_root_from_script()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive 3D cs_a/cs_c surfaces for Batch-4 R3 battery-7, 3 windows x 4 cycles."
    )
    p.add_argument(
        "--softlabel-dir",
        default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell",
        help="D15-P0 P2Dlite-RG soft-label root directory.",
    )
    p.add_argument(
        "--run-dir",
        default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p1_rg_closedset_nn_smoke",
        help="D15-P1 closed-set NN run directory containing model/best_with_state.pt.",
    )
    p.add_argument(
        "--profile-key",
        default="Batch-4_R3_battery-7",
        help="Substring used to find the target profile directory.",
    )
    p.add_argument("--filename", default="solution_softlabels.npz")
    p.add_argument(
        "--surface-source",
        choices=["pred", "true"],
        default="pred",
        help="Plot NN prediction surfaces or RG soft-label true surfaces.",
    )
    p.add_argument(
        "--plot-unit",
        choices=["cs", "theta"],
        default="cs",
        help="Z-axis unit. Default cs uses cs_a/cs_c mol/m^3 when available; prediction cs is reconstructed from theta*csmax.",
    )
    p.add_argument(
        "--window-cycles",
        type=int,
        default=4,
        help="Number of cycles per 3D figure. Default: 4.",
    )
    p.add_argument(
        "--cycle-window",
        action="append",
        default=[],
        help="Manual cycle window as start:end, inclusive. Provide exactly three for fixed windows, e.g. --cycle-window 20:23.",
    )
    p.add_argument(
        "--max-time-points-per-plot",
        type=int,
        default=900,
        help="Decimate time axis for smooth interactive plotting. Radial grid is not decimated.",
    )
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-135.0)
    p.add_argument("--save-png", action="store_true", help="Also save PNG files.")
    p.add_argument(
        "--out-dir",
        default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_d15_batch4_battery7_cs3d_plots",
        help="Output directory for optional PNG and window report.",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open interactive windows; useful if only saving PNG.")
    p.add_argument(
        "--backend",
        default="",
        help="Optional Matplotlib backend, e.g. Qt5Agg or TkAgg. Leave empty for PyCharm default.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Prediction device.")
    p.add_argument("--batch-size", type=int, default=65536)
    return p.parse_args()


def load_npz_dict(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def first_key(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def as_1d_float(d: Mapping[str, Any], keys: Sequence[str], name: str, required: bool = True) -> Optional[np.ndarray]:
    k = first_key(d, keys)
    if k is None:
        if required:
            raise KeyError(f"Missing {name}; tried {keys}")
        return None
    arr = np.asarray(d[k])
    if arr.dtype.kind in {"U", "S", "O"}:
        # Attempt numeric conversion from object/string arrays.
        try:
            arr = arr.astype(float)
        except Exception as exc:
            raise TypeError(f"{k} is not numeric and cannot be converted: {exc}")
    return arr.astype(np.float64).reshape(-1)


def orient_time_radial(d: Mapping[str, Any], keys: Sequence[str], n_time: int, name: str, required: bool = True) -> Optional[np.ndarray]:
    k = first_key(d, keys)
    if k is None:
        if required:
            raise KeyError(f"Missing {name}; tried {keys}")
        return None
    arr = np.asarray(d[k])
    if arr.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"{k} is non-numeric")
    arr = arr.astype(np.float64)
    if arr.ndim == 1:
        if arr.size != n_time:
            raise ValueError(f"{k}: length {arr.size} != n_time={n_time}")
        return arr.reshape(n_time, 1)
    if arr.ndim != 2:
        raise ValueError(f"{k}: expected 1D/2D, got shape={arr.shape}")
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f"{k}: cannot orient shape={arr.shape} with n_time={n_time}")


def cycle_ids_to_int(raw: Any, n_time: int) -> np.ndarray:
    arr = np.asarray(raw).reshape(-1)
    if arr.size != n_time:
        raise ValueError(f"cycle_id length {arr.size} != n_time={n_time}")
    if arr.dtype.kind in {"i", "u", "f"}:
        return arr.astype(int)
    out = []
    fallback_map: Dict[str, int] = {}
    next_id = 1
    for val in arr:
        s = str(val)
        m = re.search(r"[-+]?\d+", s)
        if m:
            out.append(int(m.group(0)))
        else:
            if s not in fallback_map:
                fallback_map[s] = next_id
                next_id += 1
            out.append(fallback_map[s])
    return np.asarray(out, dtype=int)


def find_profile_npz(root: Path, profile_key: str, filename: str) -> Path:
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} found under {root}")
    matches = [p for p in files if profile_key.lower() in str(p.parent).lower()]
    if not matches:
        preview = "\n".join(str(p.parent.relative_to(root)) for p in files[:30])
        raise FileNotFoundError(
            f"Could not find profile containing {profile_key!r} under {root}.\n"
            f"First available profiles:\n{preview}"
        )
    if len(matches) > 1:
        # Prefer exact parent name match if possible.
        exact = [p for p in matches if p.parent.name.lower() == profile_key.lower()]
        if exact:
            matches = exact
    return matches[0]


def estimate_csmax(theta: np.ndarray, cs: Optional[np.ndarray]) -> float:
    if cs is None:
        return 1.0
    th = np.asarray(theta, dtype=float)
    c = np.asarray(cs, dtype=float)
    mask = np.isfinite(th) & np.isfinite(c) & (th > 0.02) & (th < 0.98) & (c > 0)
    if mask.sum() < 100:
        mask = np.isfinite(th) & np.isfinite(c) & (th > 1e-6) & (c > 0)
    if mask.sum() == 0:
        mx = float(np.nanmax(c))
        return mx if np.isfinite(mx) and mx > 0 else 1.0
    ratio = c[mask] / th[mask]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return 1.0
    return float(np.nanmedian(ratio))


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(p)
    if mask.sum() < 3:
        return {"count": int(mask.sum()), "mae": math.nan, "rmse": math.nan, "r2": math.nan, "nrmse": math.nan, "corr": math.nan}
    t = t[mask]
    p = p[mask]
    e = p - t
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    denom = float(np.sum((t - np.mean(t)) ** 2))
    r2 = float(1.0 - np.sum(e ** 2) / denom) if denom > 1e-18 else math.nan
    rng = float(np.max(t) - np.min(t))
    nrmse = float(rmse / rng) if rng > 1e-18 else math.nan
    corr = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 1e-12 and np.std(p) > 1e-12 else math.nan
    return {"count": int(mask.sum()), "mae": mae, "rmse": rmse, "r2": r2, "nrmse": nrmse, "corr": corr}


def fmt_metric(m: Dict[str, float]) -> str:
    def f(x: float, digits: int = 4) -> str:
        return "nan" if not np.isfinite(x) else f"{x:.{digits}f}"
    return f"R²={f(m['r2'])}, NRMSE={f(100.0 * m['nrmse'], 2)}%, MAE={f(m['mae'], 5)}"


def load_profile_true(npz_path: Path) -> Dict[str, Any]:
    d = load_npz_dict(npz_path)
    t = as_1d_float(d, ["t_global_s", "time_s", "t_s", "t", "time"], "time")
    assert t is not None
    n = t.size
    I = as_1d_float(d, ["I_profile", "current_A", "I_A", "current", "I"], "current", required=False)
    if I is None:
        I = np.zeros(n, dtype=float)
    cycle_key = first_key(d, ["cycle_id", "cycle", "cycle_index"])
    if cycle_key is None:
        raise KeyError("Missing cycle_id in soft-label npz; cannot make 4-cycle windows.")
    cycle_id = cycle_ids_to_int(d[cycle_key], n)
    theta_a = orient_time_radial(d, ["theta_a", "theta_n", "theta_negative"], n, "theta_a")
    theta_c = orient_time_radial(d, ["theta_c", "theta_p", "theta_positive"], n, "theta_c")
    cs_a = orient_time_radial(d, ["cs_a", "cs_n", "cs_negative"], n, "cs_a", required=False)
    cs_c = orient_time_radial(d, ["cs_c", "cs_p", "cs_positive"], n, "cs_c", required=False)
    r_a = as_1d_float(d, ["r_a", "r_n", "r_negative"], "r_a", required=False)
    r_c = as_1d_float(d, ["r_c", "r_p", "r_positive"], "r_c", required=False)
    if r_a is None:
        r_a = np.linspace(0.0, 1.0, theta_a.shape[1])
    if r_c is None:
        r_c = np.linspace(0.0, 1.0, theta_c.shape[1])
    return {
        "npz_raw": d,
        "t": t.astype(float),
        "I": np.asarray(I, dtype=float),
        "cycle_id": cycle_id,
        "theta_a_true": theta_a.astype(float),
        "theta_c_true": theta_c.astype(float),
        "cs_a_true": None if cs_a is None else cs_a.astype(float),
        "cs_c_true": None if cs_c is None else cs_c.astype(float),
        "r_a": np.asarray(r_a, dtype=float),
        "r_c": np.asarray(r_c, dtype=float),
    }


def _device_from_arg(device_arg: str):
    import torch
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def model_file_from_run_dir(run_dir: Path) -> Path:
    candidates = [run_dir / "model" / "best_with_state.pt", run_dir / "best_with_state.pt"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find best_with_state.pt under {run_dir} or {run_dir / 'model'}")


def predict_from_d15p1(softlabel_dir: Path, npz_path: Path, run_dir: Path, device_arg: str, batch_size: int) -> Dict[str, Any]:
    import torch
    from gv1.p2dlite_rg_nn.data import build_features, build_targets, load_profile_arrays, profile_id_from_path
    from gv1.p2dlite_rg_nn.model import build_model
    from gv1.p2dlite_rg_nn.train_eval import predict_numpy

    mf = model_file_from_run_dir(run_dir)
    device = _device_from_arg(device_arg)
    ck = torch.load(mf, map_location=device, weights_only=False)
    state = ck["state"]
    model = build_model(int(state["input_dim"]), int(state["output_dim"]), state["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    pid = profile_id_from_path(npz_path, softlabel_dir)
    profile_ids = list(state.get("profile_ids", []))
    try:
        profile_index = profile_ids.index(pid)
    except ValueError:
        # Fallback sorted order should only be used if profile_ids changed.
        files = sorted(softlabel_dir.rglob(npz_path.name))
        profile_index = files.index(npz_path) if npz_path in files else 0
        print(f"[WARN] profile_id {pid!r} not found in model state; fallback profile_index={profile_index}")
    prof = load_profile_arrays(npz_path, softlabel_dir)
    X, _ = build_features(prof, profile_index, len(profile_ids), include_profile_onehot=bool(state.get("include_profile_onehot", True)))
    Y_true, target_names, slices = build_targets(prof)
    Y_pred = predict_numpy(
        model,
        X,
        np.asarray(state["x_mean"], dtype=np.float32),
        np.asarray(state["x_std"], dtype=np.float32),
        np.asarray(state["y_mean"], dtype=np.float32),
        np.asarray(state["y_std"], dtype=np.float32),
        device,
        batch_size=batch_size,
    )
    sa = slices["theta_a"]
    sc = slices["theta_c"]
    return {
        "profile_id": pid,
        "model_file": str(mf),
        "target_names": target_names,
        "target_slices": slices,
        "theta_a_pred": Y_pred[:, sa[0]:sa[1]].astype(float),
        "theta_c_pred": Y_pred[:, sc[0]:sc[1]].astype(float),
        "theta_a_true_model": Y_true[:, sa[0]:sa[1]].astype(float),
        "theta_c_true_model": Y_true[:, sc[0]:sc[1]].astype(float),
    }


def parse_manual_windows(values: Sequence[str]) -> List[Tuple[int, int]]:
    out = []
    for v in values:
        m = re.match(r"^\s*(-?\d+)\s*[:,-]\s*(-?\d+)\s*$", v)
        if not m:
            raise ValueError(f"Invalid --cycle-window {v!r}; use start:end, inclusive.")
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out


def cycle_current_summary(cycle_id: np.ndarray, I: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in np.unique(cycle_id):
        mask = cycle_id == c
        Ii = I[mask]
        # focus on discharge magnitude if negative-current discharge exists; otherwise non-rest abs current.
        dis = np.abs(Ii[Ii < -1e-9])
        if dis.size == 0:
            dis = np.abs(Ii[np.abs(Ii) > 1e-9])
        out[int(c)] = float(np.nanmean(dis)) if dis.size else 0.0
    return out


def choose_auto_windows(cycle_id: np.ndarray, I: np.ndarray, window_cycles: int) -> List[Tuple[int, int]]:
    cycles = np.asarray(sorted(np.unique(cycle_id).astype(int)))
    if cycles.size < window_cycles:
        raise ValueError(f"Need at least {window_cycles} cycles, got {cycles.size}")
    curr = cycle_current_summary(cycle_id, I)
    candidates: List[Tuple[int, int, float]] = []
    for i in range(0, cycles.size - window_cycles + 1):
        win = cycles[i:i + window_cycles]
        vals = np.array([curr.get(int(c), 0.0) for c in win], dtype=float)
        # Rounded current levels expose Batch-4 varying xC discharge patterns while staying robust to noise.
        if np.nanmax(np.abs(vals)) > 0:
            rounded = np.round(vals / max(np.nanmax(np.abs(vals)), 1e-12), 2)
            diversity = len(np.unique(rounded))
            spread = float(np.nanmax(vals) - np.nanmin(vals)) / max(float(np.nanmax(vals)), 1e-12)
        else:
            diversity = 0
            spread = 0.0
        score = float(diversity + 0.25 * spread)
        candidates.append((int(win[0]), int(win[-1]), score))
    # Pick one high-diversity window from each temporal third to compare early/mid/late behavior.
    thirds = np.array_split(np.arange(len(candidates)), 3)
    selected: List[Tuple[int, int]] = []
    for block in thirds:
        if block.size == 0:
            continue
        best_i = max(block.tolist(), key=lambda j: candidates[j][2])
        selected.append((candidates[best_i][0], candidates[best_i][1]))
    # Fill any missing with quantile windows, avoiding duplicates.
    if len(selected) < 3:
        qs = np.linspace(0, len(candidates) - 1, 3).round().astype(int)
        for j in qs:
            w = (candidates[j][0], candidates[j][1])
            if w not in selected:
                selected.append(w)
            if len(selected) == 3:
                break
    return selected[:3]


def decimate_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def normalized_r(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float).reshape(-1)
    if r.size == 0:
        return np.array([0.0])
    mn, mx = float(np.nanmin(r)), float(np.nanmax(r))
    if np.isfinite(mx) and abs(mx) > 1e-18:
        return r / mx
    if np.isfinite(mx - mn) and abs(mx - mn) > 1e-18:
        return (r - mn) / (mx - mn)
    return np.linspace(0.0, 1.0, r.size)


def make_surface_figure(
    *,
    electrode: str,
    cycle_window: Tuple[int, int],
    t: np.ndarray,
    cycle_id: np.ndarray,
    r: np.ndarray,
    Z_plot: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    full_metrics: Dict[str, float],
    args: argparse.Namespace,
    profile_label: str,
    out_dir: Path,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    c0, c1 = cycle_window
    mask = (cycle_id >= c0) & (cycle_id <= c1)
    if mask.sum() < 5:
        raise ValueError(f"Cycle window {c0}:{c1} contains only {mask.sum()} time points")
    idx_all = np.flatnonzero(mask)
    dec = decimate_indices(idx_all.size, args.max_time_points_per_plot)
    idx = idx_all[dec]

    tw = t[idx]
    th_true_w = theta_true[idx]
    th_pred_w = theta_pred[idx]
    Z = Z_plot[idx]
    win_metrics = basic_metrics(th_true_w, th_pred_w)
    rnorm = normalized_r(r)
    xh = (tw - tw[0]) / 3600.0
    X = np.repeat(xh[:, None], rnorm.size, axis=1)
    Y = np.repeat(rnorm[None, :], xh.size, axis=0)

    fig = plt.figure(figsize=(13.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=args.cmap, linewidth=0, antialiased=True, shade=True)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10, label=(f"cs_{electrode} [mol/m³]" if args.plot_unit == "cs" else f"theta_{electrode}"))
    ax.set_xlabel("time from window start [h]")
    ax.set_ylabel("r / R")
    ax.set_zlabel(f"cs_{electrode} [mol/m³]" if args.plot_unit == "cs" else f"theta_{electrode}")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(
        f"D15-P1 {args.surface_source.upper()} surface: cs_{electrode} | {profile_label} | cycles {c0}-{c1}\n"
        f"Full-profile theta: {fmt_metric(full_metrics)} | Window theta: {fmt_metric(win_metrics)}",
        fontsize=10,
        pad=18,
    )
    plt.tight_layout()
    out: Dict[str, Any] = {
        "electrode": electrode,
        "cycle_start": int(c0),
        "cycle_end": int(c1),
        "n_time_original": int(idx_all.size),
        "n_time_plotted": int(idx.size),
        "full_metrics": full_metrics,
        "window_metrics": win_metrics,
    }
    if args.save_png:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"D15_Batch4_battery7_{args.surface_source}_cs_{electrode}_cycles_{c0}_{c1}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=180, bbox_inches="tight")
        out["png"] = str(path)
    return out


def main() -> int:
    args = parse_args()
    if args.backend:
        import matplotlib
        matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    softlabel_dir = Path(args.softlabel_dir)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    npz_path = find_profile_npz(softlabel_dir, args.profile_key, args.filename)
    true = load_profile_true(npz_path)
    profile_label = str(npz_path.parent.relative_to(softlabel_dir)).replace("\\", "/")

    theta_a_true = true["theta_a_true"]
    theta_c_true = true["theta_c_true"]

    if args.surface_source == "pred":
        pred = predict_from_d15p1(softlabel_dir, npz_path, run_dir, args.device, args.batch_size)
        theta_a_pred = pred["theta_a_pred"]
        theta_c_pred = pred["theta_c_pred"]
        print(f"[D15 plot] using model: {pred['model_file']}")
    else:
        theta_a_pred = theta_a_true.copy()
        theta_c_pred = theta_c_true.copy()
        print("[D15 plot] surface-source=true: plotting RG soft-label surfaces; prediction metrics will be perfect by construction.")

    if theta_a_pred.shape != theta_a_true.shape or theta_c_pred.shape != theta_c_true.shape:
        raise ValueError(f"Prediction/true shape mismatch: a {theta_a_pred.shape} vs {theta_a_true.shape}; c {theta_c_pred.shape} vs {theta_c_true.shape}")

    csmax_a = estimate_csmax(theta_a_true, true["cs_a_true"])
    csmax_c = estimate_csmax(theta_c_true, true["cs_c_true"])
    if args.plot_unit == "cs":
        if args.surface_source == "pred":
            Za = theta_a_pred * csmax_a
            Zc = theta_c_pred * csmax_c
        else:
            Za = true["cs_a_true"] if true["cs_a_true"] is not None else theta_a_true * csmax_a
            Zc = true["cs_c_true"] if true["cs_c_true"] is not None else theta_c_true * csmax_c
    else:
        Za = theta_a_pred if args.surface_source == "pred" else theta_a_true
        Zc = theta_c_pred if args.surface_source == "pred" else theta_c_true

    manual_windows = parse_manual_windows(args.cycle_window)
    if manual_windows:
        if len(manual_windows) != 3:
            raise ValueError(f"Please provide exactly 3 --cycle-window values, got {len(manual_windows)}")
        windows = manual_windows
    else:
        windows = choose_auto_windows(true["cycle_id"], true["I"], args.window_cycles)

    full_a_metrics = basic_metrics(theta_a_true, theta_a_pred)
    full_c_metrics = basic_metrics(theta_c_true, theta_c_pred)

    print("[D15 plot] target profile:", profile_label)
    print("[D15 plot] npz:", npz_path)
    print("[D15 plot] selected cycle windows:", windows)
    print("[D15 plot] full theta_a:", fmt_metric(full_a_metrics))
    print("[D15 plot] full theta_c:", fmt_metric(full_c_metrics))
    print(f"[D15 plot] csmax estimates: a={csmax_a:.6g}, c={csmax_c:.6g}")

    report: List[Dict[str, Any]] = []
    for w in windows:
        report.append(make_surface_figure(
            electrode="a",
            cycle_window=w,
            t=true["t"],
            cycle_id=true["cycle_id"],
            r=true["r_a"],
            Z_plot=Za,
            theta_true=theta_a_true,
            theta_pred=theta_a_pred,
            full_metrics=full_a_metrics,
            args=args,
            profile_label=profile_label,
            out_dir=out_dir,
        ))
        report.append(make_surface_figure(
            electrode="c",
            cycle_window=w,
            t=true["t"],
            cycle_id=true["cycle_id"],
            r=true["r_c"],
            Z_plot=Zc,
            theta_true=theta_c_true,
            theta_pred=theta_c_pred,
            full_metrics=full_c_metrics,
            args=args,
            profile_label=profile_label,
            out_dir=out_dir,
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "D15_BATCH4_BATTERY7_CS3D_WINDOW_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "profile_label": profile_label,
            "npz_path": str(npz_path),
            "surface_source": args.surface_source,
            "plot_unit": args.plot_unit,
            "windows": report,
        }, f, ensure_ascii=False, indent=2)
    print("[D15 plot] wrote window report:", report_path)
    print("[D15 plot] created figures:", len(report))

    if not args.no_show:
        # Keep all six figures open and draggable/rotatable in a GUI backend.
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
