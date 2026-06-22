#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORMAL55-DEPLOY on-demand selected-cycle inference, audit and 3D plotting.

The prediction path is deliberately target-isolated:
1. Load only observable/frozen-baseline arrays for the full profile so that
   target-cycle history is available.
2. Predict all original time points in the requested cycle range.
3. Only after prediction is frozen in memory, stream the selected soft-label
   cs rows for evaluation and plotting.

No full-profile prediction arrays are written.  Selected prediction NPZ output
is opt-in and remains small when a few cycles are requested.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from d18_all55_operational_runtime import (
    discover_deploy_model,
    ensure_dir,
    import_deploy_runtime,
    load_observable_profile,
    now_stamp,
    prepare_parent_runtime,
    read_csv,
    read_json,
    registry_map,
    remap_source_path,
    save_json,
    sha256_file,
    source_uid_compatibility,
    utc_now_iso,
    verify_bundle_manifest,
    verify_parent_hashes,
)

STAGE = "D18-FORMAL55-SELECTED-CYCLE-INFERENCE-PLOT"
PROTOCOL_BY_BATCH = {
    1: "2C",
    2: "3C",
    3: "R2.5",
    4: "R3",
    5: "random_walk",
    6: "GEO",
}
PRIMARY_RADIAL_METRICS = (
    "delta_cs",
    "surface_minus_mean",
    "surface_center_gradient",
    "radial_energy",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FORMAL55 selected-cycle inference, soft-label audit and interactive 3D plotting")
    p.add_argument("--request-json", type=Path, required=False)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--formal-root", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--deploy-model-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--no-show", action="store_true", help="Save plots without opening interactive windows")
    p.add_argument("--self-test", action="store_true")
    return p


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x


def save_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row.keys():
                if str(key) not in fields:
                    fields.append(str(key))
        fieldnames = fields
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def deep_get(d: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_project_root(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.resolve()
    # package is <project>/formal55_selected_cycle_tool/scripts/this_file.py
    return Path(__file__).resolve().parents[2]


def resolve_path(raw: Any, fallback: Path) -> Path:
    text = str(raw or "").strip()
    return Path(text) if text else fallback


def parse_cycle_spec(selection: Mapping[str, Any]) -> List[int]:
    if "cycle_ids" in selection:
        raw = selection["cycle_ids"]
        if not isinstance(raw, list):
            raise ValueError("selection.cycle_ids must be a JSON list")
        cycles = [int(x) for x in raw]
    elif "cycles" in selection:
        raw = selection["cycles"]
        if isinstance(raw, list):
            cycles = [int(x) for x in raw]
        else:
            text = str(raw).strip().replace(" ", "")
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", text)
            if not m:
                raise ValueError("selection.cycles must look like '35-37', '35', or be a list")
            a = int(m.group(1))
            b = int(m.group(2) or a)
            if b < a:
                raise ValueError("cycle range end must be >= start")
            cycles = list(range(a, b + 1))
    elif "cycle_start" in selection and "cycle_end" in selection:
        a, b = int(selection["cycle_start"]), int(selection["cycle_end"])
        if b < a:
            raise ValueError("cycle_end must be >= cycle_start")
        cycles = list(range(a, b + 1))
    else:
        raise ValueError("Specify selection.cycles, selection.cycle_ids, or cycle_start/cycle_end")
    cycles = sorted(set(cycles))
    if not cycles or any(x < 1 for x in cycles):
        raise ValueError("Requested cycle IDs must be positive")
    return cycles


def canonical_uid(batch: int, battery: int, protocol_override: str = "") -> Tuple[str, str]:
    if batch not in PROTOCOL_BY_BATCH:
        raise ValueError(f"batch must be one of {sorted(PROTOCOL_BY_BATCH)}")
    if battery < 1:
        raise ValueError("battery must be >= 1")
    protocol = PROTOCOL_BY_BATCH[batch]
    if protocol_override and str(protocol_override) != protocol:
        raise ValueError(f"Protocol mismatch: batch {batch} implies {protocol}, request says {protocol_override}")
    return f"Batch-{batch}_{protocol}_battery-{battery}", protocol


def read_exact(stream: Any, nbytes: int) -> bytes:
    parts: List[bytes] = []
    remaining = int(nbytes)
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            raise EOFError(f"Unexpected EOF; requested {nbytes}, remaining {remaining}")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def npy_header(stream: Any) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    return tuple(int(x) for x in shape), bool(fortran_order), np.dtype(dtype)


def find_npz_member(zf: zipfile.ZipFile, key: str) -> str:
    wanted = f"{key}.npy"
    matches = [n for n in zf.namelist() if n == wanted or n.endswith("/" + wanted)]
    if len(matches) != 1:
        raise KeyError(f"Expected one NPZ member for {key!r}; found {matches}")
    return matches[0]


def stream_selected_rows(npz_path: Path, key: str, selected_indices: np.ndarray, chunk_rows: int) -> np.ndarray:
    """Read selected target rows sequentially without materializing the full target array."""
    idx = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if np.any(idx[1:] <= idx[:-1]):
        raise ValueError("selected_indices must be strictly increasing")
    with zipfile.ZipFile(npz_path, "r") as zf:
        member = find_npz_member(zf, key)
        with zf.open(member, "r") as stream:
            shape, fortran_order, dtype = npy_header(stream)
            if fortran_order:
                raise ValueError(f"Fortran-order target array unsupported: {key}")
            if len(shape) < 2:
                raise ValueError(f"Target {key} must be at least 2D; shape={shape}")
            nrows = int(shape[0])
            tail_shape = shape[1:]
            row_elems = int(np.prod(tail_shape, dtype=np.int64))
            if int(idx[0]) < 0 or int(idx[-1]) >= nrows:
                raise IndexError(f"Selected indices [{idx[0]},{idx[-1]}] outside {key} rows={nrows}")
            row_bytes = row_elems * dtype.itemsize
            out = np.empty((idx.size, row_elems), dtype=dtype)
            ptr, row0 = 0, 0
            chunk_rows = max(1, int(chunk_rows))
            while row0 < nrows and ptr < idx.size:
                rows = min(chunk_rows, nrows - row0)
                raw = read_exact(stream, rows * row_bytes)
                hi = int(np.searchsorted(idx, row0 + rows, side="left"))
                if hi > ptr:
                    chunk = np.frombuffer(raw, dtype=dtype, count=rows * row_elems).reshape(rows, row_elems)
                    local = idx[ptr:hi] - row0
                    out[ptr:hi] = chunk[local]
                    ptr = hi
                row0 += rows
            if ptr != idx.size:
                raise EOFError(f"Recovered {ptr}/{idx.size} selected rows for {key}")
    return out.reshape((idx.size,) + tail_shape).astype(np.float32, copy=False)


def weighted_zero_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mean = np.sum(a.astype(np.float64) * w[None, :], axis=1, keepdims=True)
    return a - mean.astype(np.float32)


def safe_corr(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size < 2 or np.std(y) <= 1e-15 or np.std(p) <= 1e-15:
        return None
    value = float(np.corrcoef(y, p)[0, 1])
    return value if np.isfinite(value) else None


def metric_stats(target: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size == 0:
        return {"n": 0, "r2": None, "mae": None, "rmse": None, "nmae": None, "nrmse": None, "bias": None, "corr": None, "target_range": None, "target_std": None, "max_abs": None}
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    max_abs = float(np.max(np.abs(err)))
    target_range = float(np.max(y) - np.min(y))
    target_std = float(np.std(y))
    denom = max(target_range, 1e-12)
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = None if sst <= 1e-24 else float(1.0 - np.sum(err ** 2) / sst)
    return {
        "n": int(y.size),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / denom,
        "nrmse": rmse / denom,
        "bias": bias,
        "corr": safe_corr(y, p),
        "target_range": target_range,
        "target_std": target_std,
        "max_abs": max_abs,
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
    }


def radial_features(dev: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    d = np.asarray(dev, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "delta_cs": d,
        "surface_minus_mean": d[:, -1],
        "surface_center_gradient": d[:, -1] - d[:, 0],
        "radial_energy": np.sqrt(np.sum(d.astype(np.float64) ** 2 * w[None, :], axis=1)),
    }


def predict_selected(
    uid: str,
    model_root: Path,
    adapter_row: Mapping[str, str],
    parent_runtime: Mapping[str, Any],
    modules: Mapping[str, Any],
    idx: np.ndarray,
    profile: Mapping[str, Any],
    inference_batch_size: int,
    radial_margin_fraction: float,
) -> Dict[str, Any]:
    """Prediction-only function. No target arrays are accepted."""
    core, step4, step5, adapter_rt = modules["core"], modules["step4"], modules["step5"], modules["adapter"]
    protocol = str(adapter_row["protocol"])
    branch = str(adapter_row["branch"])
    batch = uid.split("_")[0]
    r_a, r_c = np.asarray(profile["r_a"]), np.asarray(profile["r_c"])
    nr_a, nr_c = int(r_a.size), int(r_c.size)
    semantic = parent_runtime["semantic"]
    old_w_a = np.asarray(semantic[(branch, "anode")].weights, dtype=np.float64)
    old_w_c = np.asarray(semantic[(branch, "cathode")].weights, dtype=np.float64)
    can_w_a = step4.canonical_weights(nr_a, r_a)
    can_w_c = step4.canonical_weights(nr_c, r_c)
    calibration = parent_runtime["calibrations"][protocol]
    route_name, route_kind = step5.parent_route(protocol)
    candidates = step4.base_and_seed_candidates(
        idx=idx,
        n=int(profile["n"]),
        signals=profile["signals"],
        cycle_ids=profile["cycle_ids"],
        cbar_a=profile["cbar_a"],
        cbar_c=profile["cbar_c"],
        protocol=protocol,
        branch=branch,
        batch=batch,
        old_w_a=old_w_a,
        old_w_c=old_w_c,
        can_w_a=can_w_a,
        can_w_c=can_w_c,
        calibration=calibration,
        step3fix_decisions=parent_runtime["decisions"],
        f64_models=parent_runtime["f64_models"],
        old_specialists=parent_runtime["specialists"],
        step34_runtimes=parent_runtime["step34_runtimes"],
        route_name=route_name,
        lags=parent_runtime["lags"],
        inference_batch_size=int(inference_batch_size),
        radial_margin_fraction=float(radial_margin_fraction),
        nr_a=nr_a,
        nr_c=nr_c,
    )
    use_learned = route_kind == "learned" and bool(candidates["learned_available"])
    parent_a = np.asarray(candidates["ensemble_a"] if use_learned else candidates["base_a"], dtype=np.float32)
    parent_c = np.asarray(candidates["ensemble_c"] if use_learned else candidates["base_c"], dtype=np.float32)

    ranges = step5.contiguous_ranges(profile["cycle_ids"])
    phase, age, _ = step5.cycle_phase_and_age(profile["cycle_ids"], ranges)
    q_signed, q_abs, di_norm = step5.cumulative_features(profile["signals"])
    scalar = step5.build_scalar_features(idx, profile["signals"], phase, age, q_signed, q_abs, di_norm)

    adapter_path = model_root / Path(str(adapter_row["adapter_relative_path"]))
    expected_hash = str(adapter_row["adapter_sha256"]).strip().lower()
    actual_hash = sha256_file(adapter_path).lower()
    if actual_hash != expected_hash:
        raise RuntimeError(f"Adapter hash mismatch for {uid}")
    adapter = adapter_rt.load_cell_adapter(adapter_path)
    if adapter.uid != uid:
        raise ValueError(f"Adapter UID mismatch: expected={uid}, observed={adapter.uid}")
    dev_a = adapter.anode.apply(parent_a, scalar, can_w_a)
    dev_c = adapter.cathode.apply(parent_c, scalar, can_w_c)
    dev_a, scale_a = core.cap_radial_dev(
        dev_a,
        profile["cbar_a"][idx],
        float(calibration["anode"]["csmax"]),
        float(calibration["anode"]["radial_q995_theta"]),
        radial_margin_fraction,
        can_w_a,
    )
    dev_c, scale_c = core.cap_radial_dev(
        dev_c,
        profile["cbar_c"][idx],
        float(calibration["cathode"]["csmax"]),
        float(calibration["cathode"]["radial_q995_theta"]),
        radial_margin_fraction,
        can_w_c,
    )
    dev_a = np.asarray(dev_a, dtype=np.float32)
    dev_c = np.asarray(dev_c, dtype=np.float32)
    cbar_a = np.asarray(profile["cbar_a"])[idx].astype(np.float32, copy=False)
    cbar_c = np.asarray(profile["cbar_c"])[idx].astype(np.float32, copy=False)
    cs_a = cbar_a[:, None] + dev_a
    cs_c = cbar_c[:, None] + dev_c
    csmax_a = float(calibration["anode"]["csmax"])
    csmax_c = float(calibration["cathode"]["csmax"])
    return {
        "dev_a": dev_a,
        "dev_c": dev_c,
        "cs_a": cs_a,
        "cs_c": cs_c,
        "theta_a": cs_a / np.float32(csmax_a),
        "theta_c": cs_c / np.float32(csmax_c),
        "cbar_a": cbar_a,
        "cbar_c": cbar_c,
        "weights_a": np.asarray(can_w_a, dtype=np.float64),
        "weights_c": np.asarray(can_w_c, dtype=np.float64),
        "csmax_a": csmax_a,
        "csmax_c": csmax_c,
        "parent_route": f"{route_name}:{'learned_ensemble' if use_learned else 'frozen_base'}",
        "anode_adapter_route": adapter.anode.selected_route,
        "cathode_adapter_route": adapter.cathode.selected_route,
        "cap_scale_anode_min": float(np.min(scale_a)),
        "cap_scale_cathode_min": float(np.min(scale_c)),
        "adapter_sha256": actual_hash,
    }


def cycle_ranges(cycle_ids: np.ndarray, step5: Any) -> List[Tuple[int, int, int]]:
    return [(int(cid), int(start), int(stop)) for cid, start, stop in step5.contiguous_ranges(cycle_ids)]


def selected_indices_for_cycles(ranges: Sequence[Tuple[int, int, int]], requested: Sequence[int]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    lookup = {cid: (start, stop) for cid, start, stop in ranges}
    missing = [cid for cid in requested if cid not in lookup]
    if missing:
        available = sorted(lookup)
        raise KeyError(f"Requested cycle IDs not found: {missing}. Available range={available[0]}..{available[-1]}, count={len(available)}")
    parts: List[np.ndarray] = []
    ledger: List[Dict[str, Any]] = []
    for cid in requested:
        start, stop = lookup[cid]
        idx = np.arange(start, stop, dtype=np.int64)
        parts.append(idx)
        ledger.append({
            "cycle_id": int(cid),
            "start_idx": int(start),
            "stop_idx_exclusive": int(stop),
            "point_count": int(stop - start),
        })
    return np.concatenate(parts), ledger


def downsample_plot_indices(cycle_ids: np.ndarray, max_points: int) -> np.ndarray:
    n = int(cycle_ids.size)
    if n <= max_points or max_points <= 0:
        return np.arange(n, dtype=np.int64)
    base = np.linspace(0, n - 1, max_points, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[True, cycle_ids[1:] != cycle_ids[:-1]])
    ends = np.r_[boundaries[1:] - 1, n - 1]
    idx = np.unique(np.concatenate([base, boundaries, ends]))
    if idx.size > max_points:
        # Keep every boundary, thin only the non-boundary points.
        fixed = np.unique(np.concatenate([boundaries, ends]))
        remaining = max(0, max_points - fixed.size)
        candidates = np.setdiff1d(idx, fixed, assume_unique=True)
        if remaining and candidates.size:
            pick = candidates[np.linspace(0, candidates.size - 1, min(remaining, candidates.size), dtype=np.int64)]
            idx = np.unique(np.concatenate([fixed, pick]))
        else:
            idx = fixed
    return idx.astype(np.int64)


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except Exception:
        return "NA"
    return f"{v:.{digits}f}" if np.isfinite(v) else "NA"


def plot_surface_pair(
    *,
    uid: str,
    cycles: Sequence[int],
    electrode: str,
    time_values: np.ndarray,
    radial_values: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    metrics: Mapping[str, Any],
    cycle_ids: np.ndarray,
    plot_cfg: Mapping[str, Any],
    plots_dir: Path,
    show_interactive: bool,
) -> Tuple[List[Path], List[Any]]:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    font = str(plot_cfg.get("font_family", "Times New Roman"))
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    max_time_points = int(plot_cfg.get("max_time_points", 800))
    ds = downsample_plot_indices(cycle_ids, max_time_points)
    t = np.asarray(time_values, dtype=np.float64)[ds]
    cyc = np.asarray(cycle_ids)[ds]
    p = np.asarray(pred)[ds]
    y = np.asarray(truth)[ds]
    r = np.asarray(radial_values, dtype=np.float64).reshape(-1)
    radial_mode = str(plot_cfg.get("radial_axis", "physical")).lower()
    if radial_mode in {"normalized", "r/r", "r_over_r"}:
        span = float(np.max(r) - np.min(r))
        r_plot = (r - np.min(r)) / span if span > 0 else np.linspace(0.0, 1.0, r.size)
        ylabel = "r/R"
    else:
        r_plot = r
        ylabel = "r (m)"

    vmin = float(min(np.min(p), np.min(y)))
    vmax = float(max(np.max(p), np.max(y)))
    pad = max((vmax - vmin) * 0.02, 1e-8)
    zmin, zmax = vmin - pad, vmax + pad
    norm = Normalize(vmin=vmin, vmax=vmax)
    t_mesh, r_mesh = np.meshgrid(t, r_plot)
    cycles_text = f"{min(cycles)}–{max(cycles)}" if len(cycles) > 1 else str(cycles[0])
    electrode_title = "Anode" if electrode == "anode" else "Cathode"
    symbol = "C_s,a" if electrode == "anode" else "C_s,c"
    metric_line = (
        f"R$^2$={format_metric(metrics.get('cs_r2'))} | "
        f"NMAE={format_metric(metrics.get('cs_nmae'))} | "
        f"NRMSE={format_metric(metrics.get('cs_nrmse'))} | "
        f"$\\delta c_s$ R$^2$={format_metric(metrics.get('delta_cs_r2'))}"
    )
    saved: List[Path] = []
    figures: List[Any] = []
    for kind, values, cmap in [
        ("Prediction", p, str(plot_cfg.get("prediction_cmap", "coolwarm"))),
        ("Soft-label reference", y, str(plot_cfg.get("truth_cmap", "viridis"))),
    ]:
        fig = plt.figure(figsize=(float(plot_cfg.get("figure_width_in", 11.5)), float(plot_cfg.get("figure_height_in", 7.5))))
        figures.append(fig)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(t_mesh, r_mesh, values.T, cmap=cmap, norm=norm, linewidth=0, antialiased=True, rcount=min(r.size, 80), ccount=min(t.size, max_time_points))
        ax.set_xlabel("Global t (s)" if str(plot_cfg.get("time_axis", "global")).lower() == "global" else "Selected-segment t (s)", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_zlabel(f"{symbol} (kmol/m$^3$)", labelpad=9)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=float(plot_cfg.get("elevation_deg", 24)), azim=float(plot_cfg.get("azimuth_deg", -58)))
        title = f"{electrode_title} {symbol}(t,r) {kind} Surface\n{uid} | cycles {cycles_text}\n{metric_line}"
        ax.set_title(title, pad=18)
        # Mark cycle centers and boundaries using the same selected time axis.
        unique_c = []
        for c in cyc.tolist():
            if not unique_c or unique_c[-1] != int(c):
                unique_c.append(int(c))
        for cid in unique_c:
            mask = cyc == cid
            if np.any(mask):
                tm = float(np.median(t[mask]))
                ax.text(tm, float(r_plot[-1]), zmin + 0.70 * (zmax - zmin), f"cycle {cid}", fontsize=8)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.60, pad=0.10)
        try:
            fig.canvas.manager.set_window_title(f"{uid} | {electrode_title} | {kind}")
        except Exception:
            pass
        fig.tight_layout()
        if bool(plot_cfg.get("save_png", True)):
            filename = f"{uid}_cycles_{min(cycles)}_{max(cycles)}_{electrode}_{'prediction' if kind == 'Prediction' else 'truth'}.png"
            path = plots_dir / filename
            fig.savefig(path, dpi=int(plot_cfg.get("dpi", 160)), bbox_inches="tight")
            saved.append(path)
    return saved, figures


def build_metrics(
    *,
    cycles: Sequence[int],
    selected_cycle_ids: np.ndarray,
    pred: Mapping[str, Any],
    target_cs_a: np.ndarray,
    target_cs_c: np.ndarray,
    exact_r2_threshold: float,
    exact_nmae_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_dev_a = weighted_zero_mean(target_cs_a, pred["weights_a"])
    target_dev_c = weighted_zero_mean(target_cs_c, pred["weights_c"])
    targets = {
        "anode": {
            "cs": target_cs_a,
            "theta": target_cs_a / np.float32(pred["csmax_a"]),
            "dev": target_dev_a,
            "weights": pred["weights_a"],
            "csmax": pred["csmax_a"],
        },
        "cathode": {
            "cs": target_cs_c,
            "theta": target_cs_c / np.float32(pred["csmax_c"]),
            "dev": target_dev_c,
            "weights": pred["weights_c"],
            "csmax": pred["csmax_c"],
        },
    }
    preds = {
        "anode": {"cs": pred["cs_a"], "theta": pred["theta_a"], "dev": pred["dev_a"]},
        "cathode": {"cs": pred["cs_c"], "theta": pred["theta_c"], "dev": pred["dev_c"]},
    }
    global_metrics: Dict[str, Any] = {}
    global_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    suspicious: List[Dict[str, Any]] = []

    def add(scope: str, cycle_id: Optional[int], electrode: str, metric_name: str, target: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
        stats = metric_stats(target, prediction)
        row = {"scope": scope, "cycle_id": "" if cycle_id is None else int(cycle_id), "electrode": electrode, "metric": metric_name, **stats}
        if scope == "global":
            global_rows.append(row)
        else:
            cycle_rows.append(row)
        if stats.get("r2") is not None and float(stats["r2"]) >= exact_r2_threshold and float(stats.get("nmae") or 0.0) <= exact_nmae_threshold:
            suspicious.append({**row, "reason": "R2_near_1_and_NMAE_near_0"})
        return stats

    for electrode in ("anode", "cathode"):
        t = targets[electrode]
        p = preds[electrode]
        cs_stats = add("global", None, electrode, "cs", t["cs"], p["cs"])
        theta_stats = add("global", None, electrode, "theta", t["theta"], p["theta"])
        tf = radial_features(t["dev"], t["weights"])
        pf = radial_features(p["dev"], t["weights"])
        radial_stats: Dict[str, Any] = {}
        for name in PRIMARY_RADIAL_METRICS:
            radial_stats[name] = add("global", None, electrode, name, tf[name], pf[name])
        global_metrics[electrode] = {
            "cs": cs_stats,
            "theta": theta_stats,
            "radial": radial_stats,
            "plot_title_metrics": {
                "cs_r2": cs_stats.get("r2"),
                "cs_nmae": cs_stats.get("nmae"),
                "cs_nrmse": cs_stats.get("nrmse"),
                "delta_cs_r2": radial_stats["delta_cs"].get("r2"),
            },
        }
        for cid in cycles:
            mask = selected_cycle_ids == cid
            ctf = radial_features(t["dev"][mask], t["weights"])
            cpf = radial_features(p["dev"][mask], t["weights"])
            add("cycle", cid, electrode, "cs", t["cs"][mask], p["cs"][mask])
            add("cycle", cid, electrode, "theta", t["theta"][mask], p["theta"][mask])
            for name in PRIMARY_RADIAL_METRICS:
                add("cycle", cid, electrode, name, ctf[name], cpf[name])
    return global_metrics, global_rows + cycle_rows, suspicious


def self_test() -> Dict[str, Any]:
    request = {"cycles": "13-15"}
    parsed = parse_cycle_spec(request)
    rng = np.random.default_rng(42)
    target = rng.normal(size=(64, 17)).astype(np.float32)
    pred = target + rng.normal(scale=0.02, size=target.shape).astype(np.float32)
    m = metric_stats(target, pred)
    exact = metric_stats(target, target.copy())
    idx = downsample_plot_indices(np.repeat([13, 14, 15], [30, 40, 50]), 35)
    ok = parsed == [13, 14, 15] and m["r2"] is not None and m["r2"] < 1.0 and exact["r2"] == 1.0 and idx.size <= 35
    return {
        "self_test": "PASS" if ok else "FAIL",
        "cycle_parser": parsed,
        "non_exact_r2": m.get("r2"),
        "exact_r2_detection_input": exact.get("r2"),
        "plot_downsample_points": int(idx.size),
    }


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["self_test"] == "PASS" else 1
    if args.request_json is None:
        parser().error("--request-json is required unless --self-test is used")

    run_start = time.monotonic()
    request_path = args.request_json.resolve()
    request = read_json(request_path)
    project_root = resolve_project_root(args.project_root)
    formal_root = resolve_path(args.formal_root or deep_get(request, ["paths", "formal_root"]), Path(r"C:\Users\Tiga_QJW\Desktop\XJTUstation\D18\Formal-A"))
    cache_root = resolve_path(args.cache_root or deep_get(request, ["paths", "cache_root"]), Path(r"E:\XJTU battery dataset\_gv1_cache"))
    default_output = project_root / "formal55_selected_cycle_outputs"
    output_root = resolve_path(args.output_root or deep_get(request, ["paths", "output_root"]), default_output)
    deploy_raw = args.deploy_model_root or deep_get(request, ["paths", "deploy_model_root"])
    deploy_explicit = Path(str(deploy_raw)) if str(deploy_raw or "").strip() else None
    request_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(request.get("request_name", "selected_cycle_request"))).strip("_") or "selected_cycle_request"
    run_dir = output_root / f"{request_name}_{now_stamp()}"
    ensure_dir(run_dir)
    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    failures: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {"stage": STAGE, "status": "FAIL", "created_at": utc_now_iso()}
    try:
        selection = request.get("selection", {})
        batch = int(selection.get("batch"))
        battery = int(selection.get("battery"))
        cycles = parse_cycle_spec(selection)
        uid, protocol = canonical_uid(batch, battery, str(selection.get("protocol", "")))

        model_root, deploy_manifest = discover_deploy_model(formal_root, deploy_explicit)
        modules = import_deploy_runtime(model_root)
        device = str(deep_get(request, ["runtime", "device"], "cuda"))
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("CUDA requested but unavailable; deploy runtime will use CPU where applicable")
                    device = "cpu"
            except Exception:
                warnings.append("CUDA requested but torch import failed; using CPU")
                device = "cpu"
        parent_runtime = prepare_parent_runtime(model_root, device, modules)

        adapter_rows = registry_map(read_csv(model_root / "manifests" / "adapter_registry.csv"))
        source_rows = registry_map(read_csv(model_root / "manifests" / "source_registry.csv"))
        confidence_rows = registry_map(read_csv(model_root / "manifests" / "confidence_registry.csv"))
        if uid not in adapter_rows or uid not in source_rows:
            raise KeyError(f"Canonical UID not present in deploy registries: {uid}")
        adapter_row = adapter_rows[uid]
        source_row = source_rows[uid]
        confidence_row = confidence_rows.get(uid, {})
        source_path = remap_source_path(source_row["source_path_external"], cache_root)

        verify_hashes = bool(deep_get(request, ["runtime", "verify_bundle_hashes"], True))
        bundle_hash_rows: List[Dict[str, Any]] = []
        parent_hash_rows: List[Dict[str, Any]] = []
        if verify_hashes:
            bundle_hash_rows, bundle_failures, bundle_warnings = verify_bundle_manifest(model_root)
            parent_hash_rows, parent_failures = verify_parent_hashes(model_root)
            failures.extend(bundle_failures)
            failures.extend(parent_failures)
            warnings.extend(bundle_warnings)
            if failures:
                raise RuntimeError("Deploy bundle hash verification failed")

        # Observable/frozen-baseline load only. No target cs/theta arrays are materialized here.
        profile = load_observable_profile(source_path, modules["core"])
        source_uid_ok, source_uid_mode, source_uid_detail = source_uid_compatibility(uid, str(profile.get("source_uid", "")), source_path)
        if not source_uid_ok:
            raise RuntimeError(f"Source UID mismatch for {uid}: {source_uid_mode}; detail={source_uid_detail}")
        ranges = cycle_ranges(profile["cycle_ids"], modules["step5"])
        idx, cycle_ledger = selected_indices_for_cycles(ranges, cycles)
        max_selected = int(deep_get(request, ["runtime", "max_selected_points"], 250000))
        allow_large = bool(deep_get(request, ["runtime", "allow_large_selection"], False))
        if idx.size > max_selected and not allow_large:
            raise RuntimeError(f"Selected {idx.size} points, exceeding max_selected_points={max_selected}. Narrow the cycle range or set allow_large_selection=true.")

        history_points = int(idx[0])
        history_cycles = sum(stop <= int(idx[0]) for _, _, stop in ranges)
        inference_batch_size = int(deep_get(request, ["runtime", "inference_batch_size"], 8192))
        radial_margin_fraction = float(deep_get(request, ["runtime", "radial_margin_fraction"], 0.95))

        # Prediction is completed and frozen in memory before target arrays are read.
        prediction_started = time.monotonic()
        pred = predict_selected(uid, model_root, adapter_row, parent_runtime, modules, idx, profile, inference_batch_size, radial_margin_fraction)
        prediction_seconds = time.monotonic() - prediction_started

        target_chunk_rows = int(deep_get(request, ["runtime", "target_stream_chunk_rows"], 32768))
        target_started = time.monotonic()
        target_cs_a = stream_selected_rows(source_path, "cs_a", idx, target_chunk_rows)
        target_cs_c = stream_selected_rows(source_path, "cs_c", idx, target_chunk_rows)
        target_read_seconds = time.monotonic() - target_started
        if target_cs_a.shape != pred["cs_a"].shape or target_cs_c.shape != pred["cs_c"].shape:
            raise ValueError(f"Target/prediction shape mismatch: a={target_cs_a.shape}/{pred['cs_a'].shape}, c={target_cs_c.shape}/{pred['cs_c'].shape}")

        selected_cycle_ids = np.asarray(profile["cycle_ids"])[idx]
        exact_r2 = float(deep_get(request, ["evaluation", "exact_match_r2_threshold"], 0.999999))
        exact_nmae = float(deep_get(request, ["evaluation", "exact_match_nmae_threshold"], 1e-8))
        metrics, metric_rows, suspicious = build_metrics(
            cycles=cycles,
            selected_cycle_ids=selected_cycle_ids,
            pred=pred,
            target_cs_a=target_cs_a,
            target_cs_c=target_cs_c,
            exact_r2_threshold=exact_r2,
            exact_nmae_threshold=exact_nmae,
        )
        if suspicious:
            failures.append(f"Suspicious near-exact learned metrics detected: {len(suspicious)}")

        zero_a = float(np.max(np.abs(np.sum(pred["dev_a"].astype(np.float64) * pred["weights_a"][None, :], axis=1))) / max(pred["csmax_a"], 1e-12))
        zero_c = float(np.max(np.abs(np.sum(pred["dev_c"].astype(np.float64) * pred["weights_c"][None, :], axis=1))) / max(pred["csmax_c"], 1e-12))
        target_cbar_a = np.sum(target_cs_a.astype(np.float64) * pred["weights_a"][None, :], axis=1)
        target_cbar_c = np.sum(target_cs_c.astype(np.float64) * pred["weights_c"][None, :], axis=1)
        cbar_gap_a = float(np.max(np.abs(target_cbar_a - pred["cbar_a"])) / max(pred["csmax_a"], 1e-12))
        cbar_gap_c = float(np.max(np.abs(target_cbar_c - pred["cbar_c"])) / max(pred["csmax_c"], 1e-12))
        finite_ok = all(np.all(np.isfinite(x)) for x in [pred["cs_a"], pred["cs_c"], pred["theta_a"], pred["theta_c"], target_cs_a, target_cs_c])
        theta_min = float(min(np.min(pred["theta_a"]), np.min(pred["theta_c"])))
        theta_max = float(max(np.max(pred["theta_a"]), np.max(pred["theta_c"])))
        theta_ok = theta_min >= -1e-5 and theta_max <= 1.0 + 1e-5
        if not finite_ok:
            failures.append("Non-finite prediction or target values")
        if not theta_ok:
            failures.append(f"Predicted theta outside [0,1]: [{theta_min},{theta_max}]")
        if zero_a > 1e-5 or zero_c > 1e-5:
            failures.append(f"Zero-mean radial residual check failed: a={zero_a}, c={zero_c}")

        time_global = np.asarray(profile["signals"]["t_global_s"])[idx]
        time_axis = str(deep_get(request, ["plot", "time_axis"], "global")).lower()
        time_plot = time_global - time_global[0] if time_axis in {"local", "selection_local", "selected"} else time_global
        plot_cfg = request.get("plot", {})
        show_interactive = bool(plot_cfg.get("show_interactive", True)) and not args.no_show
        saved_plots: List[Path] = []
        plot_figures: List[Any] = []
        if bool(plot_cfg.get("enabled", True)):
            import matplotlib
            if not show_interactive:
                matplotlib.use("Agg", force=True)
            saved, figures = plot_surface_pair(
                uid=uid,
                cycles=cycles,
                electrode="anode",
                time_values=time_plot,
                radial_values=profile["r_a"],
                pred=pred["cs_a"],
                truth=target_cs_a,
                metrics=metrics["anode"]["plot_title_metrics"],
                cycle_ids=selected_cycle_ids,
                plot_cfg=plot_cfg,
                plots_dir=plots_dir,
                show_interactive=show_interactive,
            )
            saved_plots.extend(saved)
            plot_figures.extend(figures)
            saved, figures = plot_surface_pair(
                uid=uid,
                cycles=cycles,
                electrode="cathode",
                time_values=time_plot,
                radial_values=profile["r_c"],
                pred=pred["cs_c"],
                truth=target_cs_c,
                metrics=metrics["cathode"]["plot_title_metrics"],
                cycle_ids=selected_cycle_ids,
                plot_cfg=plot_cfg,
                plots_dir=plots_dir,
                show_interactive=show_interactive,
            )
            saved_plots.extend(saved)
            plot_figures.extend(figures)

        write_csv(run_dir / "selected_cycle_index_ledger.csv", cycle_ledger)
        write_csv(run_dir / "metrics_global_and_by_cycle.csv", metric_rows)
        write_csv(run_dir / "suspicious_exact_metrics.csv", suspicious, fieldnames=["scope", "cycle_id", "electrode", "metric", "n", "r2", "mae", "rmse", "nmae", "nrmse", "bias", "corr", "target_range", "target_std", "max_abs", "reason"])
        if verify_hashes:
            write_csv(run_dir / "bundle_hash_audit.csv", bundle_hash_rows)
            write_csv(run_dir / "parent_hash_audit.csv", parent_hash_rows)

        output_cfg = request.get("output", {})
        selected_npz_path = ""
        if bool(output_cfg.get("save_selected_npz", False)):
            selected_npz = run_dir / f"{uid}_cycles_{min(cycles)}_{max(cycles)}_selected.npz"
            payload: Dict[str, Any] = {
                "source_index": idx.astype(np.int64),
                "t_global_s": time_global.astype(np.float64),
                "cycle_id": selected_cycle_ids.astype(np.int64),
                "I_profile": np.asarray(profile["signals"]["I_profile"])[idx],
                "voltage_exp": np.asarray(profile["signals"]["voltage_exp"])[idx],
                "temperature_C": np.asarray(profile["signals"]["temperature_C"])[idx],
                "r_a": np.asarray(profile["r_a"]),
                "r_c": np.asarray(profile["r_c"]),
                "cbar_a": pred["cbar_a"],
                "cbar_c": pred["cbar_c"],
                "cs_a_pred": pred["cs_a"],
                "cs_c_pred": pred["cs_c"],
                "theta_a_pred": pred["theta_a"],
                "theta_c_pred": pred["theta_c"],
                "phie_reference": np.asarray(profile["phie"])[idx],
                "phis_c_reference": np.asarray(profile["phis_c"])[idx],
                "metadata_json": np.asarray(json.dumps({"uid": uid, "cycles": cycles, "scientific_boundary": "Step2/P2Dlite-RG-assisted closed-set model-consistent emulation"}, ensure_ascii=False)),
            }
            if bool(output_cfg.get("save_truth_in_npz", False)):
                payload.update({"cs_a_truth": target_cs_a, "cs_c_truth": target_cs_c})
            np.savez_compressed(selected_npz, **payload)
            selected_npz_path = str(selected_npz)

        resolved_request = {
            "request_path": str(request_path),
            "request_name": request_name,
            "project_root": str(project_root),
            "formal_root": str(formal_root),
            "cache_root": str(cache_root),
            "deploy_model_root": str(model_root),
            "deploy_ready_manifest": str(deploy_manifest),
            "canonical_uid": uid,
            "protocol": protocol,
            "branch": adapter_row.get("branch", ""),
            "confidence": confidence_row.get("confidence", ""),
            "cycles": cycles,
            "source_path": str(source_path),
            "source_uid_match_mode": source_uid_mode,
            "selected_point_count": int(idx.size),
            "history_points_before_selection": history_points,
            "history_cycles_before_selection": history_cycles,
            "history_policy": "full observable/frozen-baseline profile loaded; cumulative features computed from profile start; only requested cycles emitted",
            "target_policy": "target cs arrays streamed only after prediction completed",
        }
        save_json_atomic(run_dir / "request_resolved.json", resolved_request)
        save_json_atomic(run_dir / "metrics_global.json", metrics)
        save_json_atomic(run_dir / "physical_and_leakage_audit.json", {
            "finite_ok": finite_ok,
            "theta_bounds_ok": theta_ok,
            "theta_min": theta_min,
            "theta_max": theta_max,
            "prediction_zero_mean_anode": zero_a,
            "prediction_zero_mean_cathode": zero_c,
            "target_cbar_vs_frozen_cbar_max_gap_anode_theta": cbar_gap_a,
            "target_cbar_vs_frozen_cbar_max_gap_cathode_theta": cbar_gap_c,
            "target_state_arrays_present_in_source": profile.get("target_state_arrays_present", []),
            "target_state_arrays_loaded_before_prediction": False,
            "target_state_arrays_loaded_after_prediction": True,
            "suspicious_exact_metric_count": len(suspicious),
            "phie_phis_c_policy": "reference-only; no predictive R2 reported",
        })

        summary.update({
            "status": "PASS" if not failures else "FAIL",
            "request_name": request_name,
            "canonical_uid": uid,
            "protocol": protocol,
            "confidence": confidence_row.get("confidence", ""),
            "cycles": cycles,
            "selected_point_count": int(idx.size),
            "history_points_before_selection": history_points,
            "prediction_seconds": prediction_seconds,
            "target_stream_seconds": target_read_seconds,
            "elapsed_seconds_after_plot_generation": time.monotonic() - run_start,
            "parent_route": pred["parent_route"],
            "anode_adapter_route": pred["anode_adapter_route"],
            "cathode_adapter_route": pred["cathode_adapter_route"],
            "metrics": metrics,
            "suspicious_exact_metric_count": len(suspicious),
            "saved_plot_files": [str(p) for p in saved_plots],
            "selected_npz": selected_npz_path,
            "scientific_boundary": "Step2/P2Dlite-RG-assisted closed-set model-consistent state emulation; soft labels are reference, not experimental truth",
            "failures": failures,
            "warnings": warnings,
        })
        # Persist a complete status before opening blocking GUI windows.
        save_json_atomic(run_dir / "RUN_STATUS.json", summary)
        if show_interactive and plot_figures:
            import matplotlib.pyplot as plt
            plt.show(block=True)
        elif plot_figures:
            import matplotlib.pyplot as plt
            plt.close("all")
    except Exception as exc:
        failures.append(f"Unhandled exception: {exc!r}")
        summary.update({
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
            "traceback": traceback.format_exc(),
        })
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        summary["elapsed_seconds_total"] = time.monotonic() - run_start
        save_json_atomic(run_dir / "RUN_STATUS.json", summary)
        print(f"[FORMAL55 selected-cycle] status: {summary.get('status')}")
        print(f"[FORMAL55 selected-cycle] output: {run_dir}")
        if failures:
            print("[FORMAL55 selected-cycle] failures:")
            for failure in failures:
                print(f"  - {failure}")
    gc.collect()
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
