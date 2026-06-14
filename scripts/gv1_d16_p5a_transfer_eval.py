#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D16-P5A ALL55 existing-model transfer evaluation for QJW-2 / PINN-for-ASSB-V1.

Purpose
-------
Evaluate existing D15 P2Dlite-RG neural-network checkpoints or existing prediction.npz
outputs against the D15 ALL55 P2Dlite-RG soft-label directory, without training any new
model and without modifying legacy ASSB / D9 / D12 / D15 artifacts.

Recommended D16-P5A modes
-------------------------
1) preflight-only:
   Check ALL55 soft labels and output a cell manifest.

2) evaluate existing predictions:
   Read prediction.npz files that were produced by existing D15-P2 / Batch-2 / P3C models
   and produce cell-wise, batch-wise, seen/unseen, raw/projected scorecards.

3) generate predictions then evaluate:
   Use --inference-command-template to call an existing project inference script once per
   cell. This package intentionally does not assume a private model architecture; instead
   it orchestrates your current inference entrypoint and audits its outputs.

Important boundary
------------------
These soft labels are P2Dlite-RG model-consistent labels, not experimentally measured
internal-state truth. This script evaluates transfer to those labels only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np

try:
    import pandas as pd  # optional but convenient
except Exception:  # pragma: no cover
    pd = None


TARGET_FILENAME_CANDIDATES = (
    "solution_softlabels.npz",
    "solution_softlabel.npz",
    "solution.npz",
)
PREDICTION_FILENAME_CANDIDATES = (
    "prediction.npz",
    "predictions.npz",
    "projected_prediction.npz",
    "raw_prediction.npz",
)
DEFAULT_METRICS = ("phis_c", "phie", "theta_a", "theta_c", "cs_a", "cs_c")
DEFAULT_REQUIRED_LABEL_KEYS = ("t_global_s", "I_profile")

# Conservative default thresholds for transfer evaluation.
# Use these for pass/review triage only; exact publication thresholds should be frozen
# after you inspect D16-P5A outputs.
DEFAULT_THRESHOLDS = {
    "phis_c": {"pass_mae": 0.010, "review_mae": 0.020, "pass_corr": 0.995, "review_corr": 0.990},
    "phie": {"pass_mae": 0.010, "review_mae": 0.020, "pass_corr": 0.995, "review_corr": 0.990},
    "theta_a": {"pass_mae": 0.005, "review_mae": 0.015, "pass_corr": 0.995, "review_corr": 0.990},
    "theta_c": {"pass_mae": 0.005, "review_mae": 0.015, "pass_corr": 0.995, "review_corr": 0.990},
    # cs scale varies across implementations; these are used as soft review gates.
    "cs_a": {"pass_nmae": 0.010, "review_nmae": 0.030, "pass_corr": 0.995, "review_corr": 0.990},
    "cs_c": {"pass_nmae": 0.010, "review_nmae": 0.030, "pass_corr": 0.995, "review_corr": 0.990},
    "grad_cs_a": {"pass_mae": 0.010, "review_mae": 0.030, "pass_corr": 0.990, "review_corr": 0.950},
    "grad_cs_c": {"pass_mae": 0.010, "review_mae": 0.030, "pass_corr": 0.990, "review_corr": 0.950},
}


@dataclass
class CellRecord:
    cell_uid: str
    batch: str
    protocol: str
    soft_label_npz: str
    cell_dir: str
    source_stage: str = "unknown"
    is_seen: str = "unknown"
    flagged: bool = False
    note: str = ""


@dataclass
class FileCheck:
    ok: bool
    path: str
    message: str


@dataclass
class MetricRecord:
    model_name: str
    cell_uid: str
    batch: str
    protocol: str
    is_seen: str
    flagged: bool
    projection_mode: str
    metric: str
    n: int
    true_shape: str
    pred_shape: str
    mae: float
    rmse: float
    max_abs: float
    bias: float
    corr: float
    r2: float
    nmae_range: float
    status: str
    message: str


# ----------------------------- small utilities -----------------------------


def as_posixish(path: Path) -> str:
    return str(path).replace("\\", "/")


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json_maybe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return {}


def parse_batch_protocol_cell_from_name(name: str) -> Tuple[str, str, str]:
    # Examples seen in project history can vary:
    # Batch-1_2C_battery-1, 0023_battery-7_R3_battery-7, Batch-4_R3_battery-7
    batch = "unknown"
    protocol = "unknown"
    cell_uid = name
    m = re.search(r"Batch[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        batch = f"Batch-{m.group(1)}"
    # Protocol candidates
    for pat in (r"\b(2C)\b", r"\b(3C)\b", r"\b(R2\.5|R2p5|R25)\b", r"\b(R3)\b", r"\b(random[_-]?walk|RW)\b", r"\b(GEO)\b"):
        pm = re.search(pat, name, flags=re.IGNORECASE)
        if pm:
            protocol = pm.group(1).replace("R2p5", "R2.5").replace("R25", "R2.5")
            protocol = protocol.upper() if protocol.lower() in {"geo", "rw"} else protocol
            break
    bm = re.search(r"battery[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    if m and bm:
        cell_uid = f"Batch-{m.group(1)}_battery-{bm.group(1)}"
    return batch, protocol, cell_uid


def extract_scalar_from_npz(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return ""
    arr = data[key]
    try:
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            v = arr.reshape(-1)[0]
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)
    except Exception:
        return ""
    return ""


def choose_npz_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    files = set(data.files)
    for c in candidates:
        if c in files:
            return c
    # Accept common aliases
    aliases = {
        "phis_c": ["phis_c_soft", "voltage_pred", "voltage", "V", "V_pred", "terminal_voltage"],
        "phie": ["phi_e", "phi_e_soft"],
        "theta_a": ["theta_n", "theta_neg", "theta_negative", "theta_anode"],
        "theta_c": ["theta_p", "theta_pos", "theta_positive", "theta_cathode"],
        "cs_a": ["cs_n", "cs_neg", "cs_negative", "cs_anode"],
        "cs_c": ["cs_p", "cs_pos", "cs_positive", "cs_cathode"],
    }
    for c in candidates:
        for a in aliases.get(c, []):
            if a in files:
                return a
    return None


# ----------------------------- discovery -----------------------------------


def find_soft_label_npz(cell_dir: Path) -> Optional[Path]:
    for name in TARGET_FILENAME_CANDIDATES:
        p = cell_dir / name
        if p.exists():
            return p
    matches = sorted(cell_dir.glob("*.npz"))
    for p in matches:
        if "soft" in p.name.lower() or "solution" in p.name.lower():
            return p
    return matches[0] if matches else None


def read_manifest_if_available(root: Path) -> Optional[List[CellRecord]]:
    manifest_candidates = [
        root / "D15_ALL55_SOFTLABEL_MANIFEST.csv",
        root / "all55_softlabel_manifest.csv",
        root / "softlabel_manifest.csv",
    ]
    for manifest in manifest_candidates:
        if manifest.exists():
            records: List[CellRecord] = []
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Best-effort field names from D15 docs.
                    target = row.get("target_path") or row.get("soft_label_path") or row.get("path") or row.get("cell_dir") or ""
                    if target:
                        target_path = Path(target)
                        if not target_path.is_absolute():
                            target_path = root / target_path
                        if target_path.is_dir():
                            npz = find_soft_label_npz(target_path)
                            cell_dir = target_path
                        else:
                            npz = target_path if target_path.suffix.lower() == ".npz" else None
                            cell_dir = target_path.parent
                    else:
                        cell_name = row.get("cell_uid") or row.get("cell") or row.get("cell_id") or row.get("profile_key") or ""
                        cell_dir = root / cell_name
                        npz = find_soft_label_npz(cell_dir) if cell_dir.exists() else None
                    if not npz or not npz.exists():
                        continue
                    raw_name = row.get("cell_uid") or row.get("cell") or row.get("cell_id") or cell_dir.name
                    batch, protocol, cell_uid = parse_batch_protocol_cell_from_name(raw_name)
                    batch = row.get("batch") or batch
                    protocol = row.get("protocol") or protocol
                    flagged = str(row.get("flagged", row.get("is_flagged", "false"))).lower() in {"true", "1", "yes", "y"}
                    rec = CellRecord(
                        cell_uid=cell_uid,
                        batch=batch,
                        protocol=protocol,
                        soft_label_npz=str(npz),
                        cell_dir=str(cell_dir),
                        source_stage=row.get("source_stage") or row.get("stage") or "manifest",
                        is_seen=row.get("is_seen") or row.get("seen") or "unknown",
                        flagged=flagged,
                        note=row.get("note") or "",
                    )
                    records.append(rec)
            if records:
                return records
    return None


def discover_cells(soft_label_root: Path) -> List[CellRecord]:
    records = read_manifest_if_available(soft_label_root)
    if records:
        return sorted(records, key=lambda r: (r.batch, r.cell_uid, r.protocol))

    records = []
    for d in sorted(soft_label_root.iterdir()):
        if not d.is_dir():
            continue
        npz = find_soft_label_npz(d)
        if not npz:
            continue
        batch, protocol, cell_uid = parse_batch_protocol_cell_from_name(d.name)
        try:
            with np.load(npz, allow_pickle=True) as data:
                batch_from_npz = extract_scalar_from_npz(data, "batch") or batch
                protocol_from_npz = extract_scalar_from_npz(data, "protocol") or protocol
                cell_from_npz = extract_scalar_from_npz(data, "cell_uid") or extract_scalar_from_npz(data, "cell_id") or cell_uid
                batch, protocol, cell_uid = batch_from_npz, protocol_from_npz, cell_from_npz
        except Exception:
            pass
        flagged = "battery-8" in d.name.lower() and ("batch-1" in d.name.lower() or "batch_1" in d.name.lower())
        records.append(CellRecord(cell_uid=cell_uid, batch=batch, protocol=protocol, soft_label_npz=str(npz), cell_dir=str(d), flagged=flagged))
    return sorted(records, key=lambda r: (r.batch, r.cell_uid, r.protocol))


def check_cell_npz(record: CellRecord, required_keys: Sequence[str]) -> FileCheck:
    path = Path(record.soft_label_npz)
    if not path.exists():
        return FileCheck(False, str(path), "missing soft-label npz")
    try:
        with np.load(path, allow_pickle=True) as data:
            missing = [k for k in required_keys if k not in data.files]
            # Need at least one target state for evaluation.
            missing_targets = [k for k in DEFAULT_METRICS if choose_npz_key(data, [k]) is None]
            msg_parts = []
            if missing:
                msg_parts.append("missing_required=" + ",".join(missing))
            if len(missing_targets) == len(DEFAULT_METRICS):
                msg_parts.append("no_known_target_state_keys")
            if msg_parts:
                return FileCheck(False, str(path), "; ".join(msg_parts))
            return FileCheck(True, str(path), f"ok keys={len(data.files)}")
    except Exception as e:
        return FileCheck(False, str(path), f"np.load failed: {type(e).__name__}: {e}")


# ---------------------------- inference ------------------------------------


def build_format_context(record: CellRecord, model_dir: Optional[Path], output_npz: Path, extra: Dict[str, str]) -> Dict[str, str]:
    ctx = {
        "cell_uid": record.cell_uid,
        "batch": record.batch,
        "protocol": record.protocol,
        "soft_label_npz": record.soft_label_npz,
        "cell_dir": record.cell_dir,
        "output_npz": str(output_npz),
        "output_dir": str(output_npz.parent),
        "model_dir": str(model_dir) if model_dir else "",
    }
    ctx.update(extra)
    return ctx


def run_inference_for_cell(
    record: CellRecord,
    prediction_root: Path,
    model_dir: Optional[Path],
    template: str,
    dry_run: bool,
    extra_context: Dict[str, str],
    log_dir: Path,
) -> Tuple[Optional[Path], str]:
    cell_out_dir = prediction_root / sanitize_filename(record.cell_uid)
    cell_out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = cell_out_dir / "prediction.npz"
    if output_npz.exists():
        return output_npz, "existing_prediction"
    ctx = build_format_context(record, model_dir, output_npz, extra_context)
    try:
        cmd = template.format(**ctx)
    except KeyError as e:
        return None, f"template_missing_key:{e}"

    log_path = log_dir / f"{sanitize_filename(record.cell_uid)}.log"
    if dry_run:
        log_path.write_text("DRY RUN\n" + cmd + "\n", encoding="utf-8")
        return None, "dry_run_command_written"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n" + cmd + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        wall = time.time() - start
        log.write(f"\nEXIT_CODE={proc.returncode}\nWALL_SECONDS={wall:.3f}\n")
    if proc.returncode != 0:
        return None, f"inference_failed_exit_{proc.returncode}"
    if not output_npz.exists():
        return None, "inference_exit0_but_prediction_missing"
    return output_npz, "generated_prediction"


def sanitize_filename(s: str) -> str:
    s = str(s).strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s)


def find_prediction_npz(record: CellRecord, prediction_root: Optional[Path]) -> Optional[Path]:
    if prediction_root is None:
        return None
    candidates: List[Path] = []
    cell_safe = sanitize_filename(record.cell_uid)
    cell_dir_name = Path(record.cell_dir).name
    for base in [prediction_root / cell_safe, prediction_root / record.cell_uid, prediction_root / cell_dir_name]:
        for name in PREDICTION_FILENAME_CANDIDATES:
            candidates.append(base / name)
    for name in [f"{cell_safe}_prediction.npz", f"{record.cell_uid}_prediction.npz", f"{cell_dir_name}_prediction.npz"]:
        candidates.append(prediction_root / sanitize_filename(name))
        candidates.append(prediction_root / name)
    for p in candidates:
        if p.exists():
            return p
    # Recursive fallback, limited by name matching to avoid full huge search.
    try:
        for p in prediction_root.rglob("*.npz"):
            lower = str(p).lower()
            if cell_safe.lower() in lower or record.cell_uid.lower() in lower or cell_dir_name.lower() in lower:
                if "pred" in p.name.lower() or "prediction" in lower:
                    return p
    except Exception:
        return None
    return None


# ----------------------------- metric math ---------------------------------


def flatten_pair(true: np.ndarray, pred: np.ndarray, allow_time_trim: bool = False) -> Tuple[np.ndarray, np.ndarray, str]:
    t = np.asarray(true)
    p = np.asarray(pred)
    t = np.squeeze(t)
    p = np.squeeze(p)
    msg = ""
    if t.shape != p.shape:
        if allow_time_trim and t.ndim >= 1 and p.ndim >= 1:
            # Trim only first dimension; preserve remaining shape if possible.
            if t.shape[1:] == p.shape[1:]:
                n = min(t.shape[0], p.shape[0])
                t = t[:n]
                p = p[:n]
                msg = f"trimmed_time_to_{n}"
            else:
                raise ValueError(f"shape mismatch true={t.shape} pred={p.shape}")
        else:
            raise ValueError(f"shape mismatch true={t.shape} pred={p.shape}")
    return t.reshape(-1).astype(np.float64), p.reshape(-1).astype(np.float64), msg


def compute_basic_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "max_abs": np.nan, "bias": np.nan, "corr": np.nan, "r2": np.nan, "nmae_range": np.nan}
    t = true[mask]
    p = pred[mask]
    e = p - t
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    max_abs = float(np.max(np.abs(e)))
    bias = float(np.mean(e))
    if t.size > 1 and np.std(t) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = float("nan")
    denom = float(np.sum((t - np.mean(t)) ** 2))
    r2 = float(1.0 - np.sum(e ** 2) / denom) if denom > 0 else float("nan")
    rng = float(np.max(t) - np.min(t))
    nmae_range = float(mae / rng) if rng > 0 else float("nan")
    return {"n": int(mask.sum()), "mae": mae, "rmse": rmse, "max_abs": max_abs, "bias": bias, "corr": corr, "r2": r2, "nmae_range": nmae_range}


def status_for_metric(metric: str, m: Dict[str, float]) -> Tuple[str, str]:
    th = DEFAULT_THRESHOLDS.get(metric)
    if th is None:
        return "INFO", "no_threshold"
    mae_key = "mae"
    if "pass_nmae" in th:
        val = m.get("nmae_range", float("nan"))
        pass_lim = th["pass_nmae"]
        review_lim = th["review_nmae"]
        name = "nmae_range"
    else:
        val = m.get("mae", float("nan"))
        pass_lim = th["pass_mae"]
        review_lim = th["review_mae"]
        name = "mae"
    corr = m.get("corr", float("nan"))
    pass_corr = th.get("pass_corr", -np.inf)
    review_corr = th.get("review_corr", -np.inf)
    if not np.isfinite(val):
        return "FAIL", f"{name}=nan"
    if val <= pass_lim and (not np.isfinite(corr) or corr >= pass_corr):
        return "PASS", f"{name}<={pass_lim}"
    if val <= review_lim and (not np.isfinite(corr) or corr >= review_corr):
        return "REVIEW", f"{name}<={review_lim}"
    return "FAIL", f"{name}={val:.6g}, corr={corr:.6g}"


def infer_csmax(target_theta: Optional[np.ndarray], target_cs: Optional[np.ndarray]) -> Optional[float]:
    if target_theta is None or target_cs is None:
        return None
    theta = np.asarray(target_theta, dtype=np.float64).reshape(-1)
    cs = np.asarray(target_cs, dtype=np.float64).reshape(-1)
    mask = np.isfinite(theta) & np.isfinite(cs) & (np.abs(theta) > 1e-6) & (theta > 0.02) & (theta < 0.98)
    if mask.sum() < 100:
        return None
    ratio = cs[mask] / theta[mask]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < 100:
        return None
    return float(np.median(ratio))


def apply_theta_projection(pred: Dict[str, np.ndarray], target: Dict[str, np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    if mode == "raw":
        return pred
    out = {k: np.array(v, copy=True) for k, v in pred.items()}
    for side in ("a", "c"):
        theta_key = f"theta_{side}"
        cs_key = f"cs_{side}"
        if theta_key in out:
            out[theta_key] = np.clip(out[theta_key], 0.0, 1.0)
        # If both theta and cs are present, reconstruct cs using target-derived csmax.
        # This mirrors inference-time projection while avoiding hidden generator parameters.
        if theta_key in out and cs_key in out:
            csmax = infer_csmax(target.get(theta_key), target.get(cs_key))
            if csmax is not None:
                out[cs_key] = out[theta_key] * csmax
    return out


def extract_arrays(npz_path: Path, metrics: Sequence[str], role: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    arrays: Dict[str, np.ndarray] = {}
    keymap: Dict[str, str] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for metric in metrics:
            key = choose_npz_key(data, [metric])
            if key is not None:
                arrays[metric] = np.array(data[key])
                keymap[metric] = key
        # Some prediction files store dictionaries under pred/target names; intentionally not
        # parsed here because numpy object arrays are unsafe across unknown sources.
    return arrays, keymap


def compute_radial_gradient(arr: np.ndarray) -> Optional[np.ndarray]:
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim < 2:
        return None
    # Assume last dimension is radial grid. Supports (N,R) or (N,?,R) by flattening all
    # non-radial dimensions into samples.
    return (a[..., -1] - a[..., 0]).reshape(-1)


def evaluate_cell(
    record: CellRecord,
    prediction_npz: Path,
    metrics: Sequence[str],
    model_name: str,
    projection_modes: Sequence[str],
    allow_time_trim: bool,
) -> Tuple[List[MetricRecord], List[str], Dict[str, str]]:
    messages: List[str] = []
    target_arrays, target_keymap = extract_arrays(Path(record.soft_label_npz), metrics, "target")
    pred_arrays_raw, pred_keymap = extract_arrays(prediction_npz, metrics, "prediction")
    if not pred_arrays_raw:
        return [], ["no recognized prediction keys in " + str(prediction_npz)], {}
    keymap = {"target_keymap": json.dumps(target_keymap, ensure_ascii=False), "prediction_keymap": json.dumps(pred_keymap, ensure_ascii=False)}
    rows: List[MetricRecord] = []
    for projection_mode in projection_modes:
        pred_arrays = apply_theta_projection(pred_arrays_raw, target_arrays, projection_mode)
        for metric in metrics:
            if metric not in target_arrays:
                messages.append(f"target_missing:{metric}")
                continue
            if metric not in pred_arrays:
                messages.append(f"pred_missing:{metric}")
                continue
            try:
                true_flat, pred_flat, msg = flatten_pair(target_arrays[metric], pred_arrays[metric], allow_time_trim=allow_time_trim)
                if msg:
                    messages.append(f"{metric}:{msg}")
                mm = compute_basic_metrics(true_flat, pred_flat)
                status, status_msg = status_for_metric(metric, mm)
                rows.append(MetricRecord(
                    model_name=model_name,
                    cell_uid=record.cell_uid,
                    batch=record.batch,
                    protocol=record.protocol,
                    is_seen=record.is_seen,
                    flagged=record.flagged,
                    projection_mode=projection_mode,
                    metric=metric,
                    n=int(mm["n"]),
                    true_shape=str(np.squeeze(target_arrays[metric]).shape),
                    pred_shape=str(np.squeeze(pred_arrays[metric]).shape),
                    mae=safe_float(mm["mae"]),
                    rmse=safe_float(mm["rmse"]),
                    max_abs=safe_float(mm["max_abs"]),
                    bias=safe_float(mm["bias"]),
                    corr=safe_float(mm["corr"]),
                    r2=safe_float(mm["r2"]),
                    nmae_range=safe_float(mm["nmae_range"]),
                    status=status,
                    message=status_msg,
                ))
            except Exception as e:
                rows.append(MetricRecord(
                    model_name=model_name,
                    cell_uid=record.cell_uid,
                    batch=record.batch,
                    protocol=record.protocol,
                    is_seen=record.is_seen,
                    flagged=record.flagged,
                    projection_mode=projection_mode,
                    metric=metric,
                    n=0,
                    true_shape=str(np.squeeze(target_arrays[metric]).shape),
                    pred_shape=str(np.squeeze(pred_arrays.get(metric, np.array([]))).shape),
                    mae=float("nan"), rmse=float("nan"), max_abs=float("nan"), bias=float("nan"), corr=float("nan"), r2=float("nan"), nmae_range=float("nan"),
                    status="FAIL",
                    message=f"metric_failed:{type(e).__name__}:{e}",
                ))
        # radial gradient metrics for cs_a/cs_c/theta_a/theta_c if available
        for base_metric in ("cs_a", "cs_c", "theta_a", "theta_c"):
            if base_metric not in target_arrays or base_metric not in pred_arrays:
                continue
            t_grad = compute_radial_gradient(target_arrays[base_metric])
            p_grad = compute_radial_gradient(pred_arrays[base_metric])
            if t_grad is None or p_grad is None:
                continue
            grad_metric = "grad_" + base_metric
            try:
                true_flat, pred_flat, msg = flatten_pair(t_grad, p_grad, allow_time_trim=allow_time_trim)
                mm = compute_basic_metrics(true_flat, pred_flat)
                # Use cs gradient thresholds for theta gradients only as review hints.
                status, status_msg = status_for_metric(grad_metric if grad_metric in DEFAULT_THRESHOLDS else ("grad_cs_a" if base_metric.endswith("a") else "grad_cs_c"), mm)
                rows.append(MetricRecord(
                    model_name=model_name,
                    cell_uid=record.cell_uid,
                    batch=record.batch,
                    protocol=record.protocol,
                    is_seen=record.is_seen,
                    flagged=record.flagged,
                    projection_mode=projection_mode,
                    metric=grad_metric,
                    n=int(mm["n"]),
                    true_shape=str(np.squeeze(t_grad).shape),
                    pred_shape=str(np.squeeze(p_grad).shape),
                    mae=safe_float(mm["mae"]), rmse=safe_float(mm["rmse"]), max_abs=safe_float(mm["max_abs"]), bias=safe_float(mm["bias"]), corr=safe_float(mm["corr"]), r2=safe_float(mm["r2"]), nmae_range=safe_float(mm["nmae_range"]),
                    status=status,
                    message=status_msg,
                ))
            except Exception as e:
                messages.append(f"{grad_metric}:failed:{type(e).__name__}:{e}")
    return rows, messages, keymap


# ----------------------------- aggregation ---------------------------------


def aggregate_rows(rows: List[MetricRecord], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    dicts = [asdict(r) for r in rows]
    if pd is not None:
        df = pd.DataFrame(dicts)
        groups = []
        for keys, g in df.groupby(list(group_fields), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            out = {field: key for field, key in zip(group_fields, keys)}
            for col in ["mae", "rmse", "max_abs", "bias", "corr", "r2", "nmae_range"]:
                out[f"mean_{col}"] = safe_float(g[col].mean())
                out[f"median_{col}"] = safe_float(g[col].median())
                out[f"max_{col}"] = safe_float(g[col].max())
            out["cell_count"] = int(g["cell_uid"].nunique())
            out["row_count"] = int(len(g))
            out["fail_count"] = int((g["status"] == "FAIL").sum())
            out["review_count"] = int((g["status"] == "REVIEW").sum())
            out["pass_count"] = int((g["status"] == "PASS").sum())
            if out["fail_count"] > 0:
                out["status"] = "FAIL"
            elif out["review_count"] > 0:
                out["status"] = "REVIEW"
            else:
                out["status"] = "PASS"
            groups.append(out)
        return groups
    # fallback without pandas
    buckets: Dict[Tuple[Any, ...], List[MetricRecord]] = {}
    for r in rows:
        key = tuple(getattr(r, f) for f in group_fields)
        buckets.setdefault(key, []).append(r)
    result = []
    for key, vals in buckets.items():
        out = {field: k for field, k in zip(group_fields, key)}
        for col in ["mae", "rmse", "max_abs", "bias", "corr", "r2", "nmae_range"]:
            arr = np.array([getattr(v, col) for v in vals], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            out[f"mean_{col}"] = safe_float(np.mean(arr)) if arr.size else float("nan")
            out[f"median_{col}"] = safe_float(np.median(arr)) if arr.size else float("nan")
            out[f"max_{col}"] = safe_float(np.max(arr)) if arr.size else float("nan")
        out["cell_count"] = len(set(v.cell_uid for v in vals))
        out["row_count"] = len(vals)
        out["fail_count"] = sum(v.status == "FAIL" for v in vals)
        out["review_count"] = sum(v.status == "REVIEW" for v in vals)
        out["pass_count"] = sum(v.status == "PASS" for v in vals)
        out["status"] = "FAIL" if out["fail_count"] else ("REVIEW" if out["review_count"] else "PASS")
        result.append(out)
    return result


def make_scorecard(cell_records: List[CellRecord], metric_rows: List[MetricRecord], failures: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    n_cells = len(cell_records)
    cells_with_metrics = len(set(r.cell_uid for r in metric_rows))
    fail_rows = [r for r in metric_rows if r.status == "FAIL"]
    review_rows = [r for r in metric_rows if r.status == "REVIEW"]
    prediction_failures = [f for f in failures if f.get("stage") in {"prediction", "inference", "evaluation"}]
    if prediction_failures or fail_rows:
        status = "FAIL"
    elif review_rows or cells_with_metrics < n_cells:
        status = "REVIEW"
    else:
        status = "PASS"
    return {
        "task": "D16-P5A ALL55 existing-model transfer evaluation",
        "status": status,
        "model_name": args.model_name,
        "soft_label_root": str(args.soft_label_root),
        "prediction_root": str(args.prediction_root) if args.prediction_root else None,
        "output_root": str(args.output_root),
        "cell_count_discovered": n_cells,
        "cell_count_evaluated": cells_with_metrics,
        "metric_row_count": len(metric_rows),
        "fail_metric_row_count": len(fail_rows),
        "review_metric_row_count": len(review_rows),
        "failure_count": len(failures),
        "projection_modes": args.projection_modes,
        "boundary_note": "P2Dlite-RG labels are model-consistent soft labels, not experimental internal-state truth. This is transfer evaluation, not new training.",
        "next_decision": "If old models clearly underperform on ALL55, proceed to D16-P5B ALL55 unified NN training benchmark; otherwise archive transfer evidence and design held-out D16-P5C.",
        "created_at_unix": time.time(),
    }


# ----------------------------- CLI -----------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D16-P5A ALL55 existing-model transfer evaluation")
    p.add_argument("--soft-label-root", type=Path, required=True, help="D15 ALL55 final soft-label root directory")
    p.add_argument("--output-root", type=Path, required=True, help="Output directory for D16-P5A scorecards")
    p.add_argument("--prediction-root", type=Path, default=None, help="Root containing existing or generated prediction.npz files")
    p.add_argument("--model-dir", type=Path, default=None, help="Existing model directory, passed to inference command template")
    p.add_argument("--model-name", type=str, default="existing_model", help="Name to write in scorecards")
    p.add_argument("--inference-command-template", type=str, default="", help=(
        "Optional shell command template to generate one prediction per cell. Available placeholders: "
        "{model_dir}, {soft_label_npz}, {output_npz}, {output_dir}, {cell_uid}, {batch}, {protocol}, {cell_dir}."
    ))
    p.add_argument("--extra-context", action="append", default=[], help="Extra template context as KEY=VALUE; can be repeated")
    p.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS), help="Comma-separated metrics to evaluate")
    p.add_argument("--projection-modes", type=str, default="raw,projected", help="raw, projected, or raw,projected")
    p.add_argument("--preflight-only", action="store_true", help="Only check ALL55 soft labels and write manifest; no prediction evaluation")
    p.add_argument("--run-inference", action="store_true", help="Run inference command for missing predictions")
    p.add_argument("--dry-run", action="store_true", help="Write generated inference commands but do not execute them")
    p.add_argument("--allow-time-trim", action="store_true", help="Allow trimming first dimension when prediction/target length differs. Default false to avoid silent truncation.")
    p.add_argument("--required-label-keys", type=str, default=",".join(DEFAULT_REQUIRED_LABEL_KEYS), help="Comma-separated required label keys for preflight")
    p.add_argument("--seen-cells", type=Path, default=None, help="Optional CSV/TXT containing cell_uid values seen by the existing model")
    p.add_argument("--flagged-cells", type=str, default="Batch-1_battery-8", help="Comma-separated flagged cell_uid substrings")
    p.add_argument("--max-cells", type=int, default=0, help="Debug only: evaluate first N cells")
    p.add_argument("--skip-flagged", action="store_true", help="Skip flagged cells such as Batch-1 battery-8")
    return p.parse_args(argv)


def parse_extra_context(items: Sequence[str]) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-context must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        ctx[k] = v
    return ctx


def load_seen_cells(path: Optional[Path]) -> set:
    if path is None or not path.exists():
        return set()
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    vals = set()
    if path.suffix.lower() == ".csv":
        for line in text.splitlines():
            for part in line.split(","):
                part = part.strip()
                if part and part.lower() not in {"cell_uid", "cell", "cell_id"}:
                    vals.add(part)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                vals.add(line)
    return vals


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.soft_label_root = args.soft_label_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.prediction_root is not None:
        args.prediction_root = args.prediction_root.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    projection_modes = [m.strip() for m in args.projection_modes.split(",") if m.strip()]
    args.projection_modes = projection_modes
    required_keys = [k.strip() for k in args.required_label_keys.split(",") if k.strip()]
    flagged_substrings = [s.strip().lower() for s in args.flagged_cells.split(",") if s.strip()]
    seen_cells = load_seen_cells(args.seen_cells)
    extra_context = parse_extra_context(args.extra_context)

    json_dump({
        "argv": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metrics": metrics,
        "thresholds": DEFAULT_THRESHOLDS,
        "boundary_note": "D16-P5A does not train a new model and does not treat P2Dlite-RG labels as experimental truth.",
    }, args.output_root / "run_config.json")

    records = discover_cells(args.soft_label_root)
    if not records:
        print(f"[FAIL] No cell soft-label NPZ files discovered under {args.soft_label_root}", file=sys.stderr)
        json_dump({"status": "FAIL", "reason": "no_cells_discovered"}, args.output_root / "d16_p5a_scorecard.json")
        return 2

    # Apply seen/flagged metadata.
    for rec in records:
        rec.is_seen = "seen" if rec.cell_uid in seen_cells or Path(rec.cell_dir).name in seen_cells else ("unseen" if seen_cells else "unknown")
        lower = (rec.cell_uid + " " + Path(rec.cell_dir).name).lower()
        if any(s in lower for s in flagged_substrings):
            rec.flagged = True

    if args.skip_flagged:
        records = [r for r in records if not r.flagged]
    if args.max_cells and args.max_cells > 0:
        records = records[: args.max_cells]

    cell_manifest_rows = [asdict(r) for r in records]
    write_csv(cell_manifest_rows, args.output_root / "d16_p5a_cell_manifest.csv")

    checks: List[FileCheck] = [check_cell_npz(r, required_keys=required_keys) for r in records]
    check_rows = [asdict(c) for c in checks]
    write_csv(check_rows, args.output_root / "d16_p5a_softlabel_preflight.csv")
    bad_checks = [c for c in checks if not c.ok]

    print(f"[INFO] discovered_cells={len(records)} bad_softlabel_checks={len(bad_checks)} output={args.output_root}")
    if args.preflight_only:
        status = "FAIL" if bad_checks else "PASS"
        json_dump({
            "task": "D16-P5A preflight",
            "status": status,
            "cell_count": len(records),
            "bad_softlabel_check_count": len(bad_checks),
            "soft_label_root": str(args.soft_label_root),
            "output_root": str(args.output_root),
        }, args.output_root / "d16_p5a_scorecard.json")
        print(f"[DONE] preflight_only status={status}")
        return 0 if status == "PASS" else 1

    failures: List[Dict[str, Any]] = []
    for c in bad_checks:
        failures.append({"stage": "softlabel_preflight", **asdict(c)})

    if args.prediction_root is None:
        args.prediction_root = args.output_root / "generated_predictions"
    args.prediction_root.mkdir(parents=True, exist_ok=True)

    metric_rows: List[MetricRecord] = []
    keymap_rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, 1):
        print(f"[CELL {idx:03d}/{len(records):03d}] {rec.cell_uid} batch={rec.batch} protocol={rec.protocol} flagged={rec.flagged}")
        pred_npz = find_prediction_npz(rec, args.prediction_root)
        pred_source = "found_prediction" if pred_npz else "missing_prediction"
        if pred_npz is None and args.run_inference:
            if not args.inference_command_template:
                failures.append({"stage": "inference", "cell_uid": rec.cell_uid, "message": "--run-inference set but no --inference-command-template provided"})
                continue
            pred_npz, pred_source = run_inference_for_cell(
                rec,
                prediction_root=args.prediction_root,
                model_dir=args.model_dir,
                template=args.inference_command_template,
                dry_run=args.dry_run,
                extra_context=extra_context,
                log_dir=args.output_root / "logs",
            )
        if pred_npz is None:
            failures.append({"stage": "prediction", "cell_uid": rec.cell_uid, "batch": rec.batch, "protocol": rec.protocol, "message": pred_source})
            continue
        if args.dry_run:
            continue
        try:
            rows, messages, keymap = evaluate_cell(
                rec,
                prediction_npz=pred_npz,
                metrics=metrics,
                model_name=args.model_name,
                projection_modes=projection_modes,
                allow_time_trim=args.allow_time_trim,
            )
            metric_rows.extend(rows)
            keymap_rows.append({"cell_uid": rec.cell_uid, "prediction_npz": str(pred_npz), **keymap, "messages": " | ".join(messages)})
            if messages:
                failures.append({"stage": "evaluation", "cell_uid": rec.cell_uid, "prediction_npz": str(pred_npz), "message": " | ".join(messages)})
        except Exception as e:
            failures.append({"stage": "evaluation", "cell_uid": rec.cell_uid, "prediction_npz": str(pred_npz), "message": f"{type(e).__name__}: {e}"})

    metric_dicts = [asdict(r) for r in metric_rows]
    write_csv(metric_dicts, args.output_root / "d16_p5a_cell_metrics.csv")
    write_csv(keymap_rows, args.output_root / "d16_p5a_keymap_and_messages.csv")
    write_csv(failures, args.output_root / "d16_p5a_failures.csv")

    batch_summary = aggregate_rows(metric_rows, ["model_name", "projection_mode", "batch", "metric"])
    protocol_summary = aggregate_rows(metric_rows, ["model_name", "projection_mode", "protocol", "metric"])
    seen_summary = aggregate_rows(metric_rows, ["model_name", "projection_mode", "is_seen", "metric"])
    flagged_summary = aggregate_rows(metric_rows, ["model_name", "projection_mode", "flagged", "metric"])
    overall_summary = aggregate_rows(metric_rows, ["model_name", "projection_mode", "metric"])

    write_csv(batch_summary, args.output_root / "d16_p5a_batch_metrics.csv")
    write_csv(protocol_summary, args.output_root / "d16_p5a_protocol_metrics.csv")
    write_csv(seen_summary, args.output_root / "d16_p5a_seen_unseen_metrics.csv")
    write_csv(flagged_summary, args.output_root / "d16_p5a_flagged_metrics.csv")
    write_csv(overall_summary, args.output_root / "d16_p5a_overall_metrics.csv")

    scorecard = make_scorecard(records, metric_rows, failures, args)
    scorecard["overall_summary"] = overall_summary
    scorecard["batch_summary_file"] = str(args.output_root / "d16_p5a_batch_metrics.csv")
    scorecard["cell_metrics_file"] = str(args.output_root / "d16_p5a_cell_metrics.csv")
    scorecard["failures_file"] = str(args.output_root / "d16_p5a_failures.csv")
    json_dump(scorecard, args.output_root / "d16_p5a_scorecard.json")

    print(f"[DONE] status={scorecard['status']} cells={scorecard['cell_count_evaluated']}/{scorecard['cell_count_discovered']} metric_rows={scorecard['metric_row_count']}")
    print(f"[OUT] {args.output_root / 'd16_p5a_scorecard.json'}")
    return 0 if scorecard["status"] in {"PASS", "REVIEW"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
