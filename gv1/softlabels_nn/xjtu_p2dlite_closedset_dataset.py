# -*- coding: utf-8 -*-
"""Closed-set dataset utilities for D14-P5B XJTU P2Dlite precision benchmark.

Design goals
------------
- Read the eight D14-P4B-v3 P2Dlite soft-label profiles.
- Keep all selected profiles in a closed-set train+eval benchmark.
- Build richer features than P5, including Fourier time/charge features and
  profile one-hot features, so the NN can reach calibration-level accuracy.
- Keep the whole dataset on GPU when requested, to improve GPU utilization.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch


BATCHES = ["Batch-1", "Batch-3", "Batch-4", "Batch-5", "Batch-6", "unknown"]
PROTOCOLS = ["2C", "R2.5", "R3", "random_walk", "GEO", "unknown"]
STEP_TYPES = ["charge", "discharge", "rest", "unknown"]


def read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scalar_to_str(x) -> str:
    try:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        s = str(x).strip()
        if s in {"", "None", "none", "nan", "NaN", "[]"}:
            return ""
        return s
    except Exception:
        return ""


def infer_batch_protocol(cell_uid: str) -> Tuple[str, str]:
    text = str(cell_uid)
    lower = text.lower()
    if "r2.5" in lower or "r2_5" in lower or "r2-5" in lower:
        return "Batch-3", "R2.5"
    if re.search(r"(^|[_\-\s])r3($|[_\-\s])", lower) or "_r3_" in lower:
        return "Batch-4", "R3"
    if "batch-4" in lower or "batch_4" in lower:
        return "Batch-4", "R3"
    if "batch-3" in lower or "batch_3" in lower:
        return "Batch-3", "R2.5"
    if "batch-5" in lower or "batch_5" in lower:
        return "Batch-5", "random_walk"
    if "batch-6" in lower or "batch_6" in lower:
        return "Batch-6", "GEO"
    if "_2c_" in lower or "-2c-" in lower or "batch-1" in lower or "batch_1" in lower:
        return "Batch-1", "2C"
    return "unknown", "unknown"


def one_hot(value: str, choices: List[str]) -> np.ndarray:
    vec = np.zeros(len(choices), dtype=np.float32)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1
    vec[idx] = 1.0
    return vec


def discover_profiles(root: str | Path, cfg: dict) -> List[dict]:
    root = Path(root)
    rows: List[dict] = []
    if not root.exists():
        return rows
    exclude = [s.lower() for s in cfg.get("profile_policy", {}).get("exclude_exact_outliers", [])]
    for npz_path in root.rglob("solution_softlabels.npz"):
        cell_dir = npz_path.parent
        audit = read_json(cell_dir / "soft_label_audit.json") or {}
        summary = read_json(cell_dir / "soft_label_summary.json") or {}
        try:
            data = np.load(npz_path, allow_pickle=True)
            cell_uid = scalar_to_str(data["cell_uid"]) if "cell_uid" in data.files else cell_dir.name
            batch = scalar_to_str(data["batch"]) if "batch" in data.files else ""
            protocol = scalar_to_str(data["protocol"]) if "protocol" in data.files else ""
            prior_hash = scalar_to_str(data["resolved_spec_hash"]) if "resolved_spec_hash" in data.files else ""
            n_points = int(data["t_global_s"].shape[0])
            n_r = int(data["theta_a"].shape[1])
        except Exception as exc:
            rows.append({
                "cell_uid": cell_dir.name,
                "softlabel_npz": str(npz_path),
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue

        if not batch or not protocol:
            b2, p2 = infer_batch_protocol(cell_uid)
            batch = batch or b2
            protocol = protocol or p2

        text = (str(npz_path) + " " + cell_uid).lower()
        excluded = any(e in text for e in exclude)
        rows.append({
            "cell_uid": cell_uid,
            "batch": batch,
            "protocol": protocol,
            "split": "closed_train_eval",
            "softlabel_npz": str(npz_path),
            "summary_json": str(cell_dir / "soft_label_summary.json"),
            "audit_json": str(cell_dir / "soft_label_audit.json"),
            "status": "FAIL" if excluded else audit.get("status", "PASS"),
            "audit_detail": "excluded_by_policy" if excluded else audit.get("detail", ""),
            "n_points": n_points,
            "n_r": n_r,
            "prior_hash": prior_hash,
            "source_profile_npz": summary.get("source_profile_npz", ""),
        })
    rows.sort(key=lambda r: (r.get("batch", ""), r.get("protocol", ""), r.get("cell_uid", "")))
    return rows


def normalize_step_type_array(step_type: np.ndarray, I: np.ndarray) -> np.ndarray:
    out = []
    for s, i in zip(step_type, I):
        ss = str(s).lower()
        if "charge" in ss and "discharge" not in ss:
            out.append("charge")
        elif "discharge" in ss:
            out.append("discharge")
        elif "rest" in ss or abs(float(i)) < 1e-8:
            out.append("rest")
        else:
            if float(i) > 0:
                out.append("charge")
            elif float(i) < 0:
                out.append("discharge")
            else:
                out.append("unknown")
    return np.asarray(out, dtype="<U16")


def select_indices(t: np.ndarray, voltage: np.ndarray, step_id: np.ndarray, cfg: dict) -> np.ndarray:
    sampling = cfg.get("sampling", {})
    N = len(t)
    max_points = int(sampling.get("max_points_per_profile", 50000))
    if N <= max_points:
        return np.arange(N, dtype=np.int64)

    uniform_n = int(max_points * float(sampling.get("uniform_fraction", 0.55)))
    random_n = int(max_points * float(sampling.get("random_fraction", 0.25)))
    low_n = int(max_points * float(sampling.get("low_voltage_fraction", 0.10)))
    trans_n = int(max_points * float(sampling.get("transition_fraction", 0.10)))
    idx = set(np.linspace(0, N - 1, max(2, uniform_n), dtype=np.int64).tolist())

    rng = np.random.default_rng(int(sampling.get("seed", 42)))
    if random_n > 0:
        idx.update(rng.choice(np.arange(N), size=min(random_n, N), replace=False).tolist())

    if sampling.get("preserve_low_voltage", True) and low_n > 0:
        low_thr = float(sampling.get("low_voltage_threshold_V", 3.05))
        low_idx = np.where(voltage <= low_thr)[0]
        if low_idx.size:
            if low_idx.size <= low_n:
                idx.update(low_idx.tolist())
            else:
                idx.update(low_idx[np.linspace(0, low_idx.size - 1, low_n).round().astype(np.int64)].tolist())

    if sampling.get("preserve_step_transitions", True) and trans_n > 0 and step_id is not None and len(step_id) == N:
        trans = np.where(step_id[1:] != step_id[:-1])[0] + 1
        trans_points = []
        for k in trans:
            for j in range(max(0, int(k) - 3), min(N, int(k) + 4)):
                trans_points.append(j)
        trans_points = np.asarray(sorted(set(trans_points)), dtype=np.int64)
        if trans_points.size:
            if trans_points.size <= trans_n:
                idx.update(trans_points.tolist())
            else:
                idx.update(trans_points[np.linspace(0, trans_points.size - 1, trans_n).round().astype(np.int64)].tolist())

    arr = np.asarray(sorted(idx), dtype=np.int64)
    if arr.size > max_points:
        keep = np.linspace(0, arr.size - 1, max_points).round().astype(np.int64)
        arr = arr[keep]
    return np.unique(arr)


@dataclass
class ClosedSetStats:
    current_scale_A: float
    voltage_mean_V: float
    voltage_scale_V: float
    temperature_mean_C: float
    temperature_scale_C: float
    phie_mean: float
    phie_std: float
    phis_c_mean: float
    phis_c_std: float
    cmax_a_est: float
    cmax_c_est: float
    feature_dim: int
    n_r: int
    prior_hash: str
    profile_ids: List[str]

    def to_dict(self):
        return dict(self.__dict__)


def stats_from_json(path: str | Path) -> ClosedSetStats:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClosedSetStats(**obj)


def fourier_features(x: np.ndarray, n_freq: int) -> List[np.ndarray]:
    cols = []
    if n_freq <= 0:
        return cols
    for k in range(n_freq):
        freq = 2.0 ** k * np.pi
        cols.append(np.sin(freq * x)[:, None].astype(np.float32))
        cols.append(np.cos(freq * x)[:, None].astype(np.float32))
    return cols


def build_features(arr: Dict[str, np.ndarray], row: dict, profile_index: int, stats: Optional[ClosedSetStats], cfg: dict, profile_ids: List[str]) -> np.ndarray:
    feature_cfg = cfg.get("input_features", {})
    t = arr["t_global_s"].astype(np.float32)
    I = arr["I_profile"].astype(np.float32)
    V = arr["voltage_exp"].astype(np.float32)
    T = arr["temperature_C"].astype(np.float32)
    cycle = arr["cycle_id"].astype(np.float32)
    step_type = normalize_step_type_array(arr["step_type"], I)

    t_norm = (t - t.min()) / max(float(t.max() - t.min()), 1.0)
    q = np.cumsum(np.r_[0.0, 0.5 * (I[1:] + I[:-1]) * np.diff(t)]).astype(np.float32)
    q_norm = q / max(float(np.nanmax(np.abs(q))), 1e-6)

    if stats:
        current_scale = stats.current_scale_A
        voltage_mean = stats.voltage_mean_V
        voltage_scale = stats.voltage_scale_V
        temp_mean = stats.temperature_mean_C
        temp_scale = stats.temperature_scale_C
    else:
        current_scale = max(float(np.nanmax(np.abs(I))), 1e-6)
        voltage_mean = float(np.nanmean(V))
        voltage_scale = max(float(np.nanstd(V)), 0.2)
        temp_mean = float(np.nanmean(T)) if np.isfinite(T).any() else 25.0
        temp_scale = max(float(np.nanstd(T)), 1.0)

    cycle_norm = (cycle - np.nanmin(cycle)) / max(float(np.nanmax(cycle) - np.nanmin(cycle)), 1.0)

    cols = [
        t_norm[:, None].astype(np.float32),
        q_norm[:, None].astype(np.float32),
        (I / current_scale)[:, None].astype(np.float32),
        ((V - voltage_mean) / voltage_scale)[:, None].astype(np.float32),
        ((np.nan_to_num(T, nan=temp_mean) - temp_mean) / temp_scale)[:, None].astype(np.float32),
        cycle_norm[:, None].astype(np.float32),
    ]

    cols.extend(fourier_features(t_norm, int(feature_cfg.get("fourier_time_features", 12))))
    cols.extend(fourier_features(q_norm, int(feature_cfg.get("fourier_charge_features", 6))))

    if feature_cfg.get("include_current_voltage_interactions", True):
        I_scaled = I / current_scale
        V_scaled = (V - voltage_mean) / voltage_scale
        cols.extend([
            (I_scaled * V_scaled)[:, None].astype(np.float32),
            (I_scaled ** 2)[:, None].astype(np.float32),
            (V_scaled ** 2)[:, None].astype(np.float32),
        ])

    if feature_cfg.get("include_step_type_onehot", True):
        cols.append(np.stack([one_hot(s, STEP_TYPES) for s in step_type], axis=0))
    if feature_cfg.get("include_batch_onehot", True):
        cols.append(np.repeat(one_hot(row.get("batch", "unknown"), BATCHES)[None, :], len(t), axis=0))
    if feature_cfg.get("include_protocol_onehot", True):
        cols.append(np.repeat(one_hot(row.get("protocol", "unknown"), PROTOCOLS)[None, :], len(t), axis=0))
    if feature_cfg.get("include_profile_onehot", True):
        prof_vec = np.zeros(len(profile_ids), dtype=np.float32)
        prof_vec[profile_index] = 1.0
        cols.append(np.repeat(prof_vec[None, :], len(t), axis=0))

    return np.concatenate(cols, axis=1).astype(np.float32)


def load_profile_sample(row: dict, cfg: dict, stats: Optional[ClosedSetStats], profile_index: int, profile_ids: List[str]) -> Dict[str, Any]:
    data = np.load(row["softlabel_npz"], allow_pickle=True)
    idx = select_indices(data["t_global_s"], data["voltage_exp"], data["step_id"], cfg)
    arr = {
        "t_global_s": data["t_global_s"][idx].astype(np.float32),
        "I_profile": data["I_profile"][idx].astype(np.float32),
        "voltage_exp": data["voltage_exp"][idx].astype(np.float32),
        "temperature_C": data["temperature_C"][idx].astype(np.float32),
        "cycle_id": data["cycle_id"][idx].astype(np.int32),
        "step_id": data["step_id"][idx].astype(np.int32),
        "step_type": data["step_type"][idx],
        "theta_a": data["theta_a"][idx].astype(np.float32),
        "theta_c": data["theta_c"][idx].astype(np.float32),
        "cs_a": data["cs_a"][idx].astype(np.float32),
        "cs_c": data["cs_c"][idx].astype(np.float32),
        "phie": data["phie"][idx].astype(np.float32),
        "phis_c": data["phis_c"][idx].astype(np.float32),
    }
    X = build_features(arr, row, profile_index, stats, cfg, profile_ids)
    return {"row": row, "idx": idx, "arrays": arr, "X": X, "profile_index": profile_index}


def estimate_stats(samples: List[Dict[str, Any]], prior_hash: str, profile_ids: List[str]) -> ClosedSetStats:
    I = np.concatenate([s["arrays"]["I_profile"] for s in samples])
    V = np.concatenate([s["arrays"]["voltage_exp"] for s in samples])
    T = np.concatenate([s["arrays"]["temperature_C"] for s in samples])
    phie = np.concatenate([s["arrays"]["phie"] for s in samples])
    phis = np.concatenate([s["arrays"]["phis_c"] for s in samples])
    cs_a = np.concatenate([s["arrays"]["cs_a"].reshape(-1) for s in samples])
    cs_c = np.concatenate([s["arrays"]["cs_c"].reshape(-1) for s in samples])
    th_a = np.concatenate([s["arrays"]["theta_a"].reshape(-1) for s in samples])
    th_c = np.concatenate([s["arrays"]["theta_c"].reshape(-1) for s in samples])

    def safe_std(x, default=1.0):
        val = float(np.nanstd(x))
        return val if np.isfinite(val) and val > 1e-8 else default

    mask_a = np.isfinite(cs_a) & np.isfinite(th_a) & (np.abs(th_a) > 1e-4)
    mask_c = np.isfinite(cs_c) & np.isfinite(th_c) & (np.abs(th_c) > 1e-4)
    cmax_a = float(np.nanmedian(cs_a[mask_a] / th_a[mask_a])) if mask_a.any() else 31410.0
    cmax_c = float(np.nanmedian(cs_c[mask_c] / th_c[mask_c])) if mask_c.any() else 48839.0

    return ClosedSetStats(
        current_scale_A=max(float(np.nanmax(np.abs(I))), 1e-6),
        voltage_mean_V=float(np.nanmean(V)),
        voltage_scale_V=max(safe_std(V, 0.8), 0.2),
        temperature_mean_C=float(np.nanmean(T)) if np.isfinite(T).any() else 25.0,
        temperature_scale_C=max(safe_std(T, 10.0), 1.0),
        phie_mean=float(np.nanmean(phie)),
        phie_std=max(safe_std(phie, 1.0), 0.05),
        phis_c_mean=float(np.nanmean(phis)),
        phis_c_std=max(safe_std(phis, 1.0), 0.05),
        cmax_a_est=cmax_a,
        cmax_c_est=cmax_c,
        feature_dim=samples[0]["X"].shape[1],
        n_r=samples[0]["arrays"]["theta_a"].shape[1],
        prior_hash=prior_hash,
        profile_ids=profile_ids,
    )


def concatenate_tensors(samples: List[Dict[str, Any]], stats: ClosedSetStats, device: torch.device, gpu_resident: bool) -> Dict[str, torch.Tensor]:
    X = np.concatenate([s["X"] for s in samples], axis=0).astype(np.float32)
    theta_a = np.concatenate([s["arrays"]["theta_a"] for s in samples], axis=0).astype(np.float32)
    theta_c = np.concatenate([s["arrays"]["theta_c"] for s in samples], axis=0).astype(np.float32)
    phie = np.concatenate([((s["arrays"]["phie"] - stats.phie_mean) / stats.phie_std)[:, None] for s in samples], axis=0).astype(np.float32)
    phis_c = np.concatenate([((s["arrays"]["phis_c"] - stats.phis_c_mean) / stats.phis_c_std)[:, None] for s in samples], axis=0).astype(np.float32)
    profile_id = np.concatenate([np.full((s["X"].shape[0],), s["profile_index"], dtype=np.int64) for s in samples], axis=0)

    tensors = {
        "X": torch.from_numpy(X),
        "theta_a": torch.from_numpy(theta_a),
        "theta_c": torch.from_numpy(theta_c),
        "phie": torch.from_numpy(phie),
        "phis_c": torch.from_numpy(phis_c),
        "profile_id": torch.from_numpy(profile_id),
    }
    if gpu_resident and device.type == "cuda":
        tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    return tensors


def memory_summary(tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out = {}
    total = 0
    for k, v in tensors.items():
        bytes_i = v.numel() * v.element_size()
        out[f"{k}_shape"] = list(v.shape)
        out[f"{k}_MB"] = round(bytes_i / 1024 / 1024, 3)
        total += bytes_i
    out["total_tensor_MB"] = round(total / 1024 / 1024, 3)
    return out
