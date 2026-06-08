# -*- coding: utf-8 -*-
"""Dataset and feature utilities for D14-P5 XJTU P2Dlite soft-label NN smoke.

The goal is not full training. This module builds a small supervised dataset
from P4B-v3 `solution_softlabels.npz` files and keeps enough metadata to audit
prior-hash consistency, split membership, and target shapes.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


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


def infer_batch_protocol_from_cell_uid(cell_uid: str) -> Tuple[str, str]:
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


def discover_softlabel_profiles(softlabel_root: str | Path) -> List[dict]:
    root = Path(softlabel_root)
    rows: List[dict] = []
    if not root.exists():
        return rows
    for npz_path in root.rglob("solution_softlabels.npz"):
        cell_dir = npz_path.parent
        summary = read_json(cell_dir / "soft_label_summary.json") or {}
        audit = read_json(cell_dir / "soft_label_audit.json") or {}
        try:
            data = np.load(npz_path, allow_pickle=True)
            cell_uid = scalar_to_str(data["cell_uid"]) if "cell_uid" in data.files else cell_dir.name
            batch = scalar_to_str(data["batch"]) if "batch" in data.files else ""
            protocol = scalar_to_str(data["protocol"]) if "protocol" in data.files else ""
            prior_hash = scalar_to_str(data["resolved_spec_hash"]) if "resolved_spec_hash" in data.files else ""
            n_points = int(data["t_global_s"].shape[0])
            n_r_a = int(data["theta_a"].shape[1])
            n_r_c = int(data["theta_c"].shape[1])
        except Exception as exc:
            rows.append({
                "cell_uid": cell_dir.name,
                "softlabel_npz": str(npz_path),
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue

        if not batch or not protocol:
            b2, p2 = infer_batch_protocol_from_cell_uid(cell_uid)
            batch = batch or b2
            protocol = protocol or p2

        rows.append({
            "cell_uid": cell_uid or cell_dir.name,
            "batch": batch,
            "protocol": protocol,
            "split": "",
            "softlabel_npz": str(npz_path),
            "summary_json": str(cell_dir / "soft_label_summary.json"),
            "audit_json": str(cell_dir / "soft_label_audit.json"),
            "status": audit.get("status", "PASS"),
            "audit_detail": audit.get("detail", ""),
            "n_points": n_points,
            "n_r_a": n_r_a,
            "n_r_c": n_r_c,
            "prior_hash": prior_hash,
            "source_profile_npz": summary.get("source_profile_npz", ""),
        })
    rows.sort(key=lambda r: (r.get("batch", ""), r.get("protocol", ""), r.get("cell_uid", "")))
    return rows


def assign_splits(rows: List[dict], cfg: dict) -> List[dict]:
    split_cfg = cfg.get("profile_split", {})
    explicit = split_cfg.get("explicit_cell_uid_split", {})
    fallback = split_cfg.get("fallback_split_order", ["train", "val", "test"])
    out = []
    for i, row in enumerate(rows):
        rr = dict(row)
        cell_uid = rr.get("cell_uid", "")
        split = explicit.get(cell_uid, "")
        if not split:
            # More forgiving match: explicit key substring or cell_uid substring.
            for key, val in explicit.items():
                if key in cell_uid or cell_uid in key:
                    split = val
                    break
        if not split:
            split = fallback[i % len(fallback)]
        rr["split"] = split
        out.append(rr)
    return out


def save_manifest(rows: List[dict], out_csv: str | Path, out_json: str | Path) -> None:
    write_csv(Path(out_csv), rows)
    write_json(Path(out_json), {"profiles": rows, "profile_count": len(rows)})


def one_hot(value: str, choices: List[str]) -> np.ndarray:
    vec = np.zeros(len(choices), dtype=np.float32)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1
    vec[idx] = 1.0
    return vec


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


def select_indices(t: np.ndarray, voltage: np.ndarray, step_id: np.ndarray, split: str, cfg: dict) -> np.ndarray:
    sampling = cfg.get("sampling", {})
    if split == "train":
        max_points = int(sampling.get("max_points_per_profile_train", 12000))
    elif split == "val":
        max_points = int(sampling.get("max_points_per_profile_val", 12000))
    else:
        max_points = int(sampling.get("max_points_per_profile_test", 16000))

    N = len(t)
    if N <= max_points:
        return np.arange(N, dtype=np.int64)

    idx = set(np.linspace(0, N - 1, int(max_points * (1.0 - float(sampling.get("random_fraction", 0.2)))), dtype=np.int64).tolist())

    if sampling.get("preserve_low_voltage", True):
        low_thr = float(sampling.get("low_voltage_threshold_V", 3.05))
        low_idx = np.where(voltage <= low_thr)[0]
        if low_idx.size:
            low_stride = max(1, int(math.ceil(low_idx.size / max(1, max_points // 5))))
            idx.update(low_idx[::low_stride].tolist())

    if sampling.get("preserve_step_transitions", True) and step_id is not None and len(step_id) == N:
        trans = np.where(step_id[1:] != step_id[:-1])[0] + 1
        for k in trans:
            for j in range(max(0, int(k) - 2), min(N, int(k) + 3)):
                idx.add(j)

    rng = np.random.default_rng(int(sampling.get("seed", 42)) + (0 if split == "train" else 1000))
    random_target = max_points - len(idx)
    if random_target > 0:
        rand_idx = rng.choice(np.arange(N), size=min(random_target, N), replace=False)
        idx.update(rand_idx.tolist())

    arr = np.array(sorted(idx), dtype=np.int64)
    if arr.size > max_points:
        keep = np.linspace(0, arr.size - 1, max_points).round().astype(np.int64)
        arr = arr[keep]
    return np.unique(arr)


@dataclass
class FeatureStats:
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

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_feature_matrix(arrays: Dict[str, np.ndarray], meta: dict, stats: Optional[FeatureStats], cfg: dict) -> np.ndarray:
    t = arrays["t_global_s"].astype(np.float32)
    I = arrays["I_profile"].astype(np.float32)
    V = arrays["voltage_exp"].astype(np.float32)
    T = arrays["temperature_C"].astype(np.float32)
    cycle = arrays["cycle_id"].astype(np.float32)
    step_type = normalize_step_type_array(arrays["step_type"], I)

    t_norm = (t - t.min()) / max(float(t.max() - t.min()), 1.0)
    q = np.cumsum(np.r_[0.0, 0.5 * (I[1:] + I[:-1]) * np.diff(t)]).astype(np.float32)
    q_scale = max(float(np.nanmax(np.abs(q))), 1e-6)
    q_norm = q / q_scale

    current_scale = stats.current_scale_A if stats else max(float(np.nanmax(np.abs(I))), 1e-6)
    voltage_mean = stats.voltage_mean_V if stats else 3.5
    voltage_scale = stats.voltage_scale_V if stats else 0.8
    temp_mean = stats.temperature_mean_C if stats else (float(np.nanmean(T)) if np.isfinite(T).any() else 25.0)
    temp_scale = stats.temperature_scale_C if stats else 10.0

    cycle_norm = (cycle - np.nanmin(cycle)) / max(float(np.nanmax(cycle) - np.nanmin(cycle)), 1.0)

    cols = [
        t_norm[:, None],
        q_norm[:, None],
        (I / current_scale)[:, None],
        ((V - voltage_mean) / voltage_scale)[:, None],
        ((np.nan_to_num(T, nan=temp_mean) - temp_mean) / temp_scale)[:, None],
        cycle_norm[:, None],
    ]

    batch_oh = one_hot(meta.get("batch", "unknown"), BATCHES)
    protocol_oh = one_hot(meta.get("protocol", "unknown"), PROTOCOLS)
    step_oh = np.stack([one_hot(s, STEP_TYPES) for s in step_type], axis=0)
    cols.append(step_oh)
    cols.append(np.repeat(batch_oh[None, :], len(t), axis=0))
    cols.append(np.repeat(protocol_oh[None, :], len(t), axis=0))

    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X


def load_profile_sample(row: dict, cfg: dict, stats: Optional[FeatureStats] = None) -> Dict[str, Any]:
    path = Path(row["softlabel_npz"])
    data = np.load(path, allow_pickle=True)
    split = row.get("split", "train")
    idx = select_indices(data["t_global_s"], data["voltage_exp"], data["step_id"], split, cfg)

    arrays = {
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

    X = build_feature_matrix(arrays, row, stats, cfg)
    return {"X": X, "arrays": arrays, "idx": idx, "row": row}


def estimate_feature_stats(samples: List[Dict[str, Any]], prior_hash: str) -> FeatureStats:
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

    feature_dim = samples[0]["X"].shape[1]
    n_r = samples[0]["arrays"]["theta_a"].shape[1]
    return FeatureStats(
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
        feature_dim=feature_dim,
        n_r=n_r,
        prior_hash=prior_hash,
    )


def stats_from_json(path: str | Path) -> FeatureStats:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureStats(**obj)


class P2DliteTensorDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], stats: FeatureStats):
        Xs = []
        theta_a = []
        theta_c = []
        phie = []
        phis = []
        profile_ids = []
        for pid, s in enumerate(samples):
            Xs.append(s["X"].astype(np.float32))
            theta_a.append(s["arrays"]["theta_a"].astype(np.float32))
            theta_c.append(s["arrays"]["theta_c"].astype(np.float32))
            phie.append(((s["arrays"]["phie"] - stats.phie_mean) / stats.phie_std).astype(np.float32)[:, None])
            phis.append(((s["arrays"]["phis_c"] - stats.phis_c_mean) / stats.phis_c_std).astype(np.float32)[:, None])
            profile_ids.append(np.full((s["X"].shape[0], 1), pid, dtype=np.int64))

        self.X = torch.from_numpy(np.concatenate(Xs, axis=0))
        self.theta_a = torch.from_numpy(np.concatenate(theta_a, axis=0))
        self.theta_c = torch.from_numpy(np.concatenate(theta_c, axis=0))
        self.phie = torch.from_numpy(np.concatenate(phie, axis=0))
        self.phis_c = torch.from_numpy(np.concatenate(phis, axis=0))
        self.profile_id = torch.from_numpy(np.concatenate(profile_ids, axis=0).reshape(-1))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "theta_a": self.theta_a[idx],
            "theta_c": self.theta_c[idx],
            "phie": self.phie[idx],
            "phis_c": self.phis_c[idx],
            "profile_id": self.profile_id[idx],
        }
