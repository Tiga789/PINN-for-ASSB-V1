#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D17-G7-S0 full-cycle sampling / data coverage audit.

Purpose
-------
This is a no-training, no-checkpoint-selection, coverage-only stage.
It reads only time/cycle/index-like arrays from replay/soft-label NPZ files,
then builds a deterministic full-cycle stratified sampling plan for later
full-cycle/cycle-aware surrogate smoke training.

It intentionally does NOT load large state arrays such as cs_a/cs_c/theta/phie/phis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TIME_KEYS = ["t_global_s", "time_s", "t_s", "t", "time"]
CYCLE_KEYS = ["cycle_id", "cycle", "cycles", "cycle_index"]
STEP_ID_KEYS = ["step_id", "step"]
STEP_TYPE_KEYS = ["step_type", "mode", "state"]

FORBIDDEN_LARGE_STATE_KEYS = {
    "cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c",
    "cs_a_soft", "cs_c_soft", "theta_a_soft", "theta_c_soft", "phie_soft", "phis_c_soft",
    "cs_a_source_p2dlite_v1", "cs_c_source_p2dlite_v1",
}

ID_FIELDS = [
    "canonical_cell_uid", "canonical_cell_id", "cell_uid", "cell_id", "profile_id",
    "source_profile_name", "profile", "name",
]


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        if not fieldnames:
            fieldnames = ["empty"]
            rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_splits(s: str) -> List[str]:
    if not s or s.lower() == "all":
        return ["train", "validation", "frozen_test", "flagged_probe"]
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def canonicalize_id(s: Any) -> str:
    text = str(s or "").strip().replace("\\", "/")
    if "/" in text:
        text = text.split("/")[-1]
    return text


def load_split_manifest(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    d = read_json(path)
    if isinstance(d, list):
        records = d
        meta = {"records": records}
    else:
        records = d.get("records") or d.get("manifest") or d.get("items") or []
        meta = d
    if not isinstance(records, list):
        raise ValueError(f"Cannot find record list in split manifest: {path}")
    return records, meta


def build_semantics_map(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not csv_path or not csv_path.exists():
        return {}
    rows = read_csv_dicts(csv_path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        ids = []
        for f in ID_FIELDS:
            if f in row and row[f]:
                ids.append(row[f])
        # Also search any column that looks like a UID/name.
        for k, v in row.items():
            lk = k.lower()
            if v and ("uid" in lk or "cell" in lk or "profile" in lk):
                ids.append(v)
        for ident in ids:
            ident = canonicalize_id(ident)
            if ident:
                out[ident] = row
    return out


def semantic_for_record(rec: Dict[str, Any], sem_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    candidates = []
    for f in ID_FIELDS:
        if rec.get(f):
            candidates.append(canonicalize_id(rec.get(f)))
    # Softlabel dir often ends with Batch-X_battery-Y.
    for f in ["softlabel_dir", "softlabel_npz", "replay_npz"]:
        p = rec.get(f)
        if p:
            parts = str(p).replace("\\", "/").split("/")
            for part in parts:
                if "Batch-" in part and "battery-" in part:
                    candidates.append(part)
    for c in candidates:
        if c in sem_map:
            return sem_map[c]
    return {}


def get_branch(rec: Dict[str, Any], sem: Dict[str, str]) -> str:
    for source in [sem, rec]:
        for k in ["semantic_branch", "branch", "generator_branch", "source_branch", "semantics", "semantic"]:
            if source.get(k):
                return str(source.get(k))
    stage = str(rec.get("source_stage") or sem.get("source_stage") or "")
    if "P4D" in stage or "current" in stage:
        return "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"
    if "P0" in stage or "P3" in stage or "P4B" in stage or "RG" in stage:
        return "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH"
    return "UNKNOWN_OR_MIXED_BRANCH"


def get_protocol(rec: Dict[str, Any]) -> str:
    if rec.get("protocol"):
        return str(rec.get("protocol"))
    cid = str(rec.get("canonical_cell_uid") or rec.get("cell_uid") or "")
    for p in ["random_walk", "GEO", "R2.5", "R3", "3C", "2C"]:
        if p in cid:
            return p
    return "UNKNOWN"


def infer_batch_battery(rec: Dict[str, Any]) -> Tuple[str, str]:
    batch = str(rec.get("batch") or "")
    battery = str(rec.get("battery") or "")
    candidates = [
        str(rec.get("canonical_cell_uid") or ""),
        str(rec.get("cell_uid") or ""),
        str(rec.get("softlabel_dir") or ""),
        str(rec.get("softlabel_npz") or ""),
    ]
    for text in candidates:
        parts = text.replace("\\", "/").replace("_", "/").split("/")
        for part in parts:
            if part.startswith("Batch-"):
                batch = part
            if part.startswith("battery-"):
                battery = part
    return batch or "Batch-UNKNOWN", battery or "battery-UNKNOWN"


def find_key(files: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    s = set(files)
    for k in candidates:
        if k in s:
            return k
    # case-insensitive fallback
    lower = {k.lower(): k for k in s}
    for k in candidates:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def npz_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=True) as z:
            return list(z.files)
    except Exception:
        # Fallback to zip namelist if np.load metadata fails.
        try:
            with zipfile.ZipFile(path, "r") as zf:
                return [Path(n).stem for n in zf.namelist() if n.endswith(".npy")]
        except Exception:
            return []


def choose_obs_source(rec: Dict[str, Any], prefer: str = "softlabel") -> Tuple[Path, str]:
    soft = Path(str(rec.get("softlabel_npz") or ""))
    replay = Path(str(rec.get("replay_npz") or ""))
    if prefer == "replay" and replay.exists():
        return replay, "replay"
    if soft.exists():
        return soft, "softlabel"
    if replay.exists():
        return replay, "replay"
    return soft if str(soft) else replay, "missing"


def load_time_cycle_arrays(rec: Dict[str, Any], prefer: str = "softlabel") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    source_path, source_kind = choose_obs_source(rec, prefer=prefer)
    info = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "time_key": "",
        "cycle_key": "",
        "loaded_cycle_from": "",
    }
    if not source_path.exists():
        raise FileNotFoundError(f"No time/cycle source exists for record: {source_path}")

    with np.load(source_path, allow_pickle=True) as z:
        tkey = find_key(z.files, TIME_KEYS)
        if tkey is None:
            raise KeyError(f"No time key found in {source_path}; keys={z.files[:30]}")
        t = np.asarray(z[tkey]).reshape(-1)
        info["time_key"] = tkey
        ckey = find_key(z.files, CYCLE_KEYS)
        if ckey is not None:
            cycle = np.asarray(z[ckey]).reshape(-1)
            info["cycle_key"] = ckey
            info["loaded_cycle_from"] = source_kind
            if cycle.size == t.size:
                return t.astype(np.float64), cycle, info

    # Fallback: try replay cycle_id and map if same length or exact time.
    replay = Path(str(rec.get("replay_npz") or ""))
    if replay.exists() and replay != source_path:
        with np.load(replay, allow_pickle=True) as rz:
            rtkey = find_key(rz.files, TIME_KEYS)
            rckey = find_key(rz.files, CYCLE_KEYS)
            if rtkey and rckey:
                rt = np.asarray(rz[rtkey]).reshape(-1).astype(np.float64)
                rc = np.asarray(rz[rckey]).reshape(-1)
                if rt.size == t.size and np.nanmax(np.abs(rt - t.astype(np.float64))) < 1e-6:
                    info["cycle_key"] = rckey
                    info["loaded_cycle_from"] = "replay_same_grid"
                    return t.astype(np.float64), rc, info
                # nearest mapping for query time grid (only if needed)
                order = np.argsort(rt)
                sorted_t = rt[order]
                pos = np.searchsorted(sorted_t, t.astype(np.float64))
                pos0 = np.clip(pos - 1, 0, sorted_t.size - 1)
                pos1 = np.clip(pos, 0, sorted_t.size - 1)
                choose1 = np.abs(sorted_t[pos1] - t) < np.abs(sorted_t[pos0] - t)
                chosen = np.where(choose1, pos1, pos0)
                idx = order[chosen]
                median_err = float(np.median(np.abs(rt[idx] - t)))
                max_err = float(np.max(np.abs(rt[idx] - t)))
                info["cycle_key"] = rckey
                info["loaded_cycle_from"] = f"replay_nearest_median_err_s={median_err:.6g}_max_err_s={max_err:.6g}"
                return t.astype(np.float64), rc[idx], info

    raise KeyError(f"No usable cycle_id found for {source_path} or replay {replay}")


def cycle_range_string(values: Sequence[int]) -> str:
    vals = sorted(set(int(v) for v in values))
    if not vals:
        return ""
    out = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
        else:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = v
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


def finite_int_cycle(cycle: np.ndarray) -> np.ndarray:
    c = np.asarray(cycle)
    # Convert strings/floats safely.
    out = []
    for x in c:
        try:
            if isinstance(x, bytes):
                x = x.decode("utf-8", errors="replace")
            if str(x).strip() == "":
                out.append(-1)
            else:
                out.append(int(float(x)))
        except Exception:
            out.append(-1)
    return np.asarray(out, dtype=np.int64)


def allocate_counts(cycle_ids: np.ndarray, budget: int, min_per_selected_cycle: int = 1) -> Dict[int, int]:
    unique, counts = np.unique(cycle_ids[cycle_ids >= 0], return_counts=True)
    if unique.size == 0 or budget <= 0:
        return {}
    unique = unique.astype(np.int64)
    counts = counts.astype(np.int64)
    if budget >= int(unique.size) * min_per_selected_cycle:
        base = {int(c): min_per_selected_cycle for c in unique}
        remaining = int(budget - int(unique.size) * min_per_selected_cycle)
        if remaining > 0:
            weights = np.sqrt(counts.astype(np.float64))
            weights = weights / max(float(np.sum(weights)), 1e-30)
            raw = weights * remaining
            add = np.floor(raw).astype(int)
            for c, a in zip(unique, add):
                base[int(c)] += int(a)
            left = remaining - int(np.sum(add))
            if left > 0:
                order = np.argsort(-(raw - add))
                for j in order[:left]:
                    base[int(unique[j])] += 1
        return base
    # More cycles than budget: select cycle IDs uniformly across cycle axis.
    pos = np.linspace(0, unique.size - 1, budget)
    chosen_idx = sorted(set(int(round(x)) for x in pos))
    # If rounding duplicated, fill gaps deterministically.
    i = 0
    while len(chosen_idx) < budget and i < unique.size:
        if i not in chosen_idx:
            chosen_idx.append(i)
        i += 1
    chosen_idx = sorted(chosen_idx[:budget])
    return {int(unique[i]): 1 for i in chosen_idx}


def select_sample_indices(t: np.ndarray, cycle: np.ndarray, budget: int, seed: int) -> np.ndarray:
    n = int(t.size)
    if n <= 0:
        return np.array([], dtype=np.int64)
    if budget <= 0 or n <= budget:
        return np.arange(n, dtype=np.int64)
    cycle_int = finite_int_cycle(cycle)
    alloc = allocate_counts(cycle_int, budget)
    selected: List[int] = []
    for cyc, k in sorted(alloc.items(), key=lambda x: x[0]):
        idx = np.where(cycle_int == cyc)[0]
        if idx.size == 0:
            continue
        k = min(int(k), int(idx.size))
        if k <= 1:
            selected.append(int(idx[0]))
        else:
            local = np.linspace(0, idx.size - 1, k)
            selected.extend([int(idx[int(round(x))]) for x in local])
    selected = sorted(set(selected))
    # Top up if duplicate points made us short.
    if len(selected) < budget:
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n, dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        mask[selected] = False
        remaining = all_idx[mask]
        take = min(budget - len(selected), remaining.size)
        if take > 0:
            extra = rng.choice(remaining, size=take, replace=False)
            selected.extend([int(x) for x in extra])
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if selected.size > budget:
        # Deterministic downsample while preserving range.
        pos = np.linspace(0, selected.size - 1, budget)
        selected = selected[[int(round(x)) for x in pos]]
    return selected


def phase_counts(cycle_values: np.ndarray) -> Dict[str, int]:
    vals = np.asarray(cycle_values, dtype=np.int64)
    vals = vals[vals >= 0]
    if vals.size == 0:
        return {"early": 0, "middle": 0, "late": 0}
    mn, mx = int(np.min(vals)), int(np.max(vals))
    if mx == mn:
        return {"early": int(vals.size), "middle": 0, "late": 0}
    x = (vals - mn) / max(mx - mn, 1)
    return {
        "early": int(np.sum(x <= 1.0 / 3.0)),
        "middle": int(np.sum((x > 1.0 / 3.0) & (x <= 2.0 / 3.0))),
        "late": int(np.sum(x > 2.0 / 3.0)),
    }


def summarize_profile(rec: Dict[str, Any], sem_map: Dict[str, Dict[str, str]], args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sem = semantic_for_record(rec, sem_map)
    branch = get_branch(rec, sem)
    protocol = get_protocol(rec)
    batch, battery = infer_batch_battery(rec)
    canonical = canonicalize_id(rec.get("canonical_cell_uid") or rec.get("canonical_cell_id") or rec.get("cell_uid") or rec.get("cell_id") or f"{batch}_{protocol}_{battery}")
    cell_uid = canonicalize_id(rec.get("cell_uid") or rec.get("cell_id") or canonical)
    split = str(rec.get("split") or "UNKNOWN")
    soft_npz = Path(str(rec.get("softlabel_npz") or ""))
    replay_npz = Path(str(rec.get("replay_npz") or ""))

    base: Dict[str, Any] = {
        "split": split,
        "canonical_cell_uid": canonical,
        "cell_uid": cell_uid,
        "batch": batch,
        "battery": battery,
        "protocol": protocol,
        "semantic_branch": branch,
        "source_stage": str(rec.get("source_stage") or sem.get("source_stage") or ""),
        "softlabel_npz": str(soft_npz),
        "softlabel_exists": bool(soft_npz.exists()),
        "replay_npz": str(replay_npz),
        "replay_exists": bool(replay_npz.exists()),
        "status": "PASS",
        "error": "",
    }

    t, cyc_raw, source_info = load_time_cycle_arrays(rec, prefer=args.prefer_time_source)
    cyc = finite_int_cycle(cyc_raw)
    valid = np.isfinite(t) & (cyc >= 0)
    if not np.any(valid):
        raise ValueError("No valid finite time/cycle entries")
    t_valid = t[valid]
    cyc_valid = cyc[valid]
    original_indices = np.arange(t.size, dtype=np.int64)[valid]

    budget = int(args.max_time_points_per_profile)
    sel_local = select_sample_indices(t_valid, cyc_valid, budget=budget, seed=int(args.seed) + abs(hash(canonical)) % 100000)
    sel_global = original_indices[sel_local]
    sel_cycles = cyc_valid[sel_local]

    unique_cycles, cycle_counts = np.unique(cyc_valid, return_counts=True)
    selected_unique_cycles, selected_cycle_counts = np.unique(sel_cycles, return_counts=True)
    all_phase = phase_counts(unique_cycles)
    sampled_phase = phase_counts(sel_cycles)

    cycle_count_map = {int(c): int(n) for c, n in zip(unique_cycles, cycle_counts)}
    sample_count_map = {int(c): int(n) for c, n in zip(selected_unique_cycles, selected_cycle_counts)}

    profile_row = dict(base)
    profile_row.update({
        "time_source_kind": source_info.get("source_kind", ""),
        "time_source_path": source_info.get("source_path", ""),
        "time_key": source_info.get("time_key", ""),
        "cycle_key": source_info.get("cycle_key", ""),
        "loaded_cycle_from": source_info.get("loaded_cycle_from", ""),
        "n_time_points": int(t.size),
        "n_valid_time_cycle_points": int(t_valid.size),
        "time_start_s": float(np.nanmin(t_valid)),
        "time_end_s": float(np.nanmax(t_valid)),
        "cycle_count": int(unique_cycles.size),
        "cycle_min": int(np.min(unique_cycles)),
        "cycle_max": int(np.max(unique_cycles)),
        "cycle_ranges": cycle_range_string(unique_cycles),
        "sample_budget": budget,
        "sample_count": int(sel_global.size),
        "sampled_cycle_count": int(selected_unique_cycles.size),
        "sampled_cycle_coverage_fraction": float(selected_unique_cycles.size / max(unique_cycles.size, 1)),
        "sample_time_start_s": float(np.nanmin(t[sel_global])) if sel_global.size else None,
        "sample_time_end_s": float(np.nanmax(t[sel_global])) if sel_global.size else None,
        "all_phase_early_cycles_or_points": all_phase["early"],
        "all_phase_middle_cycles_or_points": all_phase["middle"],
        "all_phase_late_cycles_or_points": all_phase["late"],
        "sample_phase_early_points": sampled_phase["early"],
        "sample_phase_middle_points": sampled_phase["middle"],
        "sample_phase_late_points": sampled_phase["late"],
        "sample_phase_early_fraction": sampled_phase["early"] / max(int(sel_global.size), 1),
        "sample_phase_middle_fraction": sampled_phase["middle"] / max(int(sel_global.size), 1),
        "sample_phase_late_fraction": sampled_phase["late"] / max(int(sel_global.size), 1),
    })

    # Per-cycle coverage rows.
    cycle_rows: List[Dict[str, Any]] = []
    for c in unique_cycles:
        idx = np.where(cyc_valid == c)[0]
        if idx.size == 0:
            continue
        cycle_rows.append({
            **base,
            "cycle_id": int(c),
            "cycle_n_points": int(idx.size),
            "cycle_time_start_s": float(np.nanmin(t_valid[idx])),
            "cycle_time_end_s": float(np.nanmax(t_valid[idx])),
            "sample_count": int(sample_count_map.get(int(c), 0)),
            "sampled": bool(sample_count_map.get(int(c), 0) > 0),
        })

    # Per-sample rows. Keep minimal so CSV remains usable.
    sample_rows: List[Dict[str, Any]] = []
    if args.write_sample_points:
        for j, gi in enumerate(sel_global):
            sample_rows.append({
                **base,
                "sample_order": int(j),
                "source_index": int(gi),
                "t_global_s": float(t[gi]),
                "cycle_id": int(cyc[gi]),
            })

    return profile_row, cycle_rows, sample_rows


def counts_by(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    d: Dict[Tuple[Any, ...], int] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in keys)
        d[key] = d.get(key, 0) + 1
    out = []
    for key, n in sorted(d.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["count"] = n
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G7-S0 full-cycle sampling / data coverage audit")
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,validation", help="Comma-separated splits or 'all'. Default: train,validation")
    ap.add_argument("--profile_limit", type=int, default=0, help="Limit number of profiles after split filtering; 0 = no limit")
    ap.add_argument("--max_time_points_per_profile", type=int, default=4096)
    ap.add_argument("--prefer_time_source", choices=["softlabel", "replay"], default="softlabel")
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--write_sample_points", action="store_true", help="Write D17_G7S0_SAMPLE_POINTS.csv. Recommended for S1.")
    ap.add_argument("--min_cycle_coverage_fraction", type=float, default=0.95)
    ap.add_argument("--min_phase_fraction", type=float, default=0.05)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest)
    records, manifest = load_split_manifest(manifest_path)
    sem_map = build_semantics_map(Path(args.g0_profile_semantics_csv) if args.g0_profile_semantics_csv else None)
    allowed_splits = set(parse_splits(args.splits))
    filtered = [r for r in records if str(r.get("split") or "") in allowed_splits]
    if args.profile_limit and args.profile_limit > 0:
        filtered = filtered[: int(args.profile_limit)]

    profile_rows: List[Dict[str, Any]] = []
    cycle_rows_all: List[Dict[str, Any]] = []
    sample_rows_all: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, rec in enumerate(filtered):
        try:
            pr, cr, sr = summarize_profile(rec, sem_map, args)
            pr["profile_order"] = i
            profile_rows.append(pr)
            cycle_rows_all.extend(cr)
            sample_rows_all.extend(sr)
            print(json.dumps({
                "profile_order": i,
                "status": "PASS",
                "canonical_cell_uid": pr.get("canonical_cell_uid"),
                "split": pr.get("split"),
                "protocol": pr.get("protocol"),
                "branch": pr.get("semantic_branch"),
                "cycle_count": pr.get("cycle_count"),
                "sample_count": pr.get("sample_count"),
                "coverage": pr.get("sampled_cycle_coverage_fraction"),
            }, ensure_ascii=False), flush=True)
        except Exception as e:
            batch, battery = infer_batch_battery(rec)
            canonical = canonicalize_id(rec.get("canonical_cell_uid") or rec.get("cell_uid") or f"{batch}_{battery}")
            row = {
                "profile_order": i,
                "split": str(rec.get("split") or ""),
                "canonical_cell_uid": canonical,
                "batch": batch,
                "battery": battery,
                "protocol": get_protocol(rec),
                "status": "FAIL",
                "error": f"{type(e).__name__}: {e}",
                "softlabel_npz": str(rec.get("softlabel_npz") or ""),
                "replay_npz": str(rec.get("replay_npz") or ""),
            }
            failures.append(row)
            profile_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    # Gates.
    blocker_list: List[str] = []
    if failures:
        blocker_list.append(f"profile load failures: {len(failures)}")

    ok_rows = [r for r in profile_rows if r.get("status") == "PASS"]
    if ok_rows:
        min_cycle_cov = min(float(r.get("sampled_cycle_coverage_fraction") or 0.0) for r in ok_rows)
        min_early = min(float(r.get("sample_phase_early_fraction") or 0.0) for r in ok_rows)
        min_middle = min(float(r.get("sample_phase_middle_fraction") or 0.0) for r in ok_rows)
        min_late = min(float(r.get("sample_phase_late_fraction") or 0.0) for r in ok_rows)
        if min_cycle_cov < args.min_cycle_coverage_fraction:
            blocker_list.append(f"min sampled cycle coverage below gate {args.min_cycle_coverage_fraction}: {min_cycle_cov:.6g}")
        if min(min_early, min_middle, min_late) < args.min_phase_fraction:
            blocker_list.append(
                f"min early/middle/late sample phase fraction below gate {args.min_phase_fraction}: "
                f"early={min_early:.6g}, middle={min_middle:.6g}, late={min_late:.6g}"
            )
    else:
        blocker_list.append("no profiles successfully audited")
        min_cycle_cov = None
        min_early = min_middle = min_late = None

    status = "PASS" if not failures else "REVIEW"
    s1_ready = (len(blocker_list) == 0)
    recommendation = "ENTER_G7_S1_SMALL_FULLCYCLE_SMOKE" if s1_ready else "REVIEW_S0_COVERAGE_BEFORE_S1"

    split_counts = counts_by(profile_rows, ["split"])
    split_protocol_counts = counts_by(profile_rows, ["split", "protocol"])
    split_branch_counts = counts_by(profile_rows, ["split", "semantic_branch"])
    split_protocol_branch_counts = counts_by(profile_rows, ["split", "protocol", "semantic_branch"])

    profile_csv = out_dir / "D17_G7S0_PROFILE_COVERAGE.csv"
    cycle_csv = out_dir / "D17_G7S0_CYCLE_COVERAGE.csv"
    sample_csv = out_dir / "D17_G7S0_SAMPLE_POINTS.csv"
    counts_csv = out_dir / "D17_G7S0_SPLIT_PROTOCOL_BRANCH_COUNTS.csv"
    failures_csv = out_dir / "D17_G7S0_LOAD_FAILURES.csv"
    s1_config_json = out_dir / "D17_G7S0_RECOMMENDED_S1_CONFIG.json"
    summary_json = out_dir / "D17_G7S0_FULLCYCLE_SAMPLING_AUDIT_SUMMARY.json"

    write_csv(profile_csv, profile_rows)
    write_csv(cycle_csv, cycle_rows_all)
    if args.write_sample_points:
        write_csv(sample_csv, sample_rows_all)
    else:
        # Still create a placeholder with instructions.
        write_csv(sample_csv, [{"note": "sample points not written; rerun with --write_sample_points for S1"}])
    write_csv(failures_csv, failures)
    count_rows = []
    for name, rows in [
        ("split", split_counts),
        ("split_protocol", split_protocol_counts),
        ("split_branch", split_branch_counts),
        ("split_protocol_branch", split_protocol_branch_counts),
    ]:
        for r in rows:
            rr = dict(r)
            rr["grouping"] = name
            count_rows.append(rr)
    write_csv(counts_csv, count_rows)

    s1_cfg = {
        "protocol": "D17-G7-S1_SMALL_FULLCYCLE_SMOKE_RECOMMENDED_CONFIG_FROM_S0",
        "seed": int(args.seed),
        "max_time_points_per_profile": int(args.max_time_points_per_profile),
        "sampling_plan_csv": str(sample_csv),
        "profile_coverage_csv": str(profile_csv),
        "cycle_coverage_csv": str(cycle_csv),
        "recommended_train_profile_count": 8,
        "recommended_validation_profile_count": 2,
        "recommended_epochs": 150,
        "recommended_stop_rule": "Run selected-cycle dense audit after S1; do not scale to S2 unless selected-cycle metrics recover.",
        "notes": [
            "S0 performs no training and no checkpoint selection.",
            "The plan covers full time/cycle span by stratified per-cycle sampling.",
            "If S0 blockers are present, fix data paths/cycle coverage before S1.",
        ],
    }
    write_json(s1_config_json, s1_cfg)

    manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else ""
    summary = {
        "protocol": "D17-G7-S0_FULLCYCLE_SAMPLING_DATA_COVERAGE_AUDIT",
        "created_at_unix": time.time(),
        "status": status,
        "s1_ready": s1_ready,
        "recommendation": recommendation,
        "blockers": blocker_list,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "state_arrays_loaded": False,
        "time_cycle_arrays_loaded_only": True,
        "manifest_hash_sha256": manifest_hash,
        "split_manifest": str(manifest_path),
        "g0_profile_semantics_csv": str(args.g0_profile_semantics_csv),
        "splits_requested": sorted(allowed_splits),
        "profile_count_requested": len(filtered),
        "profile_count_pass": len(ok_rows),
        "profile_count_fail": len(failures),
        "max_time_points_per_profile": int(args.max_time_points_per_profile),
        "write_sample_points": bool(args.write_sample_points),
        "coverage_gate": {
            "min_cycle_coverage_fraction_gate": float(args.min_cycle_coverage_fraction),
            "min_phase_fraction_gate": float(args.min_phase_fraction),
            "observed_min_cycle_coverage_fraction": min_cycle_cov,
            "observed_min_early_fraction": min_early,
            "observed_min_middle_fraction": min_middle,
            "observed_min_late_fraction": min_late,
        },
        "counts": {
            "split": split_counts,
            "split_protocol": split_protocol_counts,
            "split_branch": split_branch_counts,
            "split_protocol_branch": split_protocol_branch_counts,
        },
        "files": {
            "summary_json": str(summary_json),
            "profile_coverage_csv": str(profile_csv),
            "cycle_coverage_csv": str(cycle_csv),
            "sample_points_csv": str(sample_csv),
            "split_protocol_branch_counts_csv": str(counts_csv),
            "load_failures_csv": str(failures_csv),
            "recommended_s1_config_json": str(s1_config_json),
        },
        "elapsed_s": time.time() - t0,
    }
    write_json(summary_json, summary)

    print(json.dumps({
        "status": status,
        "s1_ready": s1_ready,
        "recommendation": recommendation,
        "blockers": blocker_list,
        "profile_count_pass": len(ok_rows),
        "profile_count_fail": len(failures),
        "summary_json": str(summary_json),
        "profile_coverage_csv": str(profile_csv),
        "cycle_coverage_csv": str(cycle_csv),
        "sample_points_csv": str(sample_csv),
        "elapsed_s": summary["elapsed_s"],
    }, ensure_ascii=False, indent=2))
    return 0 if status in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
