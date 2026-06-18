#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D17-G interactive 3D concentration surface plotter.

Purpose
-------
Plot PINN/surrogate predictions and P2Dlite-RG soft-label truth for the two
solid concentration targets, cs_a and cs_c, in interactive Matplotlib 3D
windows.  The script is intentionally read-only: it does not train, overwrite,
or modify any project artifact.

Typical use
-----------
python scripts/d17_g_plot_cs3d_interactive.py ^
  --batch 2 --battery 3 --cycles 13-15 --target both ^
  --pred_root "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g" ^
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" ^
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json"

Notes
-----
- Prediction files are searched recursively under --pred_root.  For exact
  control, pass --pred_npz directly.
- If the prediction npz already contains *_true_report_only arrays, those are
  used as the soft-label truth to guarantee identical time-grid sampling.
- If cycle_id is absent from the prediction npz, cycle labels are aligned from
  the soft-label or replay profile using nearest-time matching.  The script now
  prints detailed diagnostics when the requested cycles are not present in the
  saved prediction NPZ time grid, instead of failing with an opaque error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Matplotlib is imported after optional backend parsing in main().

PROTOCOL_BY_BATCH = {
    1: "2C",
    2: "3C",
    3: "R2.5",
    4: "R3",
    5: "random_walk",
    6: "GEO",
}

TIME_KEYS = [
    "t_global_s", "time_global_s", "t_s", "time_s", "time", "t", "Time", "Time_s"
]
CYCLE_KEYS = [
    "cycle_id", "cycle", "cycle_index", "cycle_number", "cycles", "Cycle", "cycle_ids"
]
R_KEYS_COMMON = ["r", "r_grid", "r_nodes", "r_norm", "r_over_R"]
R_KEYS_BY_TARGET = {
    "cs_a": ["r_a", "r_grid_a", "r_nodes_a", "r_a_m", "r_n", "r_neg", "r_negative"],
    "cs_c": ["r_c", "r_grid_c", "r_nodes_c", "r_c_m", "r_p", "r_pos", "r_positive"],
}


@dataclass
class ProfilePaths:
    batch_n: int
    battery_n: int
    protocol: str
    canonical_uid: str
    softlabel_npz: Optional[Path] = None
    replay_npz: Optional[Path] = None
    pred_npz: Optional[Path] = None


def _json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _parse_int_from_token(x: str, prefix: str) -> int:
    s = str(x).strip()
    m = re.search(rf"{re.escape(prefix)}[-_ ]*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse integer from {prefix} argument: {x!r}")


def parse_batch(x: str) -> int:
    return _parse_int_from_token(x, "batch")


def parse_battery(x: str) -> int:
    return _parse_int_from_token(x, "battery")


def canonical_candidates(batch_n: int, battery_n: int) -> List[str]:
    protocol = PROTOCOL_BY_BATCH.get(batch_n, "UNKNOWN")
    return [
        f"Batch-{batch_n}_{protocol}_battery-{battery_n}",
        f"Batch-{batch_n}_battery-{battery_n}",
        f"Batch{batch_n}_{protocol}_battery{battery_n}",
        f"batch{batch_n}_{protocol}_battery{battery_n}",
    ]


def parse_cycles(s: str) -> Optional[List[int]]:
    text = str(s).strip().lower().replace("cycle", "").replace("cycles", "")
    if text in {"", "all", "none", "*"}:
        return None
    out: List[int] = []
    for part in re.split(r"[,; ]+", text):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ia, ib = int(a), int(b)
            if ib < ia:
                ia, ib = ib, ia
            out.extend(list(range(ia, ib + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _safe_str(x: Any) -> str:
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0].item())
    except Exception:
        pass
    return str(x)


def _find_first_key(z: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in z:
            return k
    return None


def _to_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"{name} is not numeric: dtype={arr.dtype}")
    return arr.astype(float).reshape(-1)


def load_npz_keys(path: str | Path, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        use = list(z.files) if keys is None else [k for k in keys if k in z.files]
        for k in use:
            out[k] = z[k]
        out["__keys__"] = np.array(list(z.files), dtype=object)
    return out


def npz_key_list(path: str | Path) -> List[str]:
    with np.load(path, allow_pickle=True) as z:
        return list(z.files)


def orient_time_radial(x: Any, n_time: Optional[int], name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"{name} is not numeric: dtype={arr.dtype}")
    arr = arr.astype(float)
    if arr.ndim == 1:
        if n_time is not None and arr.shape[0] != n_time:
            raise ValueError(f"{name}: 1D length {arr.shape[0]} != n_time {n_time}")
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 1D or 2D, got {arr.shape}")
    if n_time is None:
        # Prefer time x radial layout when one dimension looks like a small radial grid.
        if arr.shape[1] <= 256:
            return arr
        if arr.shape[0] <= 256:
            return arr.T
        return arr
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f"{name}: cannot orient shape {arr.shape} for n_time={n_time}")


def choose_time_array(pred: Mapping[str, Any], soft: Optional[Mapping[str, Any]] = None) -> Tuple[str, np.ndarray]:
    k = _find_first_key(pred, TIME_KEYS)
    if k is not None:
        return k, _to_1d(pred[k], k)
    if soft is not None:
        k = _find_first_key(soft, TIME_KEYS)
        if k is not None:
            return k, _to_1d(soft[k], k)
    raise KeyError(f"No time key found. Tried {TIME_KEYS}")


def load_split_records(path: Optional[str | Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".json":
        data = _json_load(p)
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return [dict(r) for r in data["records"]]
        if isinstance(data, list):
            return [dict(r) for r in data]
    if p.suffix.lower() == ".csv":
        return _read_csv_dicts(p)
    return []


def record_matches(record: Mapping[str, Any], batch_n: int, battery_n: int) -> bool:
    vals = " ".join(str(record.get(k, "")) for k in ["batch", "battery", "cell_uid", "canonical_cell_uid", "softlabel_dir", "softlabel_npz"])
    batch_ok = re.search(rf"Batch[-_ ]?{batch_n}\b", vals, flags=re.IGNORECASE) is not None
    batt_ok = re.search(rf"battery[-_ ]?{battery_n}\b", vals, flags=re.IGNORECASE) is not None
    return bool(batch_ok and batt_ok)


def resolve_paths(
    batch_arg: str,
    battery_arg: str,
    split_manifest: Optional[str | Path],
    softlabel_root: Optional[str | Path],
    pred_root: Optional[str | Path],
    pred_npz: Optional[str | Path],
    softlabel_npz: Optional[str | Path],
    replay_npz: Optional[str | Path],
) -> ProfilePaths:
    batch_n = parse_batch(batch_arg)
    battery_n = parse_battery(battery_arg)
    protocol = PROTOCOL_BY_BATCH.get(batch_n, "UNKNOWN")
    canonical_uid = f"Batch-{batch_n}_{protocol}_battery-{battery_n}"
    pp = ProfilePaths(batch_n=batch_n, battery_n=battery_n, protocol=protocol, canonical_uid=canonical_uid)

    if split_manifest:
        records = load_split_records(split_manifest)
        for r in records:
            if record_matches(r, batch_n, battery_n):
                pp.canonical_uid = str(r.get("canonical_cell_uid") or r.get("cell_uid") or canonical_uid)
                if r.get("protocol"):
                    pp.protocol = str(r.get("protocol"))
                if r.get("softlabel_npz"):
                    p = Path(str(r["softlabel_npz"]))
                    if p.exists():
                        pp.softlabel_npz = p
                if r.get("replay_npz"):
                    p = Path(str(r["replay_npz"]))
                    if p.exists():
                        pp.replay_npz = p
                break

    if softlabel_npz:
        p = Path(softlabel_npz)
        if not p.exists():
            raise FileNotFoundError(f"--softlabel_npz not found: {p}")
        pp.softlabel_npz = p
    elif pp.softlabel_npz is None and softlabel_root:
        root = Path(softlabel_root)
        for cand in canonical_candidates(batch_n, battery_n):
            for rel in [
                Path("profiles") / cand / "solution_softlabels.npz",
                Path(cand) / "solution_softlabels.npz",
            ]:
                p = root / rel
                if p.exists():
                    pp.softlabel_npz = p
                    break
            if pp.softlabel_npz:
                break

    if replay_npz:
        p = Path(replay_npz)
        if not p.exists():
            raise FileNotFoundError(f"--replay_npz not found: {p}")
        pp.replay_npz = p

    if pred_npz:
        p = Path(pred_npz)
        if not p.exists():
            raise FileNotFoundError(f"--pred_npz not found: {p}")
        pp.pred_npz = p
    elif pred_root:
        pp.pred_npz = find_prediction_npz(Path(pred_root), batch_n, battery_n, pp.canonical_uid)

    return pp


def find_prediction_npz(root: Path, batch_n: int, battery_n: int, canonical_uid: str) -> Optional[Path]:
    if not root.exists():
        return None
    tokens = [
        f"Batch-{batch_n}_battery-{battery_n}",
        f"Batch-{batch_n}_{PROTOCOL_BY_BATCH.get(batch_n, '')}_battery-{battery_n}",
        canonical_uid,
    ]
    pred_files = list(root.rglob("*.npz"))
    # Search filenames first.
    scored: List[Tuple[int, Path]] = []
    for p in pred_files:
        s = str(p)
        if "PRED" not in p.name.upper() and "pred" not in s.lower() and "prediction" not in s.lower():
            continue
        score = 0
        for t in tokens:
            if t and t in s:
                score += 10
        if re.search(rf"Batch[-_ ]?{batch_n}.*battery[-_ ]?{battery_n}", s, flags=re.IGNORECASE):
            score += 5
        if score:
            scored.append((score, p))
    if scored:
        scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
        return scored[0][1]
    # Fall back to reading tiny scalar identity fields from candidate npz files.
    for p in pred_files:
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = set(z.files)
                if not ({"canonical_cell_uid", "cell_uid"} & keys):
                    continue
                uid = _safe_str(z["canonical_cell_uid"] if "canonical_cell_uid" in keys else z["cell_uid"])
                if re.search(rf"Batch[-_ ]?{batch_n}.*battery[-_ ]?{battery_n}", uid, flags=re.IGNORECASE):
                    return p
        except Exception:
            continue
    return None



def _nearest_cycle_from_time(
    query_t: np.ndarray,
    source_t: np.ndarray,
    source_cycle: np.ndarray,
    source_name: str,
    allow_offset: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Map cycle IDs from a full source time grid to a queried time grid.

    The D17-G prediction npz may store a 512-point target grid, while replay or
    soft-label files can contain the full profile.  This helper maps cycle IDs
    by nearest time, with a conservative constant-offset fallback for cases
    where one file stores profile-relative time and the other stores global time.
    """
    info: Dict[str, Any] = {"source": source_name, "status": "untried"}
    qt = np.asarray(query_t, dtype=float).reshape(-1)
    st = np.asarray(source_t, dtype=float).reshape(-1)
    sc = np.asarray(source_cycle).reshape(-1).astype(int)
    if qt.size == 0 or st.size == 0 or st.size != sc.size:
        info.update({"status": "invalid", "query_n": int(qt.size), "source_n": int(st.size), "cycle_n": int(sc.size)})
        return None, info
    order = np.argsort(st)
    st = st[order]
    sc = sc[order]

    candidates: List[Tuple[str, np.ndarray]] = [("as_is", qt)]
    if allow_offset:
        # Common cases: prediction stores t relative to profile/window start,
        # while replay/softlabel stores global time; or vice versa.
        candidates.append(("shift_query_to_source_start", qt + (float(st[0]) - float(qt[0]))))
        candidates.append(("shift_query_to_source_end", qt + (float(st[-1]) - float(qt[-1]))))

    best: Optional[Tuple[str, np.ndarray, float, float]] = None
    for mode, q in candidates:
        idx = np.searchsorted(st, q, side="left")
        idx = np.clip(idx, 0, st.size - 1)
        left = np.clip(idx - 1, 0, st.size - 1)
        choose_left = np.abs(q - st[left]) < np.abs(q - st[idx])
        idx[choose_left] = left[choose_left]
        median_dt = float(np.nanmedian(np.abs(q - st[idx]))) if idx.size else float("inf")
        span_overlap = float(max(0.0, min(float(np.nanmax(q)), float(st[-1])) - max(float(np.nanmin(q)), float(st[0]))))
        if best is None or (span_overlap, -median_dt) > (best[3], -best[2]):
            best = (mode, sc[idx], median_dt, span_overlap)
    assert best is not None
    mode, mapped, median_dt, span_overlap = best
    info.update({
        "status": "mapped",
        "mode": mode,
        "median_abs_time_error_s": median_dt,
        "span_overlap_s": span_overlap,
        "query_time_range": [float(np.nanmin(qt)), float(np.nanmax(qt))],
        "source_time_range": [float(st[0]), float(st[-1])],
        "mapped_available_cycles": compact_cycle_ranges(np.unique(mapped).astype(int).tolist()),
    })
    return mapped.astype(int), info


def get_full_cycle_diagnostics(pp: ProfilePaths) -> Dict[str, Any]:
    """Read only time/cycle arrays from replay and softlabel for diagnostics."""
    out: Dict[str, Any] = {}
    for name, path in [("replay", pp.replay_npz), ("softlabel", pp.softlabel_npz)]:
        d: Dict[str, Any] = {"path": str(path) if path else "", "available": False}
        if path and Path(path).exists():
            try:
                arr = load_npz_keys(path, TIME_KEYS + CYCLE_KEYS)
                tk = _find_first_key(arr, TIME_KEYS)
                ck = _find_first_key(arr, CYCLE_KEYS)
                if tk and ck:
                    t = _to_1d(arr[tk], tk)
                    c = _to_1d(arr[ck], ck).astype(int)
                    if t.size == c.size and t.size > 0:
                        d.update({
                            "available": True,
                            "time_key": tk,
                            "cycle_key": ck,
                            "n": int(t.size),
                            "time_range": [float(np.nanmin(t)), float(np.nanmax(t))],
                            "cycle_ranges": compact_cycle_ranges(np.unique(c).astype(int).tolist()),
                        })
            except Exception as e:
                d["error"] = repr(e)
        out[name] = d
    return out


def get_requested_cycle_time_window(pp: ProfilePaths, requested_cycles: List[int]) -> Optional[Dict[str, Any]]:
    """Return time windows for requested cycles from full replay/softlabel grids."""
    req = set(int(x) for x in requested_cycles)
    candidates: List[Dict[str, Any]] = []
    for name, path in [("replay", pp.replay_npz), ("softlabel", pp.softlabel_npz)]:
        if not path or not Path(path).exists():
            continue
        try:
            arr = load_npz_keys(path, TIME_KEYS + CYCLE_KEYS)
            tk = _find_first_key(arr, TIME_KEYS)
            ck = _find_first_key(arr, CYCLE_KEYS)
            if not tk or not ck:
                continue
            t = _to_1d(arr[tk], tk)
            c = _to_1d(arr[ck], ck).astype(int)
            if t.size != c.size:
                continue
            mask = np.isin(c, list(req))
            if np.any(mask):
                candidates.append({
                    "source": name,
                    "time_key": tk,
                    "cycle_key": ck,
                    "n_points": int(np.sum(mask)),
                    "time_min": float(np.nanmin(t[mask])),
                    "time_max": float(np.nanmax(t[mask])),
                    "cycles_found": compact_cycle_ranges(np.unique(c[mask]).astype(int).tolist()),
                })
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda d: (-int(d.get("n_points", 0)), str(d.get("source", ""))))
    return candidates[0]


def compact_cycle_ranges(vals: Sequence[int]) -> str:
    xs = sorted(set(int(v) for v in vals))
    if not xs:
        return ""
    ranges: List[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = x
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def get_cycle_ids_for_time(
    t: np.ndarray,
    pred: Mapping[str, Any],
    soft: Optional[Mapping[str, Any]],
    replay_npz: Optional[Path],
    pp: Optional[ProfilePaths] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"sources": []}
    # 1) Directly aligned cycle arrays in prediction or soft-label data.
    for source_name, src in [("prediction_npz", pred), ("softlabel_loaded", soft or {})]:
        k = _find_first_key(src, CYCLE_KEYS)
        if k is not None:
            try:
                c = _to_1d(src[k], k)
                diagnostics["sources"].append({"source": source_name, "cycle_key": k, "cycle_n": int(c.size)})
                if c.size == t.size:
                    mapped = c.astype(int)
                    diagnostics.update({
                        "chosen_source": source_name,
                        "chosen_mode": "direct_same_length",
                        "available_cycles": compact_cycle_ranges(np.unique(mapped).astype(int).tolist()),
                    })
                    return mapped, diagnostics
            except Exception as e:
                diagnostics["sources"].append({"source": source_name, "cycle_key": k, "error": repr(e)})

    # 2) Time-map full softlabel cycle_id to prediction grid if possible.
    if soft is not None:
        tk = _find_first_key(soft, TIME_KEYS)
        ck = _find_first_key(soft, CYCLE_KEYS)
        if tk and ck:
            try:
                mapped, info = _nearest_cycle_from_time(t, _to_1d(soft[tk], tk), _to_1d(soft[ck], ck), "softlabel_npz")
                diagnostics["sources"].append(info)
                if mapped is not None:
                    diagnostics.update({
                        "chosen_source": "softlabel_npz",
                        "chosen_mode": info.get("mode"),
                        "available_cycles": compact_cycle_ranges(np.unique(mapped).astype(int).tolist()),
                    })
                    return mapped, diagnostics
            except Exception as e:
                diagnostics["sources"].append({"source": "softlabel_npz", "error": repr(e)})

    # 3) Time-map replay cycle_id to prediction grid.
    if replay_npz and replay_npz.exists():
        keys = TIME_KEYS + CYCLE_KEYS
        rep = load_npz_keys(replay_npz, keys)
        tk = _find_first_key(rep, TIME_KEYS)
        ck = _find_first_key(rep, CYCLE_KEYS)
        if tk and ck:
            try:
                mapped, info = _nearest_cycle_from_time(t, _to_1d(rep[tk], tk), _to_1d(rep[ck], ck), "replay_npz")
                diagnostics["sources"].append(info)
                if mapped is not None:
                    diagnostics.update({
                        "chosen_source": "replay_npz",
                        "chosen_mode": info.get("mode"),
                        "available_cycles": compact_cycle_ranges(np.unique(mapped).astype(int).tolist()),
                    })
                    return mapped, diagnostics
            except Exception as e:
                diagnostics["sources"].append({"source": "replay_npz", "error": repr(e)})
    diagnostics["available_cycles"] = ""
    return None, diagnostics


def select_by_cycles(t: np.ndarray, cycle_ids: Optional[np.ndarray], cycles: Optional[List[int]]) -> np.ndarray:
    if cycles is None:
        return np.ones(t.shape[0], dtype=bool)
    if cycle_ids is None:
        raise ValueError(
            "Cycles were requested, but no cycle_id/cycle array could be found in prediction, softlabel, or replay npz. "
            "Pass --replay_npz explicitly or use --cycles all."
        )
    return np.isin(cycle_ids.astype(int), np.asarray(cycles, dtype=int))


def select_by_cycles_auto(
    t: np.ndarray,
    cycle_ids: Optional[np.ndarray],
    cycles: Optional[List[int]],
    args: argparse.Namespace,
    pp: ProfilePaths,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cycle selection with off-by-one and diagnostic handling."""
    info: Dict[str, Any] = {"requested_cycles": cycles, "mode": "all" if cycles is None else "exact"}
    if cycles is None:
        return np.ones(t.shape[0], dtype=bool), info
    if cycle_ids is None:
        info["error"] = "no_cycle_ids_on_prediction_grid"
        return np.zeros(t.shape[0], dtype=bool), info
    cids = cycle_ids.astype(int)
    req = np.asarray(cycles, dtype=int)
    exact = np.isin(cids, req)
    if np.any(exact):
        info.update({"mode": "exact", "selected_cycles_on_prediction_grid": compact_cycle_ranges(np.unique(cids[exact]).astype(int).tolist())})
        return exact, info
    # Optional automatic off-by-one recovery for 0-based vs 1-based cycle ids.
    if str(getattr(args, "cycle_index_base", "auto")).lower() in {"auto", "zero_based"}:
        shifted = req - 1
        mask = np.isin(cids, shifted)
        if np.any(mask):
            info.update({
                "mode": "auto_zero_based_shift_requested_minus_1",
                "note": "Requested cycles were not present, but requested-1 cycles were found. This is likely a 0-based cycle_id convention.",
                "selected_cycles_on_prediction_grid": compact_cycle_ranges(np.unique(cids[mask]).astype(int).tolist()),
            })
            return mask, info
    if str(getattr(args, "cycle_index_base", "auto")).lower() in {"auto", "one_based"}:
        shifted = req + 1
        mask = np.isin(cids, shifted)
        if np.any(mask):
            info.update({
                "mode": "auto_one_based_shift_requested_plus_1",
                "note": "Requested cycles were not present, but requested+1 cycles were found. This may indicate an offset convention.",
                "selected_cycles_on_prediction_grid": compact_cycle_ranges(np.unique(cids[mask]).astype(int).tolist()),
            })
            return mask, info
    # If requested cycles exist in full replay/softlabel but not prediction grid,
    # report this explicitly. It usually means the saved prediction NPZ was only
    # generated for another time window, e.g. first 40 ks.
    full_window = get_requested_cycle_time_window(pp, list(req))
    info.update({
        "mode": "empty",
        "available_cycles_on_prediction_grid": compact_cycle_ranges(np.unique(cids).astype(int).tolist()),
        "requested_cycle_time_window_from_full_sources": full_window,
        "full_cycle_diagnostics": get_full_cycle_diagnostics(pp),
    })
    return np.zeros(t.shape[0], dtype=bool), info

def get_r_grid(data: Mapping[str, Any], target: str, n_r: int, normalize: bool = True) -> Tuple[np.ndarray, str]:
    keys = R_KEYS_BY_TARGET.get(target, []) + R_KEYS_COMMON
    for k in keys:
        if k in data:
            try:
                r = _to_1d(data[k], k)
                if r.size == n_r:
                    label = "r"
                    if normalize and np.nanmax(np.abs(r)) > 0:
                        # If r is in meters, a normalized r/R axis is more readable.
                        if np.nanmax(np.abs(r)) < 1e-2:
                            r = r / np.nanmax(np.abs(r))
                            label = "r/R (-)"
                        else:
                            label = "r"
                    return r.astype(float), label
            except Exception:
                pass
    return np.linspace(0.0, 1.0, n_r), "r/R (-)"


def load_target_arrays(
    pp: ProfilePaths,
    target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], str, str, Dict[str, Any]]:
    if pp.pred_npz is None:
        raise FileNotFoundError(
            "No prediction npz was found. Provide --pred_npz directly or set --pred_root to a directory containing D17-G prediction npz files."
        )
    pred_keys = npz_key_list(pp.pred_npz)
    want_pred = [f"{target}_pred", f"{target}_prediction", target]
    want_true = [f"{target}_true_report_only", f"{target}_true", f"{target}_soft", f"{target}_softlabel"]
    pred_load_keys = list(set(TIME_KEYS + CYCLE_KEYS + R_KEYS_COMMON + R_KEYS_BY_TARGET.get(target, []) + want_pred + want_true + ["canonical_cell_uid", "cell_uid", "protocol", "semantic_branch"]))
    pred = load_npz_keys(pp.pred_npz, pred_load_keys)

    soft: Optional[Dict[str, Any]] = None
    if pp.softlabel_npz and pp.softlabel_npz.exists():
        soft_load_keys = list(set(TIME_KEYS + CYCLE_KEYS + R_KEYS_COMMON + R_KEYS_BY_TARGET.get(target, []) + [target]))
        try:
            soft = load_npz_keys(pp.softlabel_npz, soft_load_keys)
        except Exception:
            soft = None

    _, t = choose_time_array(pred, soft)
    n_time = int(t.size)

    pk = _find_first_key(pred, want_pred)
    if pk is None:
        raise KeyError(f"Prediction file {pp.pred_npz} has no {target}_pred-like key. Keys include: {pred_keys[:30]} ...")
    z_pred = orient_time_radial(pred[pk], n_time, pk)

    tk = _find_first_key(pred, want_true)
    true_source = "prediction_npz:true_report_only"
    if tk is not None:
        z_true = orient_time_radial(pred[tk], n_time, tk)
    else:
        if soft is None or target not in soft:
            raise KeyError(f"No truth array for {target}. Provide --softlabel_npz/root or use prediction npz with {target}_true_report_only.")
        soft_t_key = _find_first_key(soft, TIME_KEYS)
        z_soft = orient_time_radial(soft[target], None, target)
        if soft_t_key is None or z_soft.shape[0] == n_time:
            z_true = z_soft if z_soft.shape[0] == n_time else z_soft[:n_time]
        else:
            soft_t = _to_1d(soft[soft_t_key], soft_t_key)
            idx = np.searchsorted(soft_t, t, side="left")
            idx = np.clip(idx, 0, soft_t.size - 1)
            left = np.clip(idx - 1, 0, soft_t.size - 1)
            choose_left = np.abs(t - soft_t[left]) < np.abs(t - soft_t[idx])
            idx[choose_left] = left[choose_left]
            z_true = z_soft[idx]
        true_source = "softlabel_npz"

    if z_pred.shape != z_true.shape:
        raise ValueError(f"Prediction/truth shape mismatch for {target}: pred={z_pred.shape}, truth={z_true.shape}")
    r, r_label = get_r_grid(soft or pred, target, z_pred.shape[1], normalize=True)
    cycle_ids, cycle_diag = get_cycle_ids_for_time(t, pred, soft, pp.replay_npz, pp)
    meta = {
        "prediction_key": pk,
        "truth_key": tk or target,
        "truth_source": true_source,
        "pred_npz": str(pp.pred_npz),
        "softlabel_npz": str(pp.softlabel_npz) if pp.softlabel_npz else "",
        "replay_npz": str(pp.replay_npz) if pp.replay_npz else "",
        "cycle_alignment": cycle_diag,
    }
    return t, r, z_pred, z_true, cycle_ids, r_label, true_source, meta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {k: float("nan") for k in ["r2", "mae", "rmse", "nmae", "nrmse", "bias", "truth_range"]}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    yrange = float(np.max(yt) - np.min(yt))
    denom = yrange if yrange > 1e-30 else float(np.std(yt))
    nmae = float(mae / denom) if denom > 1e-30 else float("nan")
    nrmse = float(rmse / denom) if denom > 1e-30 else float("nan")
    return {"r2": r2, "mae": mae, "rmse": rmse, "nmae": nmae, "nrmse": nrmse, "bias": bias, "truth_range": yrange}


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def add_cycle_annotations(ax: Any, t: np.ndarray, r: np.ndarray, z: np.ndarray, cycle_ids: Optional[np.ndarray], max_labels: int = 12) -> None:
    if cycle_ids is None or cycle_ids.size != t.size:
        return
    unique = [int(c) for c in np.unique(cycle_ids) if np.isfinite(c)]
    if len(unique) > max_labels:
        # Keep endpoints and representative cycles.
        idxs = np.unique(np.linspace(0, len(unique) - 1, max_labels).round().astype(int))
        unique = [unique[i] for i in idxs]
    for c in unique:
        mask = cycle_ids.astype(int) == c
        if not np.any(mask):
            continue
        ti = float(np.nanmedian(t[mask]))
        ri = float(r[min(len(r) - 1, max(0, len(r) // 2))])
        zi = float(np.nanpercentile(z[mask, :], 85))
        try:
            ax.text(ti, ri, zi, f"cycle {c}", fontsize=8, fontname="Times New Roman")
        except Exception:
            pass


def pretty_target_name(target: str) -> str:
    return {"cs_a": "Anode Cs(t,r)", "cs_c": "Cathode Cs(t,r)"}.get(target, target)


def plot_target(
    target: str,
    t: np.ndarray,
    r: np.ndarray,
    z_pred: np.ndarray,
    z_true: np.ndarray,
    cycle_ids: Optional[np.ndarray],
    args: argparse.Namespace,
    pp: ProfilePaths,
    r_label: str,
    true_source: str,
    meta: Mapping[str, Any],
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    requested_cycles = parse_cycles(args.cycles)
    mask, selection_info = select_by_cycles_auto(t, cycle_ids, requested_cycles, args, pp)
    if not np.any(mask):
        diag = {
            "canonical_cell_uid": pp.canonical_uid,
            "requested_cycles": args.cycles,
            "prediction_time_range": [float(np.nanmin(t)), float(np.nanmax(t))] if t.size else [],
            "prediction_npz": str(pp.pred_npz),
            "softlabel_npz": str(pp.softlabel_npz) if pp.softlabel_npz else "",
            "replay_npz": str(pp.replay_npz) if pp.replay_npz else "",
            "cycle_alignment": meta.get("cycle_alignment", {}),
            "cycle_selection": selection_info,
        }
        print("[cycle-diagnostics]", json.dumps(diag, ensure_ascii=False, indent=2))
        if str(getattr(args, "on_empty_cycles", "error")).lower() == "all_available":
            print("[warning] Requested cycles are absent from the prediction grid; plotting all available prediction-grid cycles because --on_empty_cycles all_available was set.")
            mask = np.ones(t.shape[0], dtype=bool)
        else:
            msg = (
                f"Cycle selection {args.cycles!r} returned zero prediction-grid time points for {pp.canonical_uid}.\n"
                f"Available cycles on the selected prediction grid: {selection_info.get('available_cycles_on_prediction_grid') or meta.get('cycle_alignment', {}).get('available_cycles') or 'unknown'}.\n"
                "This usually means the saved prediction NPZ was generated for a different time window than the requested cycles. "
                "Use --cycles all or one of the available cycles, or generate/provide a prediction NPZ covering the requested cycles via --pred_npz. "
                "Run with --list_cycles_only to inspect full replay/soft-label cycle ranges."
            )
            raise ValueError(msg)
    t_sel = t[mask]
    cyc_sel = cycle_ids[mask] if cycle_ids is not None else None
    zp = z_pred[mask, :]
    zt = z_true[mask, :]

    tidx = downsample_indices(t_sel.size, int(args.max_plot_time_points))
    ridx = np.arange(0, r.size, max(1, int(args.r_stride)))
    t_plot = t_sel[tidx]
    r_plot = r[ridx]
    zp_plot = zp[tidx][:, ridx]
    zt_plot = zt[tidx][:, ridx]
    cyc_plot = cyc_sel[tidx] if cyc_sel is not None else None

    metrics = compute_metrics(zt, zp)
    T, R = np.meshgrid(t_plot, r_plot, indexing="ij")

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(16, 8), num=f"D17-G {pp.canonical_uid} {target} cycles {args.cycles}")
    ax_pred = fig.add_subplot(1, 2, 1, projection="3d")
    ax_true = fig.add_subplot(1, 2, 2, projection="3d")

    if args.sync_color_range:
        vmin = float(np.nanmin([np.nanmin(zp_plot), np.nanmin(zt_plot)]))
        vmax = float(np.nanmax([np.nanmax(zp_plot), np.nanmax(zt_plot)]))
    else:
        vmin = vmax = None  # type: ignore[assignment]

    surf1 = ax_pred.plot_surface(T, R, zp_plot, cmap=args.cmap_pred, vmin=vmin, vmax=vmax,
                                 linewidth=0, antialiased=True, shade=True)
    surf2 = ax_true.plot_surface(T, R, zt_plot, cmap=args.cmap_true, vmin=vmin, vmax=vmax,
                                 linewidth=0, antialiased=True, shade=True)
    fig.colorbar(surf1, ax=ax_pred, shrink=0.62, pad=0.08, label="Concentration")
    fig.colorbar(surf2, ax=ax_true, shrink=0.62, pad=0.08, label="Concentration")

    for ax, title, z in [
        (ax_pred, f"{pretty_target_name(target)} Prediction Surface", zp_plot),
        (ax_true, f"{pretty_target_name(target)} Soft-label Truth Surface", zt_plot),
    ]:
        ax.set_title(title, fontname="Times New Roman", fontsize=12, pad=12)
        ax.set_xlabel("Global t (s)", fontname="Times New Roman")
        ax.set_ylabel(r_label, fontname="Times New Roman")
        ax.set_zlabel(r"$c_s$ (mol m$^{-3}$)", fontname="Times New Roman")
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        if args.sync_zlim:
            zmin = float(np.nanmin([np.nanmin(zp_plot), np.nanmin(zt_plot)]))
            zmax = float(np.nanmax([np.nanmax(zp_plot), np.nanmax(zt_plot)]))
            pad = 0.02 * max(1e-12, zmax - zmin)
            ax.set_zlim(zmin - pad, zmax + pad)
        if args.annotate_cycles:
            add_cycle_annotations(ax, t_plot, r_plot, z, cyc_plot, max_labels=int(args.max_cycle_labels))

    cycles_label = args.cycles if str(args.cycles).lower() not in {"all", "*"} else "all available cycles"
    title = (
        f"{pp.canonical_uid} | {pretty_target_name(target)} | cycles {cycles_label}\n"
        f"Global metrics over selected points: "
        f"R$^2$={metrics['r2']:.6f}, "
        f"NMAE={metrics['nmae']:.4%}, NRMSE={metrics['nrmse']:.4%}, "
        f"MAE={metrics['mae']:.6g}, RMSE={metrics['rmse']:.6g}, bias={metrics['bias']:.6g}"
    )
    fig.suptitle(title, fontname="Times New Roman", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if args.save_png:
        save_dir = Path(args.save_dir or ".")
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_cycles = re.sub(r"[^0-9A-Za-z_-]+", "_", str(args.cycles))
        out = save_dir / f"D17_G_CS3D_{pp.canonical_uid}_{target}_cycles_{safe_cycles}.png"
        fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
        print(f"[saved] {out}")

    print(json.dumps({
        "target": target,
        "canonical_cell_uid": pp.canonical_uid,
        "pred_npz": str(pp.pred_npz),
        "truth_source": true_source,
        "softlabel_npz": str(pp.softlabel_npz) if pp.softlabel_npz else "",
        "replay_npz": str(pp.replay_npz) if pp.replay_npz else "",
        "cycles": args.cycles,
        "cycle_selection": selection_info,
        "selected_time_points": int(t_sel.size),
        "plotted_time_points": int(t_plot.size),
        "n_r": int(r.size),
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive 3D cs_a/cs_c prediction-vs-softlabel surface plotter for D17-G.")
    p.add_argument("--batch", required=True, help="Batch identifier, e.g. 2, batch2, or Batch-2.")
    p.add_argument("--battery", required=True, help="Battery identifier, e.g. 3, battery3, or battery-3.")
    p.add_argument("--cycles", default="all", help="Cycle selection, e.g. 13-15, 13,14,15, or all.")
    p.add_argument("--cycle_index_base", default="auto", choices=["auto", "exact", "zero_based", "one_based"], help="Cycle-id convention recovery. auto tries exact, then requested-1/requested+1 if exact is absent.")
    p.add_argument("--on_empty_cycles", default="error", choices=["error", "all_available"], help="What to do if requested cycles are absent from the prediction NPZ grid. Default error avoids silently plotting wrong cycles.")
    p.add_argument("--list_cycles_only", action="store_true", help="Resolve paths and print available replay/soft-label cycles plus prediction-grid cycles, then exit without plotting.")
    p.add_argument("--target", default="both", choices=["both", "cs_a", "cs_c"], help="Which concentration target to plot.")

    p.add_argument("--pred_npz", default="", help="Direct D17-G prediction NPZ path. Overrides --pred_root search.")
    p.add_argument("--pred_root", default="E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g", help="Root directory to recursively search for prediction NPZ files.")
    p.add_argument("--softlabel_npz", default="", help="Direct solution_softlabels.npz path. Overrides --softlabel_root/manifest.")
    p.add_argument("--softlabel_root", default="E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL", help="D15 ALL55 soft-label root.")
    p.add_argument("--split_manifest", default="E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json", help="D17 split manifest for resolving softlabel/replay paths.")
    p.add_argument("--replay_npz", default="", help="Direct replay profile NPZ path, used for cycle_id alignment when needed.")

    p.add_argument("--backend", default="", help="Optional Matplotlib backend, e.g. QtAgg or TkAgg. Leave empty for default.")
    p.add_argument("--cmap_pred", default="coolwarm", help="Colormap for PINN prediction surface. Default matches left-style red/blue.")
    p.add_argument("--cmap_true", default="viridis", help="Colormap for soft-label truth surface. Default matches right-style green/yellow.")
    p.add_argument("--elev", default=25.0, type=float, help="Initial 3D view elevation.")
    p.add_argument("--azim", default=-62.0, type=float, help="Initial 3D view azimuth.")
    p.add_argument("--max_plot_time_points", default=1200, type=int, help="Uniformly downsample plotted time points for interactive speed; 0 disables.")
    p.add_argument("--r_stride", default=1, type=int, help="Radial stride for plotting.")
    p.add_argument("--sync_zlim", action="store_true", default=True, help="Use shared z-axis limits for pred/truth.")
    p.add_argument("--no_sync_zlim", dest="sync_zlim", action="store_false", help="Do not synchronize z-axis limits.")
    p.add_argument("--sync_color_range", action="store_true", default=True, help="Use shared color range for pred/truth.")
    p.add_argument("--no_sync_color_range", dest="sync_color_range", action="store_false", help="Do not synchronize color range.")
    p.add_argument("--annotate_cycles", action="store_true", default=True, help="Annotate cycle numbers on surfaces when cycle_id is available.")
    p.add_argument("--no_annotate_cycles", dest="annotate_cycles", action="store_false", help="Disable cycle annotations.")
    p.add_argument("--max_cycle_labels", default=12, type=int, help="Maximum cycle labels to draw per subplot.")
    p.add_argument("--save_png", action="store_true", help="Also save PNG files.")
    p.add_argument("--save_dir", default="", help="Directory for PNG outputs when --save_png is used.")
    p.add_argument("--dpi", default=220, type=int, help="PNG DPI.")
    p.add_argument("--no_show", action="store_true", help="Do not call plt.show(); useful for batch PNG export.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend:
        import matplotlib
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt

    pp = resolve_paths(
        batch_arg=args.batch,
        battery_arg=args.battery,
        split_manifest=args.split_manifest or None,
        softlabel_root=args.softlabel_root or None,
        pred_root=args.pred_root or None,
        pred_npz=args.pred_npz or None,
        softlabel_npz=args.softlabel_npz or None,
        replay_npz=args.replay_npz or None,
    )
    print("[resolved]", json.dumps({
        "canonical_cell_uid": pp.canonical_uid,
        "protocol": pp.protocol,
        "pred_npz": str(pp.pred_npz) if pp.pred_npz else "",
        "softlabel_npz": str(pp.softlabel_npz) if pp.softlabel_npz else "",
        "replay_npz": str(pp.replay_npz) if pp.replay_npz else "",
    }, ensure_ascii=False, indent=2))

    if args.list_cycles_only:
        diag: Dict[str, Any] = {
            "canonical_cell_uid": pp.canonical_uid,
            "full_cycle_diagnostics": get_full_cycle_diagnostics(pp),
        }
        try:
            t0, _, _, _, cyc0, _, _, meta0 = load_target_arrays(pp, "cs_c")
            diag["prediction_time_range"] = [float(np.nanmin(t0)), float(np.nanmax(t0))] if t0.size else []
            diag["prediction_grid_available_cycles"] = compact_cycle_ranges(np.unique(cyc0).astype(int).tolist()) if cyc0 is not None else "unknown"
            diag["cycle_alignment"] = meta0.get("cycle_alignment", {})
        except Exception as e:
            diag["prediction_grid_error"] = repr(e)
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        return 0

    targets = ["cs_a", "cs_c"] if args.target == "both" else [args.target]
    for target in targets:
        t, r, z_pred, z_true, cyc, r_label, true_source, meta = load_target_arrays(pp, target)
        plot_target(target, t, r, z_pred, z_true, cyc, args, pp, r_label, true_source, meta)

    if not args.no_show:
        print("[interactive] Use the Matplotlib toolbar/mouse to drag and rotate the 3D view. Close the window(s) to exit.")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
