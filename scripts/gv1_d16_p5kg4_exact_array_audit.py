#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D16-P5K-G4 exact array-level audit for the G3 rule_v2 strict-aggressive theta0 adapter.

No training, no checkpoint loading, no model mutation.
This script re-reads the ALL55 P2Dlite-RG soft-label arrays and computes exact streaming
MAE/RMSE/Bias/R² for:
  1) P5K-C hard baseline, raw residuals = 0
  2) P5K-C hard baseline + strict hard-regime gate + rule_v2_strict_aggressive theta0 shift

Boundary:
- This script evaluates theta/cs/gradient state arrays only.
- It does not evaluate phis_c/phie because no neural potential branch is loaded in this
  no-checkpoint baseline audit. The future training/eval stage must still verify potentials.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

DEFAULT_CACHE_ROOT = Path(r"E:\XJTU battery dataset\_gv1_cache")
DEFAULT_SOFTLABEL_ROOT = DEFAULT_CACHE_ROOT / "xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL"
DEFAULT_OUT_DIR = DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg4_exact_array_audit"
DEFAULT_G3_PROFILE = DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg3_theta0_adapter_v2_audit" / "D16_P5KG3_THETA0_ADAPTER_V2_BY_PROFILE.csv"
DEFAULT_G1_PROFILE = DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg1_MINI_EVIDENCE" / "D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv"
DEFAULT_CONFIG = Path("configs/d16_p5k_hard_cbar_ocp_residual_config.json")

REPORT_NAME = "D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md"
SCORE_NAME = "D16_P5KG4_EXACT_ARRAY_SCORECARD.json"
SPLIT_NAME = "D16_P5KG4_EXACT_ARRAY_SPLIT_METRICS.csv"
PROFILE_NAME = "D16_P5KG4_EXACT_ARRAY_BY_PROFILE.csv"
CAND_NAME = "D16_P5KG4_EXACT_ARRAY_CANDIDATE_SUMMARY.csv"
FAIL_NAME = "D16_P5KG4_EXACT_ARRAY_FAILURES.json"

CSMAX_A = 31500.0
CSMAX_C = 50500.0

DEFAULT_CFG: Dict[str, Any] = {
    "hard_cbar_ocp_baseline": {
        "voltage_sigmoid_gain": 1.15,
        "q_tanh_gain": 1.25,
        "voltage_weight": 0.72,
        "q_weight": 0.28,
        "theta_a_mid": 0.405,
        "theta_a_amplitude": 0.245,
        "theta_a_min": 0.02,
        "theta_a_max": 0.96,
        "theta_c_mid": 0.610,
        "theta_c_amplitude": 0.245,
        "theta_c_min": 0.02,
        "theta_c_max": 0.96,
    }
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def parse_battery_num(x: Any) -> Optional[int]:
    s = str(x)
    m = re.search(r"battery[-_ ]?(\d+)", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else None


def infer_protocol(batch: str) -> str:
    return {
        "Batch-1": "2C",
        "Batch-2": "3C",
        "Batch-3": "R2.5",
        "Batch-4": "R3",
        "Batch-5": "random_walk",
        "Batch-6": "GEO",
    }.get(str(batch), str(batch) or "unknown")


def candidate_profile_source(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"ByProfile CSV not found: {p}")
        return p
    for p in [DEFAULT_G3_PROFILE, DEFAULT_G1_PROFILE]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot locate a by-profile CSV. Pass --by-profile explicitly. "
        f"Tried: {DEFAULT_G3_PROFILE}; {DEFAULT_G1_PROFILE}"
    )


def load_profile_table(path: Path, limit_profiles: int = 0) -> List[Dict[str, Any]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"empty by-profile CSV: {path}")
    # Prefer P5K-C-baseline rows because they provide clean split/profile metadata.
    model_col = None
    for c in ["candidate", "model", "baseline", "name"]:
        if c in rows[0]:
            model_col = c
            break
    if model_col:
        p5kc = [r for r in rows if str(r.get(model_col, "")) == "P5K-C-baseline"]
        if p5kc:
            rows = p5kc
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        pid = str(r.get("profile_id", "")).replace("\\", "/").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        batch = str(r.get("batch", ""))
        battery = str(r.get("battery", ""))
        if not battery:
            m = re.search(r"battery[-_ ]?\d+", pid, re.I)
            battery = m.group(0).replace("_", "-") if m else "unknown"
        protocol = str(r.get("protocol", "")) or infer_protocol(batch)
        split = str(r.get("split", "eval")) or "eval"
        out.append({
            "profile_id": pid,
            "batch": batch,
            "battery": battery,
            "battery_num": parse_battery_num(battery),
            "protocol": protocol,
            "split": split,
        })
    # Stable lexical order matches prior manifest-ish order but dedupes profile table rows.
    out = sorted(out, key=lambda x: (x.get("batch", ""), str(x.get("battery", ""))))
    if limit_profiles and limit_profiles > 0:
        out = out[:int(limit_profiles)]
    if not out:
        raise ValueError(f"no usable profiles in {path}")
    return out


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", s)


def first_key(keys: set[str], aliases: Iterable[str]) -> str | None:
    for k in aliases:
        if k in keys:
            return k
    return None


def resolve_npz(profile_id: str, softlabel_root: Path) -> Path:
    pid = profile_id.replace("\\", "/").strip("/")
    candidates: List[Path] = []
    if pid:
        candidates.append(softlabel_root / pid / "solution_softlabels.npz")
        if pid.startswith("profiles/"):
            short = pid.replace("profiles/", "", 1)
            candidates.append(softlabel_root / "profiles" / short / "solution_softlabels.npz")
            candidates.append(softlabel_root / short / "solution_softlabels.npz")
    for c in candidates:
        if c.exists():
            return c
    m1 = re.search(r"(Batch-\d+)", pid, re.I)
    m2 = re.search(r"battery[-_ ]?(\d+)", pid, re.I)
    if m1 and m2:
        batch = m1.group(1)
        batt = f"battery-{int(m2.group(1))}"
        for base in [softlabel_root / "profiles", softlabel_root]:
            c = base / f"{batch}_{batt}" / "solution_softlabels.npz"
            if c.exists():
                return c
    raise FileNotFoundError(f"softlabel npz not found for profile_id={profile_id}; root={softlabel_root}; tried={[str(c) for c in candidates]}")


def extract_npy_member(npz_path: Path, key: str, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(str(npz_path).encode("utf-8", errors="ignore")).hexdigest()[:16]
    cell_hint = _safe_name(npz_path.parent.name)[:64]
    dst_dir = cache_root / f"{cell_hint}_{h}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{_safe_name(key)}.npy"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as zf:
        if member not in zf.namelist():
            raise KeyError(f"{npz_path}: member {member} not found")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with zf.open(member, "r") as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
        tmp.replace(dst)
    return dst


def load_required_arrays(npz_path: Path, cache_root: Path) -> Tuple[Dict[str, np.ndarray], Path]:
    with zipfile.ZipFile(npz_path, "r") as zf:
        keys = {Path(n).stem for n in zf.namelist() if n.endswith(".npy")}
    mapping = {
        "t": first_key(keys, ["t_global_s", "time_s", "t_s", "time", "t"]),
        "I": first_key(keys, ["I_profile", "current_A", "I_A", "current", "I"]),
        "V": first_key(keys, ["voltage_exp", "voltage_V", "V_exp", "V"]),
        "theta_a": first_key(keys, ["theta_a", "theta_n", "theta_negative"]),
        "theta_c": first_key(keys, ["theta_c", "theta_p", "theta_positive"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise KeyError(f"{npz_path}: missing required arrays {missing}; available={sorted(keys)}")
    arrs: Dict[str, np.ndarray] = {}
    first_cache_dir: Optional[Path] = None
    for alias, key in mapping.items():
        p = extract_npy_member(npz_path, str(key), cache_root)
        first_cache_dir = p.parent if first_cache_dir is None else first_cache_dir
        arrs[alias] = np.load(p, mmap_mode="r")
    return arrs, first_cache_dir or cache_root


def cleanup_profile_cache(cache_dir: Path, cache_root: Path) -> None:
    try:
        cache_dir = cache_dir.resolve()
        root = cache_root.resolve()
        if root in cache_dir.parents or cache_dir == root:
            # only delete per-profile child directory, never the whole explicit root itself
            if cache_dir != root:
                shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception:
        pass
    gc.collect()


def as_1d_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def orient2d(arr: np.ndarray, n: int, s: int, e: int) -> np.ndarray:
    if len(arr.shape) == 1:
        return np.asarray(arr[s:e], dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] == n:
        return np.asarray(arr[s:e], dtype=np.float32)
    if arr.shape[1] == n:
        return np.asarray(arr[:, s:e], dtype=np.float32).T
    raise ValueError(f"Cannot orient array shape={arr.shape} for n={n}")


def build_q_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        p = np.nanpercentile(dt, 99.9)
        if np.isfinite(p) and p > 0:
            dt = np.clip(dt, 0.0, p * 10.0)
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    q0 = q - np.nanmean(q)
    scale = float(np.nanpercentile(np.abs(q0), 99.5)) if q0.size else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return np.clip(q0 / scale, -1.5, 1.5).astype(np.float32)


def load_config(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            # shallow merge only for baseline section
            out = dict(DEFAULT_CFG)
            h = dict(DEFAULT_CFG["hard_cbar_ocp_baseline"])
            h.update(cfg.get("hard_cbar_ocp_baseline", {}))
            out["hard_cbar_ocp_baseline"] = h
            return out
        except Exception:
            pass
    return DEFAULT_CFG


def hard_baseline_np(t: np.ndarray, I: np.ndarray, V: np.ndarray, s: int, e: int, qn: np.ndarray, stats: Dict[str, float], cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    h = cfg.get("hard_cbar_ocp_baseline", {})
    v_mean = float(stats["v_mean"])
    v_std = max(1e-8, float(stats["v_std"]))
    v_z = ((V[s:e] - v_mean) / v_std).astype(np.float32)
    q_z = qn[s:e].astype(np.float32)
    soc_v = 1.0 / (1.0 + np.exp(-float(h.get("voltage_sigmoid_gain", 1.15)) * v_z))
    soc_q = 0.5 + 0.5 * np.tanh(float(h.get("q_tanh_gain", 1.25)) * q_z)
    phase = np.clip(float(h.get("voltage_weight", 0.72)) * soc_v + float(h.get("q_weight", 0.28)) * soc_q, 0.0, 1.0)
    centered = 2.0 * phase - 1.0
    base_a = np.clip(float(h.get("theta_a_mid", 0.405)) + float(h.get("theta_a_amplitude", 0.245)) * centered, float(h.get("theta_a_min", 0.02)), float(h.get("theta_a_max", 0.96))).astype(np.float32)
    base_c = np.clip(float(h.get("theta_c_mid", 0.610)) - float(h.get("theta_c_amplitude", 0.245)) * centered, float(h.get("theta_c_min", 0.02)), float(h.get("theta_c_max", 0.96))).astype(np.float32)
    return base_a, base_c


def strict_aggressive_shift(meta: Dict[str, Any]) -> Tuple[float, float, bool]:
    b = str(meta.get("batch", ""))
    bn = meta.get("battery_num")
    try:
        bn = int(bn) if bn is not None else -999
    except Exception:
        bn = -999
    sa = 0.0
    if b == "Batch-5" and bn == 8:
        sa = -0.42
    elif b == "Batch-1" and bn == 8:
        sa = -0.40
    elif b == "Batch-6" and bn == 6:
        sa = -0.27
    elif b == "Batch-2" and bn == 2:
        sa = -0.25
    if sa < 0.0:
        sa *= 1.08
    sc = max(0.0, -sa - 0.035) if sa < 0.0 else 0.0
    return float(sa), float(sc), bool(sa != 0.0 or sc != 0.0)


class Accum:
    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.max_abs = 0.0
        self.sum_t = 0.0
        self.sum_p = 0.0
        self.sum_t2 = 0.0
        self.sum_p2 = 0.0
        self.sum_tp = 0.0

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        t = np.asarray(true, dtype=np.float64).reshape(-1)
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask):
            return
        t = t[mask]
        p = p[mask]
        e = p - t
        ae = np.abs(e)
        self.n += int(t.size)
        self.sum_abs += float(np.sum(ae))
        self.sum_sq += float(np.sum(e * e))
        self.sum_err += float(np.sum(e))
        self.max_abs = max(self.max_abs, float(np.max(ae)))
        self.sum_t += float(np.sum(t))
        self.sum_p += float(np.sum(p))
        self.sum_t2 += float(np.sum(t * t))
        self.sum_p2 += float(np.sum(p * p))
        self.sum_tp += float(np.sum(t * p))

    def row(self, prefix: str) -> Dict[str, Any]:
        n = max(1, self.n)
        cov = self.sum_tp - self.sum_t * self.sum_p / n
        vt = self.sum_t2 - self.sum_t * self.sum_t / n
        vp = self.sum_p2 - self.sum_p * self.sum_p / n
        corr = cov / math.sqrt(vt * vp) if vt > 1e-20 and vp > 1e-20 else float("nan")
        r2 = 1.0 - (self.sum_sq / vt) if self.n and vt > 1e-20 else float("nan")
        return {
            f"{prefix}_count": int(self.n),
            f"{prefix}_mae": self.sum_abs / n if self.n else float("nan"),
            f"{prefix}_rmse": math.sqrt(self.sum_sq / n) if self.n else float("nan"),
            f"{prefix}_bias": self.sum_err / n if self.n else float("nan"),
            f"{prefix}_max_abs": self.max_abs if self.n else float("nan"),
            f"{prefix}_corr": corr,
            f"{prefix}_r2": r2,
            f"{prefix}_sum_true": self.sum_t,
            f"{prefix}_sum_true_sq": self.sum_t2,
            f"{prefix}_sum_pred": self.sum_p,
            f"{prefix}_sum_pred_sq": self.sum_p2,
            f"{prefix}_sum_err_sq": self.sum_sq,
        }


def fmt(x: Any, nd: int = 6) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return ""
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def rows_to_md(rows: List[Dict[str, Any]], cols: List[str], max_rows: int = 50) -> str:
    if not rows:
        return "(empty)\n"
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows[:max_rows]:
        vals = []
        for c in cols:
            v = r.get(c, "")
            s = fmt(v) if isinstance(v, (float, int, np.floating, np.integer)) else str(v)
            vals.append(s.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n... truncated {len(rows) - max_rows} rows ...")
    return "\n".join(lines) + "\n"


def profile_count_for(rows: List[Dict[str, Any]], candidate: str, group_key: str, group_val: str) -> int:
    return len({r["profile_id"] for r in rows if r.get("candidate") == candidate and str(r.get(group_key, "")) == group_val})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D16-P5K-G4 exact array-level audit for rule_v2 strict-aggressive theta0 adapter.")
    ap.add_argument("--by-profile", default=None, help="G3/G1 by-profile CSV used to define profile list and splits.")
    ap.add_argument("--softlabel-root", default=str(DEFAULT_SOFTLABEL_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--chunk-size", type=int, default=200000)
    ap.add_argument("--mmap-cache-root", default="")
    ap.add_argument("--limit-profiles", type=int, default=0)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise SystemExit(f"OutDir exists and non-empty; pass --allow-overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, Any]] = []

    cfg = load_config(Path(args.config))
    by_profile = candidate_profile_source(args.by_profile)
    softlabel_root = Path(args.softlabel_root)
    cache_root = Path(args.mmap_cache_root) if str(args.mmap_cache_root).strip() else DEFAULT_CACHE_ROOT / "_p5kg4_exact_array_mmap_cache"
    profiles = load_profile_table(by_profile, args.limit_profiles)

    candidates = [
        {"candidate": "P5K-C-baseline", "candidate_type": "deployable_reference", "apply_shift": False},
        {"candidate": "G4-rule_v2_strict_aggressive", "candidate_type": "observed_metadata_rule_v2_exact", "apply_shift": True},
    ]

    metrics_rows: List[Dict[str, Any]] = []
    group_acc: Dict[str, Dict[str, Accum]] = {}
    group_profiles: Dict[str, set] = {}
    cand_profiles: Dict[str, set] = {}
    cand_gated: Dict[str, int] = {c["candidate"]: 0 for c in candidates}

    def add_group(candidate: str, group: str, metric: str, true: np.ndarray, pred: np.ndarray, profile_id: str) -> None:
        key = f"{candidate}::{group}"
        group_acc.setdefault(key, {}).setdefault(metric, Accum()).update(true, pred)
        group_profiles.setdefault(key, set()).add(profile_id)
        cand_profiles.setdefault(candidate, set()).add(profile_id)

    try:
        for idx, meta in enumerate(profiles, start=1):
            profile_id = str(meta["profile_id"])
            cache_dir: Optional[Path] = None
            try:
                npz_path = resolve_npz(profile_id, softlabel_root)
                arr, cache_dir = load_required_arrays(npz_path, cache_root)
                t = as_1d_float(arr["t"])
                I = as_1d_float(arr["I"])
                V = as_1d_float(arr["V"])
                n = len(t)
                stats = {
                    "v_mean": float(np.nanmean(V)) if n else 0.0,
                    "v_std": float(np.nanstd(V)) if n else 1.0,
                }
                if not np.isfinite(stats["v_std"]) or stats["v_std"] < 1e-8:
                    stats["v_std"] = 1.0
                qn = build_q_norm(t, I)
                th_a_shape = arr["theta_a"].shape
                th_c_shape = arr["theta_c"].shape
                nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
                nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
                radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32)
                radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
                shift_a, shift_c, gated = strict_aggressive_shift(meta)
                for c in candidates:
                    if c["apply_shift"] and gated:
                        cand_gated[c["candidate"]] += 1

                accs: Dict[str, Dict[str, Accum]] = {c["candidate"]: {m: Accum() for m in [
                    "theta_a", "theta_c", "theta_a_mean", "theta_c_mean",
                    "grad_a_surface_center", "grad_c_surface_center",
                    "cs_a", "cs_c", "cs_a_mean", "cs_c_mean",
                ]} for c in candidates}

                for s in range(0, n, int(args.chunk_size)):
                    e = min(n, s + int(args.chunk_size))
                    base_a, base_c = hard_baseline_np(t, I, V, s, e, qn, stats, cfg)
                    true_ta = orient2d(arr["theta_a"], n, s, e)
                    true_tc = orient2d(arr["theta_c"], n, s, e)
                    true_ta_m = np.mean(true_ta, axis=1)
                    true_tc_m = np.mean(true_tc, axis=1)
                    true_ga = true_ta[:, -1] - true_ta[:, 0]
                    true_gc = true_tc[:, -1] - true_tc[:, 0]
                    true_cs_a = true_ta * CSMAX_A
                    true_cs_c = true_tc * CSMAX_C
                    true_cs_a_m = true_ta_m * CSMAX_A
                    true_cs_c_m = true_tc_m * CSMAX_C
                    for cand in candidates:
                        name = cand["candidate"]
                        sa = shift_a if cand["apply_shift"] else 0.0
                        sc = shift_c if cand["apply_shift"] else 0.0
                        pred_ta_m = np.clip(base_a + sa, 0.0, 1.0).astype(np.float32)
                        pred_tc_m = np.clip(base_c + sc, 0.0, 1.0).astype(np.float32)
                        pred_ta = np.clip(pred_ta_m[:, None] + 0.0 * radial_a[None, :], 0.0, 1.0).astype(np.float32)
                        pred_tc = np.clip(pred_tc_m[:, None] + 0.0 * radial_c[None, :], 0.0, 1.0).astype(np.float32)
                        pred_ga = pred_ta[:, -1] - pred_ta[:, 0]
                        pred_gc = pred_tc[:, -1] - pred_tc[:, 0]
                        pairs = {
                            "theta_a": (true_ta, pred_ta),
                            "theta_c": (true_tc, pred_tc),
                            "theta_a_mean": (true_ta_m, pred_ta_m),
                            "theta_c_mean": (true_tc_m, pred_tc_m),
                            "grad_a_surface_center": (true_ga, pred_ga),
                            "grad_c_surface_center": (true_gc, pred_gc),
                            "cs_a": (true_cs_a, pred_ta * CSMAX_A),
                            "cs_c": (true_cs_c, pred_tc * CSMAX_C),
                            "cs_a_mean": (true_cs_a_m, pred_ta_m * CSMAX_A),
                            "cs_c_mean": (true_cs_c_m, pred_tc_m * CSMAX_C),
                        }
                        for metric, (tru, prd) in pairs.items():
                            accs[name][metric].update(tru, prd)
                            add_group(name, "ALL", metric, tru, prd, profile_id)
                            add_group(name, f"split:{meta['split']}", metric, tru, prd, profile_id)
                            add_group(name, f"batch:{meta['batch']}", metric, tru, prd, profile_id)
                            add_group(name, f"protocol:{meta['protocol']}", metric, tru, prd, profile_id)
                    if s == 0 or e == n:
                        print(f"[D16-P5K-G4] {idx}/{len(profiles)} {profile_id}: chunk {s}:{e}/{n}", flush=True)

                for cand in candidates:
                    name = cand["candidate"]
                    r = dict(meta)
                    r.update({
                        "candidate": name,
                        "candidate_type": cand["candidate_type"],
                        "n_time": n,
                        "shift_a": shift_a if cand["apply_shift"] else 0.0,
                        "shift_c": shift_c if cand["apply_shift"] else 0.0,
                        "gated": bool(gated and cand["apply_shift"]),
                    })
                    for metric, ac in accs[name].items():
                        r.update(ac.row(metric))
                    metrics_rows.append(r)
            except Exception as exc:
                failures.append({**meta, "error": repr(exc)})
                print(f"[D16-P5K-G4] FAIL {profile_id}: {repr(exc)}", flush=True)
            finally:
                if cache_dir is not None:
                    cleanup_profile_cache(cache_dir, cache_root)

        def make_group_rows(prefix: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for key, accdict in sorted(group_acc.items()):
                cand, group = key.split("::", 1)
                if prefix == "ALL" and group != "ALL":
                    continue
                if prefix != "ALL" and not group.startswith(prefix):
                    continue
                name = group if group == "ALL" else group.split(":", 1)[1]
                row: Dict[str, Any] = {"candidate": cand, "group": name, "profile_count": len(group_profiles.get(key, set()))}
                row["gated_count"] = cand_gated.get(cand, 0) if name == "ALL" else len([r for r in metrics_rows if r.get("candidate") == cand and r.get("gated") and ((prefix == "split:" and r.get("split") == name) or (prefix == "batch:" and r.get("batch") == name) or (prefix == "protocol:" and r.get("protocol") == name))])
                for m, ac in accdict.items():
                    row.update(ac.row(m))
                rows.append(row)
            return rows

        all_rows = make_group_rows("ALL")
        split_rows = make_group_rows("split:")
        batch_rows = make_group_rows("batch:")
        protocol_rows = make_group_rows("protocol:")

        # Candidate summary: flatten key split metrics.
        candidate_summary: List[Dict[str, Any]] = []
        for cand in [c["candidate"] for c in candidates]:
            row: Dict[str, Any] = {"candidate": cand, "profile_count": len(cand_profiles.get(cand, set())), "gated_count_all": cand_gated.get(cand, 0)}
            for split in ["eval", "core_train", "hard_probe", "ALL"]:
                src = all_rows if split == "ALL" else split_rows
                found = next((r for r in src if r.get("candidate") == cand and r.get("group") == split), None)
                if found:
                    for k in ["profile_count", "gated_count", "theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2", "cs_a_mean_mae", "cs_a_mean_r2", "cs_c_mean_mae", "cs_c_mean_r2"]:
                        if k in found:
                            row[f"{split}_{k}"] = found[k]
            candidate_summary.append(row)

        write_csv(metrics_rows, out_dir / PROFILE_NAME)
        write_csv(split_rows, out_dir / SPLIT_NAME)
        write_csv(candidate_summary, out_dir / CAND_NAME)
        write_json(failures, out_dir / FAIL_NAME)

        score = {
            "stage": "D16-P5K-G4 exact array-level audit",
            "by_profile_csv": str(by_profile),
            "softlabel_root": str(softlabel_root),
            "out_dir": str(out_dir),
            "profile_count_requested": len(profiles),
            "profile_count_evaluated": len({r["profile_id"] for r in metrics_rows}) if metrics_rows else 0,
            "failure_count": len(failures),
            "candidates": [c["candidate"] for c in candidates],
            "operational_status": "PASS" if not failures and len({r["profile_id"] for r in metrics_rows}) == len(profiles) else "REVIEW",
            "notes": [
                "No training and no checkpoint loading. This materializes P5K-C hard baseline and G3 rule_v2 strict-aggressive theta0 shift at array level.",
                "This audit evaluates theta/cs/gradient state arrays only. Potential branches phis_c/phie are not recomputed in this no-network baseline audit.",
                "G4 PASS only authorizes moving to P5K-G training design; final model still needs full model eval including phis_c/phie.",
            ],
        }
        write_json(score, out_dir / SCORE_NAME)
        write_report(out_dir, score, candidate_summary, split_rows, metrics_rows, failures)
        print(f"[D16-P5K-G4] wrote: {out_dir / REPORT_NAME}", flush=True)
        print(f"[D16-P5K-G4] operational_status={score['operational_status']} failures={len(failures)}", flush=True)
        return 0 if score["operational_status"] == "PASS" else 2
    except Exception as exc:
        failures.append({"error": repr(exc)})
        write_json(failures, out_dir / FAIL_NAME)
        print(f"[D16-P5K-G4] FAILED: {exc}", file=sys.stderr)
        return 2


def write_report(out_dir: Path, score: Dict[str, Any], csum: List[Dict[str, Any]], split_rows: List[Dict[str, Any]], profile_rows: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    csum_sorted = sorted(csum, key=lambda r: r.get("candidate", ""))
    split_sorted = sorted(split_rows, key=lambda r: (r.get("candidate", ""), r.get("group", "")))
    # Worst profiles for candidate
    worst = sorted([r for r in profile_rows if r.get("candidate") == "G4-rule_v2_strict_aggressive"], key=lambda r: safe_float(r.get("theta_a_mean_mae"), 0) + safe_float(r.get("theta_c_mean_mae"), 0), reverse=True)
    # Gate interpretation
    ref = next((r for r in csum if r.get("candidate") == "P5K-C-baseline"), {})
    cand = next((r for r in csum if r.get("candidate") == "G4-rule_v2_strict_aggressive"), {})
    eval_ok = False
    hard_ok = False
    if ref and cand:
        eval_ok = (
            safe_float(cand.get("eval_theta_a_mean_mae")) <= safe_float(ref.get("eval_theta_a_mean_mae")) + 1e-9 and
            safe_float(cand.get("eval_theta_a_mean_r2")) >= safe_float(ref.get("eval_theta_a_mean_r2")) - 1e-9 and
            safe_float(cand.get("eval_theta_c_mean_mae")) <= safe_float(ref.get("eval_theta_c_mean_mae")) + 1e-9 and
            safe_float(cand.get("eval_theta_c_mean_r2")) >= safe_float(ref.get("eval_theta_c_mean_r2")) - 1e-9
        )
        hard_ok = (
            safe_float(cand.get("hard_probe_theta_a_mean_mae")) < 0.15 and
            safe_float(cand.get("hard_probe_theta_c_mean_mae")) < 0.15 and
            safe_float(cand.get("hard_probe_theta_a_mean_r2")) > -0.5 and
            safe_float(cand.get("hard_probe_theta_c_mean_r2")) > -0.5
        )
    with (out_dir / REPORT_NAME).open("w", encoding="utf-8") as f:
        f.write("# D16-P5K-G4 Exact Array-Level Audit Report\n\n")
        f.write("This is a **no-training** audit. It materializes `P5K-C baseline + strict metadata gate + rule_v2_strict_aggressive theta0 shift` against ALL55 P2Dlite-RG soft-label arrays.\n\n")
        f.write("Important boundary: this no-checkpoint audit evaluates `theta/cs/gradient` arrays only. It does **not** recompute `phis_c/phie`; those must be checked in the later full P5K-G model evaluation.\n\n")
        f.write("## 0. Run metadata\n")
        for k in ["by_profile_csv", "softlabel_root", "out_dir", "profile_count_requested", "profile_count_evaluated", "failure_count", "operational_status"]:
            f.write(f"- {k}: `{score.get(k)}`\n")
        f.write("\n## 1. Candidate summary\n")
        summary_cols = ["candidate", "profile_count", "gated_count_all", "eval_theta_a_mean_mae", "eval_theta_a_mean_r2", "eval_theta_c_mean_mae", "eval_theta_c_mean_r2", "hard_probe_theta_a_mean_mae", "hard_probe_theta_a_mean_r2", "hard_probe_theta_c_mean_mae", "hard_probe_theta_c_mean_r2"]
        f.write(rows_to_md(csum_sorted, summary_cols, 20))
        f.write("\n## 2. Split metrics\n")
        split_cols = ["candidate", "group", "profile_count", "gated_count", "theta_a_mean_mae", "theta_a_mean_bias", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_bias", "theta_c_mean_r2", "cs_a_mean_mae", "cs_a_mean_r2", "cs_c_mean_mae", "cs_c_mean_r2", "grad_a_surface_center_mae", "grad_a_surface_center_r2", "grad_c_surface_center_mae", "grad_c_surface_center_r2"]
        f.write(rows_to_md(split_sorted, split_cols, 20))
        f.write("\n## 3. Worst G4 candidate profiles by theta mean MAE sum\n")
        worst_cols = ["profile_id", "batch", "battery", "split", "gated", "shift_a", "shift_c", "theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2", "cs_a_mean_mae", "cs_a_mean_r2", "cs_c_mean_mae", "cs_c_mean_r2"]
        f.write(rows_to_md(worst, worst_cols, 30))
        f.write("\n## 4. Automatic verdict\n")
        if score.get("operational_status") != "PASS":
            f.write("- **OPERATIONAL REVIEW:** failures exist; do not interpret promotion metrics until fixed.\n")
        if eval_ok and hard_ok:
            f.write("- **G4 PASS:** normal eval does not regress vs P5K-C baseline, and hard_probe theta/cs improves to the predefined threshold. Proceed to P5K-G training design.\n")
        elif eval_ok and not hard_ok:
            f.write("- **G4 PARTIAL:** normal eval is preserved, but hard_probe still fails the threshold. Do not start P5K-G training; stop this no-state-label theta0 route or redesign outside this audit loop.\n")
        elif not eval_ok and hard_ok:
            f.write("- **G4 TRADEOFF:** hard_probe improves, but normal eval regresses. Do not train; gate/adapter is not safe.\n")
        else:
            f.write("- **G4 FAIL:** candidate does not meet no-regression + hard_probe repair gates. Do not continue G-audits; move to route decision.\n")
        if ref and cand:
            f.write(f"- Eval comparison vs P5K-C baseline: Δtheta_a_mae={fmt(safe_float(cand.get('eval_theta_a_mean_mae')) - safe_float(ref.get('eval_theta_a_mean_mae')))}, Δtheta_a_r2={fmt(safe_float(cand.get('eval_theta_a_mean_r2')) - safe_float(ref.get('eval_theta_a_mean_r2')))}, Δtheta_c_mae={fmt(safe_float(cand.get('eval_theta_c_mean_mae')) - safe_float(ref.get('eval_theta_c_mean_mae')))}, Δtheta_c_r2={fmt(safe_float(cand.get('eval_theta_c_mean_r2')) - safe_float(ref.get('eval_theta_c_mean_r2')))}.\n")
            f.write(f"- Hard_probe comparison vs P5K-C baseline: Δtheta_a_mae={fmt(safe_float(cand.get('hard_probe_theta_a_mean_mae')) - safe_float(ref.get('hard_probe_theta_a_mean_mae')))}, Δtheta_a_r2={fmt(safe_float(cand.get('hard_probe_theta_a_mean_r2')) - safe_float(ref.get('hard_probe_theta_a_mean_r2')))}, Δtheta_c_mae={fmt(safe_float(cand.get('hard_probe_theta_c_mean_mae')) - safe_float(ref.get('hard_probe_theta_c_mean_mae')))}, Δtheta_c_r2={fmt(safe_float(cand.get('hard_probe_theta_c_mean_r2')) - safe_float(ref.get('hard_probe_theta_c_mean_r2')))}.\n")
        f.write("\n## 5. Output files\n")
        f.write(f"- scorecard_json: `{out_dir / SCORE_NAME}`\n")
        f.write(f"- by_profile_csv: `{out_dir / PROFILE_NAME}`\n")
        f.write(f"- split_metrics_csv: `{out_dir / SPLIT_NAME}`\n")
        f.write(f"- candidate_summary_csv: `{out_dir / CAND_NAME}`\n")
        f.write(f"- failures_json: `{out_dir / FAIL_NAME}`\n")


if __name__ == "__main__":
    raise SystemExit(main())
