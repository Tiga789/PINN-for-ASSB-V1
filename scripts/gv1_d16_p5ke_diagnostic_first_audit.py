from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# -----------------------------------------------------------------------------
# D16-P5K-E diagnostic-first audit
# No training. No soft-label data loss. No model modification.
# This script produces ONE primary Markdown report that can be pasted back.
# -----------------------------------------------------------------------------

DEFAULT_CACHE_ROOT = r"E:\XJTU battery dataset\_gv1_cache"
DEFAULT_SOFTLABEL_ROOT = DEFAULT_CACHE_ROOT + r"\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL"
DEFAULT_P5KC_RUN = DEFAULT_CACHE_ROOT + r"\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST\C_train10"
DEFAULT_P5KD_RUN = DEFAULT_CACHE_ROOT + r"\xjtu_d16_p5kd_train10_generator_aligned_hard_cbar_ocp_FAST\D_train10_prior_balanced"
DEFAULT_OUT = DEFAULT_CACHE_ROOT + r"\xjtu_d16_p5ke_diagnostic_first_audit"

STATE_KEYS = {
    "theta_a": ["theta_a", "theta_n", "theta_negative"],
    "theta_c": ["theta_c", "theta_p", "theta_positive"],
    "cs_a": ["cs_a", "c_s_a", "cs_n", "cs_negative"],
    "cs_c": ["cs_c", "c_s_c", "cs_p", "cs_positive"],
    "time": ["t_global_s", "time_s", "t_s", "time", "t"],
    "current": ["I_profile", "current_A", "I_A", "current", "I"],
    "voltage": ["voltage_exp", "voltage_V", "V_exp", "V"],
}

HIGH_RISK_KEYWORDS = (
    "generator", "aligned", "phase", "ocp", "cbar", "capacity", "theta", "csmax", "cs_",
    "residual", "guard", "slack", "floor", "ceiling", "prior", "p2dlite", "voltage",
    "q_", "coulomb", "baseline", "loss", "weight", "scale", "sign", "warm", "init",
)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def pth(x: str | Path) -> Path:
    return Path(str(x).strip().strip('"'))


def exists(path: str | Path) -> bool:
    try:
        return pth(path).exists()
    except Exception:
        return False


def read_json(path: str | Path, default: Any = None) -> Any:
    p = pth(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"__read_error__": repr(exc), "__path__": str(p)}


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    p = pth(path)
    if not p.exists():
        return []
    try:
        with p.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_text(path: str | Path, text: str) -> None:
    p = pth(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_float(x: Any, n: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "NA"
    if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.{n}e}"
    return f"{v:.{n}f}"


def file_size_str(path: str | Path) -> str:
    p = pth(path)
    try:
        if not p.exists():
            return "MISSING"
        b = p.stat().st_size
        if b > 1024**3:
            return f"{b/1024**3:.3f} GB"
        if b > 1024**2:
            return f"{b/1024**2:.3f} MB"
        if b > 1024:
            return f"{b/1024:.3f} KB"
        return f"{b} B"
    except Exception as exc:
        return f"ERR:{exc!r}"


def flatten_dict(d: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            flatten_dict(v, kk, out)
    elif isinstance(d, (list, tuple)):
        if len(d) <= 8 and all(not isinstance(x, (dict, list, tuple)) for x in d):
            out[prefix] = list(d)
        else:
            out[prefix + ".len"] = len(d)
    else:
        out[prefix] = d
    return out


def scalarish(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def high_risk_config_items(d: Dict[str, Any], max_items: int = 120) -> Dict[str, Any]:
    flat = flatten_dict(d)
    keep = {}
    for k, v in flat.items():
        lk = k.lower()
        if any(w in lk for w in HIGH_RISK_KEYWORDS):
            if scalarish(v) or (isinstance(v, list) and len(v) <= 8):
                keep[k] = v
            else:
                keep[k] = str(type(v).__name__)
    # Stable order by key.
    return dict(sorted(list(keep.items()), key=lambda kv: kv[0])[:max_items])


def diff_dict(a: Dict[str, Any], b: Dict[str, Any], max_items: int = 80) -> List[Tuple[str, Any, Any]]:
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        av = a.get(k, "<missing>")
        bv = b.get(k, "<missing>")
        if str(av) != str(bv):
            diffs.append((k, av, bv))
    return diffs[:max_items]


def pick(row: Dict[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


# -----------------------------------------------------------------------------
# Project-specific path discovery
# -----------------------------------------------------------------------------

def find_manifest(run_dir: Path, prefix: str) -> Path:
    candidates = sorted(run_dir.glob(f"{prefix}*_MANIFEST.csv"))
    if candidates:
        # Prefer non-summary, longest specific stage path.
        return candidates[0]
    return run_dir / f"{prefix}_MANIFEST.csv"


def p5kc_paths(run_dir: str | Path) -> Dict[str, Path]:
    r = pth(run_dir)
    return {
        "run_dir": r,
        "model_dir": r / "model_hard_cbar_ocp_residual",
        "manifest": r / "D16_P5K_C_train10_MANIFEST.csv",
        "scorecard": r / "eval_all55_vs_softlabels" / "D16_P5K_FINAL_SCORECARD.json",
        "split_metrics": r / "eval_all55_vs_softlabels" / "D16_P5K_SPLIT_METRICS.csv",
        "profile_metrics": r / "eval_all55_vs_softlabels" / "D16_P5K_METRICS_BY_PROFILE.csv",
        "batch_metrics": r / "eval_all55_vs_softlabels" / "D16_P5K_BATCH_METRICS.csv",
        "protocol_metrics": r / "eval_all55_vs_softlabels" / "D16_P5K_PROTOCOL_METRICS.csv",
        "train_summary": r / "model_hard_cbar_ocp_residual" / "D16_P5K_TRAINING_SUMMARY.json",
        "train_audit": r / "model_hard_cbar_ocp_residual" / "D16_P5K_TRAIN_INPUT_AUDIT.json",
        "checkpoint": r / "model_hard_cbar_ocp_residual" / "model" / "best_with_state.pt",
        "config_name": Path("configs/d16_p5k_hard_cbar_ocp_residual_config.json"),
    }


def p5kd_paths(run_dir: str | Path) -> Dict[str, Path]:
    r = pth(run_dir)
    return {
        "run_dir": r,
        "model_dir": r / "model_generator_aligned_hard_cbar_ocp",
        "manifest": r / "D16_P5KD_D_train10_prior_balanced_MANIFEST.csv",
        "scorecard": r / "eval_all55_vs_softlabels" / "D16_P5KD_FINAL_SCORECARD.json",
        "split_metrics": r / "eval_all55_vs_softlabels" / "D16_P5KD_SPLIT_METRICS.csv",
        "profile_metrics": r / "eval_all55_vs_softlabels" / "D16_P5KD_METRICS_BY_PROFILE.csv",
        "batch_metrics": r / "eval_all55_vs_softlabels" / "D16_P5KD_BATCH_METRICS.csv",
        "protocol_metrics": r / "eval_all55_vs_softlabels" / "D16_P5KD_PROTOCOL_METRICS.csv",
        "train_summary": r / "model_generator_aligned_hard_cbar_ocp" / "D16_P5KD_TRAINING_SUMMARY.json",
        "train_audit": r / "model_generator_aligned_hard_cbar_ocp" / "D16_P5KD_TRAIN_INPUT_AUDIT.json",
        "checkpoint": r / "model_generator_aligned_hard_cbar_ocp" / "model" / "best_with_state.pt",
        "config_name": Path("configs/d16_p5kd_generator_aligned_hard_cbar_ocp_config.json"),
    }


# -----------------------------------------------------------------------------
# Metrics extraction
# -----------------------------------------------------------------------------

def score_global(score: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(score, dict):
        return {}
    g = score.get("global_metrics_weighted") or score.get("global_metrics") or {}
    return g if isinstance(g, dict) else {}


def split_rows_by_group(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out = {}
    for r in rows:
        g = str(r.get("group", "")).lower()
        if g:
            out[g] = r
    return out


def metric_snapshot(paths: Dict[str, Path]) -> Dict[str, Any]:
    score = read_json(paths["scorecard"], {})
    split = split_rows_by_group(read_csv(paths["split_metrics"]))
    global_g = score_global(score)
    out = {
        "operational_status": score.get("operational_status", "NA") if isinstance(score, dict) else "NA",
        "profile_count_evaluated": score.get("profile_count_evaluated", score.get("profile_count_requested", "NA")) if isinstance(score, dict) else "NA",
        "failure_count": score.get("failure_count", "NA") if isinstance(score, dict) else "NA",
        "ALL": global_g,
        "eval": split.get("eval", {}),
        "train": split.get("train", {}),
    }
    return out


def metrics_table(models: Dict[str, Dict[str, Any]], group: str = "eval") -> str:
    cols = [
        "model", "profiles", "phis_c_mae", "phis_c_r2", "phie_mae", "phie_r2",
        "theta_a_mean_mae", "theta_a_mean_bias", "theta_a_mean_r2",
        "theta_c_mean_mae", "theta_c_mean_bias", "theta_c_mean_r2",
        "cs_a_mean_mae", "cs_a_mean_r2", "cs_c_mean_mae", "cs_c_mean_r2",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, snap in models.items():
        row = snap.get(group, {}) or {}
        vals = [name]
        vals.append(str(row.get("profile_count", row.get("profiles", snap.get("profile_count_evaluated", "NA")))))
        for c in cols[2:]:
            vals.append(fmt_float(row.get(c, ""), 6))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def regression_table(base: Dict[str, Any], cand: Dict[str, Any], base_name: str = "P5K-C", cand_name: str = "P5K-D", group: str = "eval") -> str:
    keys = [
        "phis_c_mae", "phis_c_r2", "phie_mae", "phie_r2",
        "theta_a_mean_mae", "theta_a_mean_bias", "theta_a_mean_r2",
        "theta_c_mean_mae", "theta_c_mean_bias", "theta_c_mean_r2",
        "cs_a_mean_mae", "cs_a_mean_r2", "cs_c_mean_mae", "cs_c_mean_r2",
    ]
    b = base.get(group, {}) or {}
    c = cand.get(group, {}) or {}
    lines = ["| metric | P5K-C | P5K-D | D-C | ratio(D/C, MAE only) |", "|---|---:|---:|---:|---:|"]
    for k in keys:
        bv = safe_float(b.get(k))
        cv = safe_float(c.get(k))
        diff = cv - bv if math.isfinite(bv) and math.isfinite(cv) else float("nan")
        ratio = ""
        if k.endswith("_mae") and math.isfinite(bv) and abs(bv) > 1e-12 and math.isfinite(cv):
            ratio = fmt_float(cv / bv, 3)
        lines.append(f"| {k} | {fmt_float(bv,6)} | {fmt_float(cv,6)} | {fmt_float(diff,6)} | {ratio} |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Checkpoint/config/audit introspection
# -----------------------------------------------------------------------------

def load_checkpoint_light(path: str | Path) -> Dict[str, Any]:
    p = pth(path)
    out: Dict[str, Any] = {"path": str(p), "exists": p.exists(), "size": file_size_str(p)}
    if not p.exists():
        return out
    if torch is None:
        out["error"] = "torch import failed"
        return out
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            out["type"] = str(type(ckpt))
            return out
        out["top_keys"] = sorted([str(k) for k in ckpt.keys()])
        out["model_config"] = ckpt.get("model_config", {})
        out["config"] = ckpt.get("config", {})
        state = ckpt.get("state", ckpt.get("state_dict", {}))
        if isinstance(state, dict):
            out["state_key_count"] = len(state)
            # Small fingerprint over key names and shapes only, not tensor data.
            fp_parts = []
            for k, v in list(state.items())[:2000]:
                shape = tuple(v.shape) if hasattr(v, "shape") else "?"
                fp_parts.append(f"{k}:{shape}")
            out["state_schema_sha1"] = hashlib.sha1("\n".join(fp_parts).encode()).hexdigest()[:16]
            out["first_state_keys"] = [str(k) for k in list(state.keys())[:12]]
        for k in ["x_mean", "x_std", "feature_names", "output_names", "train_set", "stage"]:
            if k in ckpt:
                v = ckpt[k]
                if hasattr(v, "shape"):
                    out[k] = {"shape": list(v.shape), "dtype": str(getattr(v, "dtype", ""))}
                elif isinstance(v, (list, tuple)):
                    out[k] = list(v)[:50]
                else:
                    out[k] = v
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def summarize_train_audit(path: str | Path) -> Dict[str, Any]:
    obj = read_json(path, {})
    out: Dict[str, Any] = {"exists": exists(path), "path": str(path), "size": file_size_str(path)}
    rows = None
    if isinstance(obj, dict):
        if isinstance(obj.get("rows"), list):
            rows = obj.get("rows")
        elif isinstance(obj.get("train_rows"), list):
            rows = obj.get("train_rows")
        else:
            rows = []
        out["top_keys"] = sorted(obj.keys())[:50]
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    out["row_count"] = len(rows)
    used = Counter()
    forbidden = Counter()
    profile_ids = []
    sidecar_summary = Counter()
    for r in rows:
        if not isinstance(r, dict):
            continue
        profile_ids.append(str(r.get("profile_id", r.get("cell_uid", ""))))
        for x in r.get("training_used_keys", []) or []:
            used[str(x)] += 1
        for x in r.get("training_forbidden_keys_not_loaded", []) or []:
            forbidden[str(x)] += 1
        for k, v in r.items():
            lk = k.lower()
            if "sidecar" in lk or "summary_found" in lk or "audit_found" in lk or "resolved_spec" in lk:
                if isinstance(v, bool):
                    sidecar_summary[f"{k}={v}"] += 1
                elif isinstance(v, str) and v:
                    sidecar_summary[f"{k}:present"] += 1
    out["profiles_first10"] = profile_ids[:10]
    out["training_used_keys_counts"] = dict(used)
    out["forbidden_keys_counts"] = dict(forbidden)
    out["sidecar_related_counts"] = dict(sidecar_summary)
    return out


# -----------------------------------------------------------------------------
# Soft-label / manifest / sidecar audit
# -----------------------------------------------------------------------------

def resolve_softlabel(row: Dict[str, str], softlabel_root: Path) -> Path:
    raw = row.get("softlabel_npz") or row.get("solution_softlabels") or row.get("softlabel_path") or ""
    if raw and pth(raw).exists():
        return pth(raw)
    pid = row.get("profile_id") or row.get("cell_uid") or ""
    if pid:
        # profiles/Batch-X_battery-Y -> root/profiles/Batch-X_battery-Y/solution_softlabels.npz
        cand = softlabel_root / pid / "solution_softlabels.npz"
        if cand.exists():
            return cand
        cand = softlabel_root / "profiles" / pid.replace("profiles/", "") / "solution_softlabels.npz"
        if cand.exists():
            return cand
    batch = row.get("batch", "")
    batt = row.get("battery", "")
    if batch and batt:
        cand = softlabel_root / "profiles" / f"{batch}_{batt}" / "solution_softlabels.npz"
        if cand.exists():
            return cand
    return pth(raw) if raw else Path("")


def npz_keys(npz_path: Path) -> List[str]:
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            keys = []
            for name in zf.namelist():
                if name.endswith(".npy"):
                    keys.append(Path(name).stem)
            return sorted(keys)
    except Exception:
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                return sorted(list(z.files))
        except Exception:
            return []


def read_small_npz_value(npz_path: Path, key: str) -> Any:
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            if key not in z.files:
                return None
            v = z[key]
            if getattr(v, "shape", ()) == ():
                return v.item()
            if v.size <= 8:
                return v.tolist()
            return {"shape": list(v.shape), "dtype": str(v.dtype)}
    except Exception:
        return None


def sidecar_compact(npz_path: Path) -> Dict[str, Any]:
    d = npz_path.parent
    out: Dict[str, Any] = {
        "profile_dir": str(d),
        "summary_found": (d / "soft_label_summary.json").exists(),
        "audit_found": (d / "soft_label_audit.json").exists(),
    }
    for fname, prefix in [("soft_label_summary.json", "summary"), ("soft_label_audit.json", "audit")]:
        p = d / fname
        obj = read_json(p, {})
        if isinstance(obj, dict) and obj:
            for k in ["status", "profile_id", "batch", "protocol", "resolved_spec_hash", "prior_hash", "generator_version", "p2dlite_rg_status", "radial_audit_status"]:
                if k in obj:
                    out[f"{prefix}.{k}"] = obj[k]
            # find any hash-like keys.
            for k, v in obj.items():
                lk = k.lower()
                if ("hash" in lk or "status" in lk) and scalarish(v):
                    out[f"{prefix}.{k}"] = v
    # Also inspect small hash key in NPZ if present.
    for key in ["resolved_spec_hash", "prior_hash", "cell_uid", "batch", "protocol"]:
        val = read_small_npz_value(npz_path, key)
        if val is not None:
            out[f"npz.{key}"] = val
    return out


def audit_manifest_and_sidecars(manifest: Path, softlabel_root: Path, max_samples: int = 8) -> Dict[str, Any]:
    rows = read_csv(manifest)
    out: Dict[str, Any] = {
        "manifest": str(manifest),
        "manifest_exists": manifest.exists(),
        "row_count": len(rows),
        "softlabel_root": str(softlabel_root),
    }
    split_counts = Counter((r.get("split") or "NA") for r in rows)
    batch_counts = Counter((r.get("batch") or "NA") for r in rows)
    out["split_counts"] = dict(split_counts)
    out["batch_counts"] = dict(batch_counts)
    missing = []
    side_counts = Counter()
    hash_counts = Counter()
    sample_sidecars = []
    sample_keys = []
    for i, r in enumerate(rows):
        npz = resolve_softlabel(r, softlabel_root)
        if not npz.exists():
            missing.append({"profile_id": r.get("profile_id", ""), "softlabel_npz": str(npz)})
            continue
        sc = sidecar_compact(npz)
        side_counts[f"summary_found={sc.get('summary_found')}"] += 1
        side_counts[f"audit_found={sc.get('audit_found')}"] += 1
        for k, v in sc.items():
            if "hash" in k.lower() and v not in (None, ""):
                hash_counts[str(v)] += 1
        if len(sample_sidecars) < max_samples:
            compact = {"profile_id": r.get("profile_id", ""), "split": r.get("split", ""), "npz": str(npz)}
            compact.update(sc)
            sample_sidecars.append(compact)
            sample_keys.append({"profile_id": r.get("profile_id", ""), "keys": npz_keys(npz)[:80]})
    out["missing_softlabel_count"] = len(missing)
    out["missing_softlabels_first10"] = missing[:10]
    out["sidecar_counts"] = dict(side_counts)
    out["resolved_or_prior_hash_unique_count"] = len(hash_counts)
    out["resolved_or_prior_hash_top10"] = hash_counts.most_common(10)
    out["sidecar_samples"] = sample_sidecars
    out["npz_key_samples"] = sample_keys
    return out


# -----------------------------------------------------------------------------
# Deep cbar / Coulomb diagnostic on selected profiles
# -----------------------------------------------------------------------------

def select_key(keys: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    s = set(keys)
    for a in aliases:
        if a in s:
            return a
    # Case-insensitive fallback.
    lower = {k.lower(): k for k in keys}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def extract_npy_member(npz_path: Path, key: str, cache_root: Path) -> Path:
    # Extract selected .npy member into short hash directory so np.load can mmap.
    h = hashlib.sha1(str(npz_path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]
    out_dir = cache_root / h
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.npy"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with zipfile.ZipFile(npz_path, "r") as zf:
        member = None
        for name in zf.namelist():
            if name == f"{key}.npy" or Path(name).stem == key:
                member = name
                break
        if member is None:
            raise KeyError(f"{npz_path}: npz member for key {key} not found")
        tmp = out_path.with_suffix(".tmp")
        with zf.open(member, "r") as src, tmp.open("wb") as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)
        tmp.replace(out_path)
    return out_path


def load_memmap(npz_path: Path, key: str, cache_root: Path) -> np.ndarray:
    p = extract_npy_member(npz_path, key, cache_root)
    return np.load(p, mmap_mode="r", allow_pickle=False)


def sample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if max_points <= 0 or max_points >= n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))


def orient_state_sample(arr: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    # Returns sampled rows shape (len(idx), radial) or (len(idx),).
    if arr.ndim == 1:
        return np.asarray(arr[idx], dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == n:
            return np.asarray(arr[idx, :], dtype=np.float64)
        if arr.shape[1] == n:
            return np.asarray(arr[:, idx].T, dtype=np.float64)
    # Fallback: flatten first axis if possible.
    try:
        return np.asarray(arr[idx], dtype=np.float64)
    except Exception:
        return np.asarray(arr, dtype=np.float64).reshape(n, -1)[idx]


def mean_state_sample(arr: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    x = orient_state_sample(arr, idx, n)
    if x.ndim == 1:
        return x.astype(np.float64)
    return np.nanmean(x, axis=1)


def build_q(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return np.array([], dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        p = np.nanpercentile(dt, 99.9)
        if np.isfinite(p) and p > 0:
            dt = np.clip(dt, 0.0, p * 10.0)
    return np.cumsum(I * dt) / 3600.0


def affine_fit_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "corr": float("nan"), "mae_to_affine": float("nan")}
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-20 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan")
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r2": r2,
        "corr": corr,
        "mae_to_affine": float(np.mean(np.abs(y - yhat))),
    }


def deep_coulomb_audit(
    manifest: Path,
    softlabel_root: Path,
    profile_metrics: Path,
    cache_root: Path,
    max_profiles: int = 10,
    sample_points: int = 12000,
) -> Dict[str, Any]:
    rows = read_csv(manifest)
    metrics = read_csv(profile_metrics)
    # Select hard profiles: top by theta_a+c mean MAE if metrics available, plus first train/eval representatives.
    selected_ids: List[str] = []
    if metrics:
        def score(r: Dict[str, str]) -> float:
            return safe_float(r.get("theta_a_mean_mae"), 0) + safe_float(r.get("theta_c_mean_mae"), 0)
        for r in sorted(metrics, key=score, reverse=True):
            pid = r.get("profile_id", "")
            if pid and pid not in selected_ids:
                selected_ids.append(pid)
            if len(selected_ids) >= max_profiles:
                break
    # Add split representatives if not enough.
    for r in rows:
        pid = r.get("profile_id", "")
        if pid and pid not in selected_ids:
            selected_ids.append(pid)
        if len(selected_ids) >= max_profiles:
            break
    row_by_id = {r.get("profile_id", ""): r for r in rows}
    out_rows = []
    failures = []
    for pid in selected_ids[:max_profiles]:
        r = row_by_id.get(pid)
        if not r:
            continue
        npz = resolve_softlabel(r, softlabel_root)
        try:
            keys = npz_keys(npz)
            kt = select_key(keys, STATE_KEYS["time"])
            ki = select_key(keys, STATE_KEYS["current"])
            kv = select_key(keys, STATE_KEYS["voltage"])
            ka = select_key(keys, STATE_KEYS["theta_a"])
            kc = select_key(keys, STATE_KEYS["theta_c"])
            if not all([kt, ki, kv, ka, kc]):
                raise KeyError(f"missing keys: time={kt} I={ki} V={kv} theta_a={ka} theta_c={kc}; keys={keys[:40]}")
            t_arr = load_memmap(npz, kt, cache_root)
            I_arr = load_memmap(npz, ki, cache_root)
            V_arr = load_memmap(npz, kv, cache_root)
            ta_arr = load_memmap(npz, ka, cache_root)
            tc_arr = load_memmap(npz, kc, cache_root)
            n = int(np.asarray(t_arr).reshape(-1).shape[0])
            idx = sample_idx(n, sample_points)
            t = np.asarray(t_arr[idx], dtype=np.float64).reshape(-1)
            I = np.asarray(I_arr[idx], dtype=np.float64).reshape(-1)
            V = np.asarray(V_arr[idx], dtype=np.float64).reshape(-1)
            # For q, sampling a nonuniform subset changes integral. Use full t/I if safe; otherwise sampled approximation.
            # t/I arrays are 1D and reasonable, so load full 1D via memmap without copying too much.
            t_full = np.asarray(t_arr, dtype=np.float64).reshape(-1)
            I_full = np.asarray(I_arr, dtype=np.float64).reshape(-1)
            q_full = build_q(t_full, I_full)
            q = q_full[idx]
            q0 = q - np.nanmean(q)
            q_scale = np.nanpercentile(np.abs(q0), 99.5) if q0.size else 1.0
            if not np.isfinite(q_scale) or q_scale < 1e-12:
                q_scale = 1.0
            qn = np.clip(q0 / q_scale, -1.5, 1.5)
            ta_m = mean_state_sample(ta_arr, idx, n)
            tc_m = mean_state_sample(tc_arr, idx, n)
            fit_a = affine_fit_stats(qn, ta_m)
            fit_c = affine_fit_stats(qn, tc_m)
            fit_v_a = affine_fit_stats(V, ta_m)
            fit_v_c = affine_fit_stats(V, tc_m)
            row = {
                "profile_id": pid,
                "batch": r.get("batch", ""),
                "battery": r.get("battery", ""),
                "split": r.get("split", ""),
                "n_time": n,
                "sample_points": int(idx.size),
                "time_key": kt,
                "current_key": ki,
                "voltage_key": kv,
                "theta_a_key": ka,
                "theta_c_key": kc,
                "I_min": float(np.nanmin(I_full)) if I_full.size else float("nan"),
                "I_max": float(np.nanmax(I_full)) if I_full.size else float("nan"),
                "q_span_Ah": float(np.nanmax(q_full) - np.nanmin(q_full)) if q_full.size else float("nan"),
                "theta_a_mean_min": float(np.nanmin(ta_m)),
                "theta_a_mean_max": float(np.nanmax(ta_m)),
                "theta_c_mean_min": float(np.nanmin(tc_m)),
                "theta_c_mean_max": float(np.nanmax(tc_m)),
                "theta_a_vs_q_slope": fit_a["slope"],
                "theta_a_vs_q_r2": fit_a["r2"],
                "theta_a_vs_q_corr": fit_a["corr"],
                "theta_a_affine_q_mae": fit_a["mae_to_affine"],
                "theta_c_vs_q_slope": fit_c["slope"],
                "theta_c_vs_q_r2": fit_c["r2"],
                "theta_c_vs_q_corr": fit_c["corr"],
                "theta_c_affine_q_mae": fit_c["mae_to_affine"],
                "theta_a_vs_V_r2": fit_v_a["r2"],
                "theta_c_vs_V_r2": fit_v_c["r2"],
            }
            # Attach existing metric row if any.
            mr = next((m for m in metrics if m.get("profile_id") == pid), {})
            for k in ["theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2", "phis_c_mae", "phis_c_r2"]:
                if k in mr:
                    row["metric_" + k] = safe_float(mr.get(k))
            out_rows.append(row)
        except Exception as exc:
            failures.append({"profile_id": pid, "error": repr(exc), "npz": str(npz)})
    return {"rows": out_rows, "failures": failures, "cache_root": str(cache_root), "selected_profile_ids": selected_ids[:max_profiles]}


# -----------------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------------

def md_table_from_dicts(rows: List[Dict[str, Any]], columns: Sequence[str], max_rows: int = 20, float_n: int = 5) -> str:
    if not rows:
        return "(no rows)"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows[:max_rows]:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(fmt_float(v, float_n))
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_sidecar_section(name: str, audit: Dict[str, Any]) -> str:
    lines = [f"### {name} manifest / sidecar audit", ""]
    lines.append(f"- manifest_exists: `{audit.get('manifest_exists')}`")
    lines.append(f"- row_count: `{audit.get('row_count')}`")
    lines.append(f"- missing_softlabel_count: `{audit.get('missing_softlabel_count')}`")
    lines.append(f"- split_counts: `{audit.get('split_counts')}`")
    lines.append(f"- batch_counts: `{audit.get('batch_counts')}`")
    lines.append(f"- sidecar_counts: `{audit.get('sidecar_counts')}`")
    lines.append(f"- resolved/prior hash unique count: `{audit.get('resolved_or_prior_hash_unique_count')}`")
    lines.append(f"- top hash values: `{audit.get('resolved_or_prior_hash_top10')}`")
    if audit.get("missing_softlabels_first10"):
        lines.append("\nFirst missing softlabels:")
        lines.append("```json")
        lines.append(json.dumps(audit.get("missing_softlabels_first10"), indent=2, ensure_ascii=False))
        lines.append("```")
    if audit.get("sidecar_samples"):
        lines.append("\nSidecar samples:")
        lines.append("```json")
        lines.append(json.dumps(audit.get("sidecar_samples")[:3], indent=2, ensure_ascii=False))
        lines.append("```")
    if audit.get("npz_key_samples"):
        lines.append("\nNPZ key samples:")
        lines.append("```json")
        lines.append(json.dumps(audit.get("npz_key_samples")[:2], indent=2, ensure_ascii=False))
        lines.append("```")
    return "\n".join(lines)


def render_checkpoint_section(ckpt_c: Dict[str, Any], ckpt_d: Dict[str, Any]) -> str:
    c_cfg = high_risk_config_items({"model_config": ckpt_c.get("model_config", {}), "config": ckpt_c.get("config", {})})
    d_cfg = high_risk_config_items({"model_config": ckpt_d.get("model_config", {}), "config": ckpt_d.get("config", {})})
    diffs = diff_dict(c_cfg, d_cfg, max_items=100)
    lines = ["## 5. Checkpoint / config high-risk diff", ""]
    lines.append("### Checkpoint schema")
    lines.append("| item | P5K-C | P5K-D |")
    lines.append("|---|---|---|")
    for k in ["exists", "size", "state_key_count", "state_schema_sha1", "first_state_keys"]:
        lines.append(f"| {k} | `{ckpt_c.get(k)}` | `{ckpt_d.get(k)}` |")
    if ckpt_c.get("error") or ckpt_d.get("error"):
        lines.append(f"\n- P5K-C checkpoint error: `{ckpt_c.get('error')}`")
        lines.append(f"- P5K-D checkpoint error: `{ckpt_d.get('error')}`")
    lines.append("\n### High-risk config differences")
    if not diffs:
        lines.append("No high-risk config differences found or configs missing.")
    else:
        lines.append("| key | P5K-C | P5K-D |")
        lines.append("|---|---|---|")
        for k, a, b in diffs:
            sa = json.dumps(a, ensure_ascii=False) if not isinstance(a, str) else a
            sb = json.dumps(b, ensure_ascii=False) if not isinstance(b, str) else b
            lines.append(f"| `{k}` | `{sa}` | `{sb}` |")
    return "\n".join(lines)


def render_audit_summary(audit_c: Dict[str, Any], audit_d: Dict[str, Any]) -> str:
    lines = ["## 4. Training input audit comparison", ""]
    lines.append("| item | P5K-C | P5K-D |")
    lines.append("|---|---|---|")
    for k in ["exists", "size", "row_count", "training_used_keys_counts", "forbidden_keys_counts", "sidecar_related_counts", "profiles_first10"]:
        lines.append(f"| {k} | `{audit_c.get(k)}` | `{audit_d.get(k)}` |")
    lines.append("\nInterpretation: training input audit should show only observed `t_global_s / I_profile / voltage_exp` loaded as training inputs. Any theta/cs/phie/phis key in `training_used_keys` is a boundary violation.")
    return "\n".join(lines)


def render_deep_audit(deep: Dict[str, Any]) -> str:
    lines = ["## 6. Soft-label Coulomb / OCP phase diagnostic on selected hard profiles", ""]
    lines.append(f"- cache_root: `{deep.get('cache_root')}`")
    lines.append(f"- selected_profile_ids: `{deep.get('selected_profile_ids')}`")
    if deep.get("failures"):
        lines.append("\nDeep audit failures:")
        lines.append("```json")
        lines.append(json.dumps(deep.get("failures"), indent=2, ensure_ascii=False))
        lines.append("```")
    rows = deep.get("rows", [])
    cols = [
        "profile_id", "split", "n_time", "q_span_Ah",
        "theta_a_vs_q_slope", "theta_a_vs_q_r2", "theta_a_affine_q_mae",
        "theta_c_vs_q_slope", "theta_c_vs_q_r2", "theta_c_affine_q_mae",
        "theta_a_vs_V_r2", "theta_c_vs_V_r2",
        "metric_theta_a_mean_mae", "metric_theta_a_mean_r2", "metric_theta_c_mean_mae", "metric_theta_c_mean_r2",
    ]
    lines.append("\nSelected profile diagnostic table:")
    lines.append(md_table_from_dicts(rows, cols, max_rows=20, float_n=5))
    lines.append("\nInterpretation: if `theta_*_vs_q_r2` is reasonably high for most normal profiles but worst/outlier profiles have poor model R², then the hard cbar/Coulomb backbone is plausible while profile-level initial inventory/OCP phase remains the main problem. If `theta_*_vs_q_r2` is low or slopes are inconsistent across normal profiles, then current sign/scale/capacity prior must be audited before new training.")
    return "\n".join(lines)


def render_report(
    args: argparse.Namespace,
    paths_c: Dict[str, Path],
    paths_d: Dict[str, Path],
    snap_c: Dict[str, Any],
    snap_d: Dict[str, Any],
    side_c: Dict[str, Any],
    side_d: Dict[str, Any],
    train_audit_c: Dict[str, Any],
    train_audit_d: Dict[str, Any],
    ckpt_c: Dict[str, Any],
    ckpt_d: Dict[str, Any],
    deep: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# D16-P5K-E Diagnostic-First Audit Report")
    lines.append("")
    lines.append("This is a **no-training** diagnostic. It does not change model checkpoints. It compares P5K-C and P5K-D and audits whether P5K-D failed because of generator/prior alignment, hard-cbar/OCP initialization, current integral sign/scale, or train-set hard-probe contamination.")
    lines.append("")
    lines.append("## 0. Run metadata")
    lines.append(f"- time: `{now_iso()}`")
    lines.append(f"- project_root: `{args.project_root}`")
    lines.append(f"- softlabel_root: `{args.softlabel_root}`")
    lines.append(f"- p5kc_run_dir: `{args.p5kc_run_dir}`")
    lines.append(f"- p5kd_run_dir: `{args.p5kd_run_dir}`")
    lines.append(f"- max_deep_profiles: `{args.max_deep_profiles}`")
    lines.append(f"- sample_points_per_profile: `{args.sample_points_per_profile}`")
    lines.append(f"- report_file: `{args.report_file}`")
    lines.append("")

    lines.append("## 1. Key file presence")
    lines.append("| file | P5K-C | P5K-D |")
    lines.append("|---|---|---|")
    for key in ["manifest", "train_summary", "train_audit", "checkpoint", "scorecard", "split_metrics", "profile_metrics", "batch_metrics", "protocol_metrics"]:
        pc = paths_c[key]
        pd = paths_d[key]
        lines.append(f"| {key} | `{pc.exists()}` / {file_size_str(pc)} | `{pd.exists()}` / {file_size_str(pd)} |")
    lines.append("")

    lines.append("## 2. P5K-C vs P5K-D metric comparison")
    lines.append("### Eval split")
    lines.append(metrics_table({"P5K-C": snap_c, "P5K-D": snap_d}, group="eval"))
    lines.append("\n### Train split")
    lines.append(metrics_table({"P5K-C": snap_c, "P5K-D": snap_d}, group="train"))
    lines.append("\n### ALL55")
    lines.append(metrics_table({"P5K-C": snap_c, "P5K-D": snap_d}, group="ALL"))
    lines.append("\n### Regression delta, eval split")
    lines.append(regression_table(snap_c, snap_d, group="eval"))
    lines.append("")

    # Auto verdict based on common fields.
    eval_c = snap_c.get("eval", {}) or {}
    eval_d = snap_d.get("eval", {}) or {}
    c_ar2 = safe_float(eval_c.get("theta_a_mean_r2"))
    c_cr2 = safe_float(eval_c.get("theta_c_mean_r2"))
    d_ar2 = safe_float(eval_d.get("theta_a_mean_r2"))
    d_cr2 = safe_float(eval_d.get("theta_c_mean_r2"))
    c_amae = safe_float(eval_c.get("theta_a_mean_mae"))
    c_cmae = safe_float(eval_c.get("theta_c_mean_mae"))
    d_amae = safe_float(eval_d.get("theta_a_mean_mae"))
    d_cmae = safe_float(eval_d.get("theta_c_mean_mae"))
    lines.append("## 3. Automatic diagnostic verdict")
    lines.append("")
    if math.isfinite(d_ar2) and math.isfinite(d_cr2) and math.isfinite(c_ar2) and math.isfinite(c_cr2):
        if d_ar2 < c_ar2 and d_cr2 < c_cr2:
            lines.append("- **P5K-D is worse than P5K-C on both theta mean exact R².** Treat P5K-D as failed ablation unless a metric-file mismatch is found.")
        elif d_ar2 > c_ar2 and d_cr2 > c_cr2:
            lines.append("- P5K-D improves both theta mean exact R² over P5K-C. Check MAE/bias before promotion.")
        else:
            lines.append("- P5K-D is mixed versus P5K-C. Inspect per-profile and split trade-offs.")
    if math.isfinite(d_amae) and math.isfinite(d_cmae) and math.isfinite(c_amae) and math.isfinite(c_cmae):
        if d_amae > c_amae * 1.25 or d_cmae > c_cmae * 1.25:
            lines.append("- **P5K-D has a large theta MAE regression relative to P5K-C.** Do not continue P5K-D epochs; diagnose initialization/scale/sign/prior alignment.")
    lines.append("- Promotion gate remains: eval theta_a_mean_mae < 0.15, eval theta_c_mean_mae < 0.15, theta_a_mean_r2 > 0.85, theta_c_mean_r2 > 0.85, phis_c_r2 > 0.99.")
    lines.append("")

    lines.append(render_audit_summary(train_audit_c, train_audit_d))
    lines.append("")
    lines.append(render_checkpoint_section(ckpt_c, ckpt_d))
    lines.append("")
    lines.append(render_sidecar_section("P5K-C", side_c))
    lines.append("")
    lines.append(render_sidecar_section("P5K-D", side_d))
    lines.append("")
    lines.append(render_deep_audit(deep))
    lines.append("")

    lines.append("## 7. Corrective interpretation checklist")
    lines.append("")
    lines.append("Use this checklist to decide the next coding step:")
    lines.append("")
    lines.append("1. **If P5K-C has positive theta R² and P5K-D has negative theta R²**, rollback to P5K-C structure and treat generator alignment as too strong or mis-scaled.")
    lines.append("2. **If sidecar hashes are missing or inconsistent**, generator/prior alignment must be audit-only until the resolved prior is made single-source-of-truth.")
    lines.append("3. **If q-integral affine fit is strong for normal profiles**, the hard cbar backbone is valid; the remaining issue is profile-level theta0/OCP phase initialization, especially hard probes/outliers.")
    lines.append("4. **If q-integral affine fit is weak or slopes vary unexpectedly**, audit current sign, capacity scale, nominal capacity, cbar sign, and theta/csmax mapping before training another model.")
    lines.append("5. **If train split is much worse than eval**, inspect whether train includes hard probes such as Batch-5 battery-8, Batch-1 battery-8, Batch-6 battery-6. Do not let hard probes dominate model selection.")
    lines.append("6. **Do not reintroduce P5G-style heuristic gap loss.** Future correction should be P5K-C + diagnostic, weak generator consistency, OCP/theta0 initializer, and hard-probe isolation.")
    lines.append("")
    lines.append("## 8. Minimal next-action recommendation")
    lines.append("")
    lines.append("- If this report confirms P5K-D regression: freeze P5K-D as failed ablation, keep P5K-C as main candidate, and build P5K-E around **diagnosed weak prior alignment**, not a strong generator target.")
    lines.append("- If this report reveals current sign/capacity mismatch: fix resolver/hard-cbar baseline first; retraining before that is not meaningful.")
    lines.append("- If this report shows train hard probes dominate failures: create `core_train10 + hard_probe` split and evaluate hard probes separately.")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D16-P5K-E diagnostic-first audit: P5K-C vs P5K-D, no training, one Markdown report.")
    ap.add_argument("--project-root", default=str(Path.cwd()))
    ap.add_argument("--softlabel-root", default=DEFAULT_SOFTLABEL_ROOT)
    ap.add_argument("--p5kc-run-dir", default=DEFAULT_P5KC_RUN)
    ap.add_argument("--p5kd-run-dir", default=DEFAULT_P5KD_RUN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--report-file", default="")
    ap.add_argument("--mmap-cache-root", default=DEFAULT_CACHE_ROOT + r"\_p5ke_diag_mmap_cache")
    ap.add_argument("--max-deep-profiles", type=int, default=10)
    ap.add_argument("--sample-points-per-profile", type=int, default=12000)
    ap.add_argument("--skip-deep-softlabel-audit", action="store_true")
    args = ap.parse_args(argv)

    args.project_root = str(pth(args.project_root))
    args.softlabel_root = str(pth(args.softlabel_root))
    args.p5kc_run_dir = str(pth(args.p5kc_run_dir))
    args.p5kd_run_dir = str(pth(args.p5kd_run_dir))
    out_dir = pth(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_file:
        args.report_file = str(out_dir / "D16_P5K_E_DIAGNOSTIC_REPORT.md")
    args.mmap_cache_root = str(pth(args.mmap_cache_root))

    paths_c = p5kc_paths(args.p5kc_run_dir)
    paths_d = p5kd_paths(args.p5kd_run_dir)
    # Recover alternative manifest names if defaults missing.
    if not paths_c["manifest"].exists():
        alt = find_manifest(paths_c["run_dir"], "D16_P5K_")
        if alt.exists(): paths_c["manifest"] = alt
    if not paths_d["manifest"].exists():
        alt = find_manifest(paths_d["run_dir"], "D16_P5KD_")
        if alt.exists(): paths_d["manifest"] = alt

    snap_c = metric_snapshot(paths_c)
    snap_d = metric_snapshot(paths_d)
    side_c = audit_manifest_and_sidecars(paths_c["manifest"], pth(args.softlabel_root), max_samples=6)
    side_d = audit_manifest_and_sidecars(paths_d["manifest"], pth(args.softlabel_root), max_samples=6)
    train_audit_c = summarize_train_audit(paths_c["train_audit"])
    train_audit_d = summarize_train_audit(paths_d["train_audit"])
    ckpt_c = load_checkpoint_light(paths_c["checkpoint"])
    ckpt_d = load_checkpoint_light(paths_d["checkpoint"])

    if args.skip_deep_softlabel_audit:
        deep = {"rows": [], "failures": [], "cache_root": args.mmap_cache_root, "selected_profile_ids": [], "skipped": True}
    else:
        deep = deep_coulomb_audit(
            paths_c["manifest"],
            pth(args.softlabel_root),
            paths_c["profile_metrics"],
            pth(args.mmap_cache_root),
            max_profiles=max(0, int(args.max_deep_profiles)),
            sample_points=max(100, int(args.sample_points_per_profile)),
        )

    report = render_report(args, paths_c, paths_d, snap_c, snap_d, side_c, side_d, train_audit_c, train_audit_d, ckpt_c, ckpt_d, deep)
    write_text(args.report_file, report)
    print(f"[D16-P5K-E] wrote report: {args.report_file}")
    print("[D16-P5K-E] Paste this single Markdown file back for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
