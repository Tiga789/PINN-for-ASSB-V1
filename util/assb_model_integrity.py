# -*- coding: utf-8 -*-
"""Integrity guards for preserving the ModelFin_107A core.

ModelFin_109 failed partly because the delivered code did not strictly preserve
107A's successful architecture.  This module turns that requirement into code:
critical core keys must not silently disappear, and new aging keys are the only
allowed mismatch category during state-dict loading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math
import re

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

PathLike = Union[str, Path]

CRITICAL_KEY_PATTERNS = [
    "net",
    "branch",
    "trunk",
    "cs_a",
    "cs_c",
    "phie",
    "phis_c",
    "potential",
    "cbar",
    "radial",
    "rescale",
]

ALLOWED_AGING_KEY_PATTERNS = [
    "aging",
    "lam",
    "theta_window",
    "r_ohm",
    "mechanism",
]

FORBIDDEN_OVERLAY_PATTERNS = [
    "_base.py",
    "107A_base",
    "overlay",
    "importlib.util.spec_from_file_location",
    "SourceFileLoader",
]


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    t = str(text).lower()
    return any(str(p).lower() in t for p in patterns)


def has_critical_missing(keys: Sequence[str]) -> bool:
    return any(_contains_any(k, CRITICAL_KEY_PATTERNS) and not _contains_any(k, ALLOWED_AGING_KEY_PATTERNS) for k in keys)


def has_unexpected_core_keys(keys: Sequence[str]) -> bool:
    # Unexpected keys from a 107A state dict often mean the new model does not
    # expose the old core modules.  Allow only clearly aging-related keys.
    return any(not _contains_any(k, ALLOWED_AGING_KEY_PATTERNS) for k in keys)


def strict_load_with_report(model, state_dict: Dict[str, Any], *, allow_new_aging_keys: bool = True, report_path: Optional[PathLike] = None):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for strict_load_with_report")
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "critical_missing": [k for k in missing if _contains_any(k, CRITICAL_KEY_PATTERNS) and not _contains_any(k, ALLOWED_AGING_KEY_PATTERNS)],
        "unexpected_core": [k for k in unexpected if not _contains_any(k, ALLOWED_AGING_KEY_PATTERNS)],
        "allow_new_aging_keys": bool(allow_new_aging_keys),
    }
    if report_path is not None:
        save_json(report, report_path)
    if report["critical_missing"]:
        raise RuntimeError("Critical 107A keys missing during load: " + ", ".join(report["critical_missing"][:20]))
    if report["unexpected_core"]:
        raise RuntimeError("Unexpected 107A core keys during load: " + ", ".join(report["unexpected_core"][:20]))
    return result, report


def state_dict_key_report(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(state_dict.keys())
    return {
        "n_keys": len(keys),
        "n_critical_like_keys": sum(1 for k in keys if _contains_any(k, CRITICAL_KEY_PATTERNS)),
        "n_aging_like_keys": sum(1 for k in keys if _contains_any(k, ALLOWED_AGING_KEY_PATTERNS)),
        "sample_keys": keys[:50],
    }


def load_state_dict_file(path: PathLike) -> Dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load state dict files")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        # Many PINNSTRIPES checkpoints are directly a state_dict.
        if all(hasattr(v, "shape") or np.isscalar(v) for v in payload.values()):
            return payload
    raise RuntimeError(f"Cannot extract state_dict from {path}")


def _as_np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    n = min(obs.size, pred.size)
    if n == 0:
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan"), "MAX": float("nan"), "corr": float("nan"), "R2": float("nan")}
    obs = obs[:n]
    pred = pred[:n]
    mask = np.isfinite(obs) & np.isfinite(pred)
    if not mask.any():
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan"), "MAX": float("nan"), "corr": float("nan"), "R2": float("nan")}
    e = pred[mask] - obs[mask]
    out = {"n": int(mask.sum()), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))), "MAX": float(np.max(np.abs(e)))}
    if mask.sum() >= 2 and np.std(obs[mask]) > 1e-15 and np.std(pred[mask]) > 1e-15:
        out["corr"] = float(np.corrcoef(obs[mask], pred[mask])[0, 1])
    else:
        out["corr"] = float("nan")
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((obs[mask] - np.mean(obs[mask])) ** 2))
    out["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    return out


def compare_npz_states(reference_npz: PathLike, candidate_npz: PathLike, *, state_keys: Sequence[str] = ("cs_a", "cs_c", "phie", "phis_c")) -> Dict[str, Any]:
    reference_npz = Path(reference_npz)
    candidate_npz = Path(candidate_npz)
    if not reference_npz.exists():
        raise FileNotFoundError(reference_npz)
    if not candidate_npz.exists():
        raise FileNotFoundError(candidate_npz)
    out: Dict[str, Any] = {"available": True, "state_keys": list(state_keys)}
    with np.load(reference_npz, allow_pickle=True) as ref, np.load(candidate_npz, allow_pickle=True) as cand:
        for key in state_keys:
            if key not in ref.files or key not in cand.files:
                out[key] = {"available": False, "reason": f"missing key {key}"}
                continue
            out[key] = _metrics(_as_np(ref[key]), _as_np(cand[key]))
    return out


def scan_for_overlay_patterns(root: PathLike, *, relative_files: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    root = Path(root)
    files: List[Path]
    if relative_files:
        files = [root / f for f in relative_files]
    else:
        files = list(root.rglob("*.py")) + [p for p in root.rglob("input_assb_ModelFin110*") if p.is_file()]
    hits = []
    for path in files:
        if not path.exists() or path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in FORBIDDEN_OVERLAY_PATTERNS:
            if pattern.lower() in text.lower():
                hits.append({"file": str(path.relative_to(root) if path.is_relative_to(root) else path), "pattern": pattern})
    return {"ok": len(hits) == 0, "hits": hits}


def save_json(obj: Dict[str, Any], path: PathLike) -> None:
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, float):
            return None if not math.isfinite(v) else v
        return v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "CRITICAL_KEY_PATTERNS",
    "ALLOWED_AGING_KEY_PATTERNS",
    "FORBIDDEN_OVERLAY_PATTERNS",
    "has_critical_missing",
    "has_unexpected_core_keys",
    "strict_load_with_report",
    "state_dict_key_report",
    "load_state_dict_file",
    "compare_npz_states",
    "scan_for_overlay_patterns",
    "save_json",
]
