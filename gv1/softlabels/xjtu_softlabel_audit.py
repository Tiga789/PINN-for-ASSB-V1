# -*- coding: utf-8 -*-
"""Audit helpers for XJTU P2Dlite soft labels.

D14-P4A changes
---------------
The audit now checks:
1. Metadata completeness: `batch`, `protocol`, and `cell_uid` must be present.
2. Terminal-voltage bounds: `phis_c_soft` must not exceed the nominal XJTU
   terminal-voltage window by more than the configured warn/fail thresholds.
3. Raw pre-bound voltage (`phis_c_soft_raw`) and bound correction are reported
   separately so the user can see how much D14-P4A changed the soft voltage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Sequence, Optional

import numpy as np


DEFAULT_VOLTAGE_BOUNDS = {
    "upper_warn_V": 4.25,
    "upper_fail_V": 4.35,
    "lower_warn_V": 2.45,
    "lower_fail_V": 2.35,
}


def _scalar_to_str(x) -> str:
    try:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        s = str(x)
    except Exception:
        s = str(x)
    return s.strip()


def _load_voltage_bounds(voltage_bounds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = dict(DEFAULT_VOLTAGE_BOUNDS)
    if voltage_bounds:
        out.update({k: float(v) for k, v in voltage_bounds.items() if k in out})
    return out


def audit_softlabel_npz(
    npz_path: str | Path,
    required_keys: Sequence[str],
    prior_hash: str = "",
    voltage_bounds: Optional[Dict[str, float]] = None,
    require_metadata: bool = True,
) -> Dict[str, Any]:
    p = Path(npz_path)
    row: Dict[str, Any] = {"npz_path": str(p), "exists": p.exists(), "status": "FAIL", "detail": ""}
    bounds = _load_voltage_bounds(voltage_bounds)

    if not p.exists():
        row["detail"] = "missing file"
        return row

    try:
        data = np.load(p, allow_pickle=True)
        keys = set(data.files)
        missing = [k for k in required_keys if k not in keys]
        if missing:
            row["detail"] = "missing keys: " + ",".join(missing)
            return row

        t = data["t_global_s"]
        cs_a = data["cs_a"]
        cs_c = data["cs_c"]
        th_a = data["theta_a"]
        th_c = data["theta_c"]
        phis = data["phis_c"]
        phis_soft = data["phis_c_soft"] if "phis_c_soft" in data.files else phis
        phis_base = data["phis_c_base"] if "phis_c_base" in data.files else np.full_like(phis_soft, np.nan)
        phis_raw = data["phis_c_soft_raw"] if "phis_c_soft_raw" in data.files else np.full_like(phis_soft, np.nan)
        vexp = data["voltage_exp"]

        finite_ok = all(np.isfinite(arr).all() for arr in [t, cs_a, cs_c, th_a, th_c, phis_soft, vexp])
        time_ok = bool(len(t) >= 10 and np.all(np.diff(t) >= 0))
        shape_ok = bool(cs_a.ndim == 2 and cs_c.ndim == 2 and cs_a.shape[0] == len(t) and cs_c.shape[0] == len(t))
        theta_a_oob = float(np.mean((th_a < -1e-4) | (th_a > 1.0001)))
        theta_c_oob = float(np.mean((th_c < -1e-4) | (th_c > 1.0001)))

        prior_hash_file = _scalar_to_str(data["resolved_spec_hash"]) if "resolved_spec_hash" in data.files else ""
        prior_ok = (not prior_hash) or (prior_hash_file == prior_hash)

        batch = _scalar_to_str(data["batch"]) if "batch" in data.files else ""
        protocol = _scalar_to_str(data["protocol"]) if "protocol" in data.files else ""
        cell_uid = _scalar_to_str(data["cell_uid"]) if "cell_uid" in data.files else ""
        metadata_ok = bool(batch and protocol and cell_uid)
        if not require_metadata:
            metadata_ok = True

        upper_warn = bounds["upper_warn_V"]
        upper_fail = bounds["upper_fail_V"]
        lower_warn = bounds["lower_warn_V"]
        lower_fail = bounds["lower_fail_V"]

        phis_soft_max = float(np.nanmax(phis_soft))
        phis_soft_min = float(np.nanmin(phis_soft))
        phis_base_max = float(np.nanmax(phis_base)) if np.isfinite(phis_base).any() else float("nan")
        phis_base_min = float(np.nanmin(phis_base)) if np.isfinite(phis_base).any() else float("nan")
        phis_raw_max = float(np.nanmax(phis_raw)) if np.isfinite(phis_raw).any() else float("nan")
        phis_raw_min = float(np.nanmin(phis_raw)) if np.isfinite(phis_raw).any() else float("nan")

        upper_fail_count = int(np.sum(phis_soft > upper_fail))
        upper_warn_count = int(np.sum(phis_soft > upper_warn))
        lower_fail_count = int(np.sum(phis_soft < lower_fail))
        lower_warn_count = int(np.sum(phis_soft < lower_warn))
        voltage_fail = upper_fail_count > 0 or lower_fail_count > 0
        voltage_warn = upper_warn_count > 0 or lower_warn_count > 0

        if "voltage_bound_correction" in data.files:
            vbc = data["voltage_bound_correction"]
            max_abs_bound_correction = float(np.nanmax(np.abs(vbc))) if len(vbc) else 0.0
            nonzero_bound_correction_count = int(np.sum(np.abs(vbc) > 1e-7))
        else:
            max_abs_bound_correction = float("nan")
            nonzero_bound_correction_count = 0

        mae = float(np.mean(np.abs(phis_soft.astype(float) - vexp.astype(float))))
        corr = float(np.corrcoef(phis_soft.astype(float), vexp.astype(float))[0, 1]) if len(t) > 2 and np.std(phis_soft) > 0 and np.std(vexp) > 0 else float("nan")

        fail_reasons = []
        warn_reasons = []
        if not finite_ok:
            fail_reasons.append("nonfinite")
        if not time_ok:
            fail_reasons.append("time_not_monotonic_or_too_short")
        if not shape_ok:
            fail_reasons.append("shape_mismatch")
        if not prior_ok:
            fail_reasons.append("prior_hash_mismatch")
        if require_metadata and not metadata_ok:
            fail_reasons.append("missing_batch_protocol_or_cell_uid")
        if voltage_fail:
            fail_reasons.append("phis_c_soft_voltage_fail_bound")
        elif voltage_warn:
            warn_reasons.append("phis_c_soft_voltage_warn_bound")

        if fail_reasons:
            status = "FAIL"
            detail = "; ".join(fail_reasons)
        elif warn_reasons:
            status = "WARN"
            detail = "; ".join(warn_reasons)
        else:
            status = "PASS"
            detail = "ok"

        row.update({
            "status": status,
            "detail": detail,
            "n_points": int(len(t)),
            "n_r_a": int(cs_a.shape[1]) if cs_a.ndim == 2 else "",
            "n_r_c": int(cs_c.shape[1]) if cs_c.ndim == 2 else "",
            "time_monotonic_nondec": time_ok,
            "finite_ok": finite_ok,
            "shape_ok": shape_ok,
            "theta_a_oob_fraction": theta_a_oob,
            "theta_c_oob_fraction": theta_c_oob,
            "phis_c_vs_voltage_mae_V": mae,
            "phis_c_vs_voltage_corr": corr,
            "phis_c_soft_min_V": phis_soft_min,
            "phis_c_soft_max_V": phis_soft_max,
            "phis_c_soft_raw_min_V": phis_raw_min,
            "phis_c_soft_raw_max_V": phis_raw_max,
            "phis_c_base_min_V": phis_base_min,
            "phis_c_base_max_V": phis_base_max,
            "voltage_upper_warn_V": upper_warn,
            "voltage_upper_fail_V": upper_fail,
            "voltage_lower_warn_V": lower_warn,
            "voltage_lower_fail_V": lower_fail,
            "voltage_upper_warn_count": upper_warn_count,
            "voltage_upper_fail_count": upper_fail_count,
            "voltage_lower_warn_count": lower_warn_count,
            "voltage_lower_fail_count": lower_fail_count,
            "max_abs_voltage_bound_correction_V": max_abs_bound_correction,
            "nonzero_voltage_bound_correction_count": nonzero_bound_correction_count,
            "batch": batch,
            "protocol": protocol,
            "cell_uid": cell_uid,
            "metadata_ok": metadata_ok,
            "prior_hash_match": prior_ok,
            "resolved_spec_hash": prior_hash_file,
        })
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def write_audit_json(npz_path: str | Path, audit_row: Dict[str, Any]) -> None:
    p = Path(npz_path)
    (p.parent / "soft_label_audit.json").write_text(json.dumps(audit_row, ensure_ascii=False, indent=2), encoding="utf-8")
