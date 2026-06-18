from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    # deterministic, broad field order
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def canonical_contains(row: Dict[str, str], needles: Sequence[str]) -> bool:
    text = " ".join(str(v) for v in row.values()).lower()
    return all(n.lower() in text for n in needles)


def path_from_row(row: Dict[str, str], names: Sequence[str]) -> Optional[Path]:
    for n in names:
        v = row.get(n)
        if v:
            return Path(v)
    return None


def resolve_profile_rows(
    g0_profile_semantics_csv: str | Path,
    split_manifest: str | Path,
    profile_contains: Sequence[str],
    require_branch_contains: Optional[str] = "P4D",
) -> List[Dict[str, Any]]:
    rows = read_csv(g0_profile_semantics_csv)
    # If no explicit selection, choose all P4D/GEO/random_walk candidates from G0.
    if profile_contains:
        chosen: List[Dict[str, str]] = []
        for needle in profile_contains:
            # Accept exact or substring match.
            matches = [r for r in rows if canonical_contains(r, [needle])]
            if not matches:
                # Split tokens by punctuation to make Batch-6_GEO_battery-2 match Batch-6_battery-2 rows.
                toks = [t for t in re.split(r"[^A-Za-z0-9]+", needle) if t]
                matches = [r for r in rows if canonical_contains(r, toks)]
            if not matches:
                raise KeyError(f"No row in G0 semantics CSV matches --profile_contains {needle!r}")
            # Prefer P4D branch if available.
            if require_branch_contains:
                filt = [r for r in matches if require_branch_contains.lower() in str(r.get("semantic_branch", r.get("branch", ""))).lower()]
                if filt:
                    matches = filt
            chosen.append(matches[0])
    else:
        chosen = [r for r in rows if (not require_branch_contains or require_branch_contains.lower() in str(r.get("semantic_branch", r.get("branch", ""))).lower())]
    # Merge split manifest info if needed.
    manifest = read_json(split_manifest)
    recs = manifest.get("records", []) if isinstance(manifest, dict) else []
    out: List[Dict[str, Any]] = []
    for r in chosen:
        d: Dict[str, Any] = dict(r)
        can = str(r.get("canonical_cell_uid") or r.get("canonical_cell_id") or r.get("cell_uid") or r.get("cell_id") or "")
        if recs:
            m = None
            for rr in recs:
                rr_text = " ".join(str(v) for v in rr.values()).lower()
                if can and can.lower() in rr_text:
                    m = rr
                    break
            if m is None:
                # Try with source cell id e.g. Batch-6_battery-2.
                cell_uid = str(r.get("cell_uid") or r.get("cell_id") or "")
                for rr in recs:
                    rr_text = " ".join(str(v) for v in rr.values()).lower()
                    if cell_uid and cell_uid.lower() in rr_text:
                        m = rr
                        break
            if m:
                for k, v in m.items():
                    d.setdefault(k, v)
        # normalized column names
        d["canonical_cell_uid"] = d.get("canonical_cell_uid") or d.get("canonical_cell_id") or d.get("cell_uid") or d.get("cell_id") or ""
        d["cell_uid"] = d.get("cell_uid") or d.get("cell_id") or d.get("canonical_cell_uid")
        d["semantic_branch"] = d.get("semantic_branch") or d.get("branch") or d.get("source_branch") or "UNKNOWN"
        out.append(d)
    return out


def npz_header(path: str | Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}


def load_npz_subset(path: str | Path, keys: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        if keys is None:
            return {k: z[k] for k in z.files}
        out = {}
        for k in keys:
            if k in z.files:
                out[k] = z[k]
        return out


def get_1d(d: Dict[str, np.ndarray], candidates: Sequence[str], n: Optional[int] = None, required: bool = True, fill: float = 0.0) -> np.ndarray:
    for k in candidates:
        if k in d:
            arr = np.asarray(d[k])
            if arr.dtype.kind in {"U", "S", "O"}:
                continue
            arr = arr.astype(float).reshape(-1)
            if n is None:
                return arr
            if arr.size >= n:
                return arr[:n]
    if required:
        raise KeyError(f"Missing numeric key among {candidates}")
    assert n is not None
    return np.full(n, fill, dtype=float)


def volume_weights(nr: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, nr + 1)
    vols = edges[1:] ** 3 - edges[:-1] ** 3
    return vols / vols.sum()


def orient_time_radial(arr: np.ndarray, n_time_hint: Optional[int] = None, name: str = "arr") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 2D time-radial array, got {a.shape}")
    # Usually (time, nr), but support transposed.
    if a.shape[1] <= 64:
        return a.astype(float)
    if a.shape[0] <= 64:
        return a.T.astype(float)
    if n_time_hint is not None:
        if a.shape[0] == n_time_hint:
            return a.astype(float)
        if a.shape[1] == n_time_hint:
            return a.T.astype(float)
    raise ValueError(f"{name}: cannot infer time/radial orientation for shape {a.shape}")


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or max_points >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return float("nan")
    y = y[mask]
    yhat = yhat[mask]
    sse = float(np.sum((yhat - y) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - sse / max(sst, 1e-30))


def metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    err = yhat - y
    rng = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
    return {
        "r2": r2_score(y, yhat),
        "mae": float(np.mean(np.abs(err))) if y.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if y.size else float("nan"),
        "bias": float(np.mean(err)) if y.size else float("nan"),
        "nmae": float(np.mean(np.abs(err)) / rng) if y.size else float("nan"),
        "nrmse": float(np.sqrt(np.mean(err**2)) / rng) if y.size else float("nan"),
        "target_range": rng,
        "target_std": float(np.std(y)) if y.size else float("nan"),
        "pred_std": float(np.std(yhat)) if yhat.size else float("nan"),
    }


def cumulative_q_Ah(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    I = np.asarray(I, dtype=float).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt[dt < 0] = 0.0
    return np.cumsum(I * dt) / 3600.0


def cycle_reset_q_Ah(t: np.ndarray, I: np.ndarray, cycle_id: Optional[np.ndarray]) -> np.ndarray:
    if cycle_id is None:
        return cumulative_q_Ah(t, I)
    t = np.asarray(t, dtype=float).reshape(-1)
    I = np.asarray(I, dtype=float).reshape(-1)
    cid = np.asarray(cycle_id).reshape(-1)
    q = np.zeros_like(t, dtype=float)
    for c in np.unique(cid):
        idx = np.flatnonzero(cid == c)
        if idx.size == 0:
            continue
        q[idx] = cumulative_q_Ah(t[idx], I[idx])
    return q


def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    yhat = coef[0] + coef[1] * x
    return coef, yhat


def fit_fixed_slope(x: np.ndarray, y: np.ndarray, slope: float) -> Tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    b = float(np.nanmean(y - slope * x))
    return b, b + slope * x


def formula_candidates(
    t: np.ndarray,
    I: np.ndarray,
    cycle_id: Optional[np.ndarray],
    y: np.ndarray,
    target: str,
    theta0_cfg: float,
    window: float,
    capacity_Ah_cfg: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    q_full = cumulative_q_Ah(t, I)
    q_reset = cycle_reset_q_Ah(t, I, cycle_id)
    target_is_c = target.endswith("_c") or target == "theta_c"
    # Config expected signs from D15-P4D script: c sign=-1, a sign=+1.
    signs = [-1.0, 1.0] if target_is_c else [1.0, -1.0]
    for q_name, q in [("global_cumsum", q_full), ("cycle_reset_cumsum", q_reset)]:
        for sign in signs:
            slope = sign * window / max(float(capacity_Ah_cfg), 1e-12)
            yhat_cfg = np.clip(theta0_cfg + slope * q, 0.0, 1.0)
            m = metrics(y, yhat_cfg)
            out.append({
                "candidate": f"config_{q_name}_sign{sign:+.0f}",
                "q_mode": q_name,
                "fit_mode": "config_theta0_capacity",
                "theta0": theta0_cfg,
                "slope_per_Ah": slope,
                "capacity_Ah_implied": capacity_Ah_cfg,
                **m,
            })
            # same slope but free intercept: diagnoses theta0 mismatch.
            b, yhat_b = fit_fixed_slope(q, y, slope)
            m = metrics(y, np.clip(yhat_b, 0.0, 1.0))
            out.append({
                "candidate": f"fit_intercept_{q_name}_sign{sign:+.0f}",
                "q_mode": q_name,
                "fit_mode": "fit_theta0_fixed_capacity",
                "theta0": b,
                "slope_per_Ah": slope,
                "capacity_Ah_implied": capacity_Ah_cfg,
                **m,
            })
        # free slope and intercept: diagnoses capacity/sign mismatch.
        coef, yhat = fit_affine(q, y)
        slope = float(coef[1])
        cap_implied = abs(window / slope) if abs(slope) > 1e-12 else float("inf")
        m = metrics(y, np.clip(yhat, 0.0, 1.0))
        out.append({
            "candidate": f"fit_affine_{q_name}",
            "q_mode": q_name,
            "fit_mode": "fit_theta0_and_capacity",
            "theta0": float(coef[0]),
            "slope_per_Ah": slope,
            "capacity_Ah_implied": cap_implied,
            **m,
        })
    # cycle-wise affine is not deployable; it tells whether softlabel resets/has per-cycle offsets.
    if cycle_id is not None:
        cid = np.asarray(cycle_id).reshape(-1)
        q = q_reset
        yhat = np.zeros_like(y, dtype=float)
        ok = np.zeros_like(y, dtype=bool)
        for c in np.unique(cid):
            idx = np.flatnonzero(cid == c)
            if idx.size < 3:
                continue
            _, h = fit_affine(q[idx], y[idx])
            yhat[idx] = h
            ok[idx] = True
        if ok.any():
            m = metrics(y[ok], np.clip(yhat[ok], 0.0, 1.0))
            out.append({
                "candidate": "diagnostic_cyclewise_affine_NOT_DEPLOYABLE",
                "q_mode": "cycle_reset_cumsum",
                "fit_mode": "per_cycle_fit_theta0_and_capacity_diagnostic_only",
                "theta0": float("nan"),
                "slope_per_Ah": float("nan"),
                "capacity_Ah_implied": float("nan"),
                **m,
            })
    return out


def summarize_candidates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"status": "NO_ROWS"}
    best = max(rows, key=lambda r: (-1e99 if not np.isfinite(float(r.get("r2", np.nan))) else float(r.get("r2"))))
    cfg = [r for r in rows if str(r.get("candidate", "")).startswith("config_global_cumsum")]
    best_cfg = max(cfg, key=lambda r: (-1e99 if not np.isfinite(float(r.get("r2", np.nan))) else float(r.get("r2")))) if cfg else None
    return {
        "best_candidate": best.get("candidate"),
        "best_r2": best.get("r2"),
        "best_bias": best.get("bias"),
        "best_capacity_Ah_implied": best.get("capacity_Ah_implied"),
        "best_theta0": best.get("theta0"),
        "best_fit_mode": best.get("fit_mode"),
        "best_q_mode": best.get("q_mode"),
        "best_config_candidate": best_cfg.get("candidate") if best_cfg else None,
        "best_config_r2": best_cfg.get("r2") if best_cfg else None,
    }


def analyze_profile(row: Dict[str, Any], cfg: Dict[str, Any], out_dir: str | Path, max_time_points: int = 4096) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    soft_npz = path_from_row(row, ["softlabel_npz", "solution_softlabels_npz", "npz_path", "output_npz"])
    if soft_npz is None or not soft_npz.exists():
        # Try softlabel_dir.
        d = row.get("softlabel_dir") or row.get("profile_dir")
        if d:
            p = Path(d) / "solution_softlabels.npz"
            if p.exists():
                soft_npz = p
    if soft_npz is None or not soft_npz.exists():
        raise FileNotFoundError(f"Cannot resolve softlabel npz for row {row}")
    keys = [
        "t_global_s", "time_s", "t", "I_profile", "current_A", "I", "cycle_id",
        "theta_a", "theta_c", "cs_a", "cs_c", "cbar_a", "cbar_c",
        "radial_volume_weights_a", "radial_volume_weights_c", "r_a", "r_c",
    ]
    d = load_npz_subset(soft_npz, keys)
    # Determine n from theta/cs or time.
    if "theta_c" in d:
        theta_c = orient_time_radial(d["theta_c"], None, "theta_c")
        n = theta_c.shape[0]
    elif "cs_c" in d:
        cs_c = orient_time_radial(d["cs_c"], None, "cs_c")
        n = cs_c.shape[0]
    else:
        n = get_1d(d, ["t_global_s", "time_s", "t"]).size
    idx = downsample_indices(n, max_time_points)
    t = get_1d(d, ["t_global_s", "time_s", "t"], n=n)[idx]
    I = get_1d(d, ["I_profile", "current_A", "I"], n=n)[idx]
    cycle_id = None
    if "cycle_id" in d:
        try:
            cycle_id = np.asarray(d["cycle_id"]).reshape(-1)[:n][idx]
        except Exception:
            cycle_id = None
    prior = cfg.get("prior", {})
    csmax_a = float(cfg.get("csmax_a_mol_m3", prior.get("csmax_a_mol_m3", 31500.0)))
    csmax_c = float(cfg.get("csmax_c_mol_m3", prior.get("csmax_c_mol_m3", 50500.0)))
    theta_min_a = float(cfg.get("theta_min_a", 0.0079))
    theta_max_a = float(cfg.get("theta_max_a", 0.8544))
    theta_min_c = float(cfg.get("theta_min_c", 0.2535))
    theta_max_c = float(cfg.get("theta_max_c", 0.9149))
    cap = float(cfg.get("capacity_scale_Ah", 2.0))
    theta0_a = float(cfg.get("theta_negative_initial", 0.08))
    theta0_c = float(cfg.get("theta_positive_initial", 0.90))
    # targets as radial mean theta.
    target_series: Dict[str, np.ndarray] = {}
    if "theta_a" in d:
        a = orient_time_radial(d["theta_a"], n, "theta_a")
        w = np.asarray(d.get("radial_volume_weights_a", volume_weights(a.shape[1])), dtype=float).reshape(-1)
        target_series["theta_a_mean"] = (a[:n][idx] * w).sum(axis=1)
    elif "cs_a" in d:
        a = orient_time_radial(d["cs_a"], n, "cs_a")
        w = np.asarray(d.get("radial_volume_weights_a", volume_weights(a.shape[1])), dtype=float).reshape(-1)
        target_series["theta_a_mean"] = (a[:n][idx] * w).sum(axis=1) / csmax_a
    if "theta_c" in d:
        c = orient_time_radial(d["theta_c"], n, "theta_c")
        w = np.asarray(d.get("radial_volume_weights_c", volume_weights(c.shape[1])), dtype=float).reshape(-1)
        target_series["theta_c_mean"] = (c[:n][idx] * w).sum(axis=1)
    elif "cs_c" in d:
        c = orient_time_radial(d["cs_c"], n, "cs_c")
        w = np.asarray(d.get("radial_volume_weights_c", volume_weights(c.shape[1])), dtype=float).reshape(-1)
        target_series["theta_c_mean"] = (c[:n][idx] * w).sum(axis=1) / csmax_c
    all_rows: List[Dict[str, Any]] = []
    for target_name, y in target_series.items():
        if target_name.startswith("theta_a"):
            cands = formula_candidates(t, I, cycle_id, y, "theta_a", theta0_a, theta_max_a - theta_min_a, cap)
        else:
            cands = formula_candidates(t, I, cycle_id, y, "theta_c", theta0_c, theta_max_c - theta_min_c, cap)
        for r in cands:
            rr = {
                "canonical_cell_uid": row.get("canonical_cell_uid", ""),
                "cell_uid": row.get("cell_uid", ""),
                "protocol": row.get("protocol", ""),
                "semantic_branch": row.get("semantic_branch", ""),
                "softlabel_npz": str(soft_npz),
                "target": target_name,
                "n_points": int(y.size),
                "time_min": float(np.nanmin(t)),
                "time_max": float(np.nanmax(t)),
                **r,
            }
            all_rows.append(rr)
    target_summaries = {}
    for target_name in target_series.keys():
        target_summaries[target_name] = summarize_candidates([r for r in all_rows if r["target"] == target_name])
    # decision logic.
    min_best = min(float(v.get("best_r2", -999.0)) for v in target_summaries.values()) if target_summaries else -999.0
    min_config = min(float(v.get("best_config_r2", -999.0)) for v in target_summaries.values()) if target_summaries else -999.0
    if min_config >= float(cfg.get("pass_r2_gate", 0.99)):
        rec = "CONFIG_FORMULA_MATCHES_SOFTLABEL_FIX_SURROGATE_FEATURES_OR_AUDIT_ONLY"
        ready = True
    elif min_best >= float(cfg.get("pass_r2_gate", 0.99)):
        rec = "FORMULA_FAMILY_MATCHES_AFTER_PARAMETER_OR_SIGN_FIT_USE_THIS_TO_PATCH_P4D_LOGIC"
        ready = True
    elif min_best >= float(cfg.get("review_r2_gate", 0.90)):
        rec = "PARTIAL_FORMULA_MATCH_REVIEW_CAPACITY_THETA0_SIGN_AND_CYCLE_RESET"
        ready = False
    else:
        rec = "NO_SIMPLE_CURRENT_INTEGRAL_FORMULA_MATCH_STOP_AND_INSPECT_GENERATOR_OUTPUT_METADATA"
        ready = False
    summary: Dict[str, Any] = {
        "canonical_cell_uid": row.get("canonical_cell_uid", ""),
        "cell_uid": row.get("cell_uid", ""),
        "protocol": row.get("protocol", ""),
        "semantic_branch": row.get("semantic_branch", ""),
        "softlabel_npz": str(soft_npz),
        "n_points_loaded": int(n),
        "n_points_analyzed": int(len(idx)),
        "time_range_analyzed": [float(np.nanmin(t)), float(np.nanmax(t))],
        "cycle_range_analyzed": str(f"{np.nanmin(cycle_id)}-{np.nanmax(cycle_id)}") if cycle_id is not None else "NA",
        "target_summaries": target_summaries,
        "min_best_r2": min_best,
        "min_config_r2": min_config,
        "formula_ready": bool(ready),
        "recommendation": rec,
    }
    return summary, all_rows


def run_forensics(args: Any) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = read_json(args.config)
    rows = resolve_profile_rows(args.g0_profile_semantics_csv, args.split_manifest, args.profile_contains, require_branch_contains=cfg.get("require_branch_contains", "P4D"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_summaries: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for r in rows:
        try:
            s, mrows = analyze_profile(r, cfg, out_dir, max_time_points=int(args.max_time_points))
            profile_summaries.append(s)
            metric_rows.extend(mrows)
        except Exception as exc:
            failures.append({"row": dict(r), "error": repr(exc)})
    write_csv(metric_rows, out_dir / "D17_G63_FORMULA_CANDIDATE_METRICS.csv")
    write_json(profile_summaries, out_dir / "D17_G63_PROFILE_FORMULA_SUMMARIES.json")
    min_best = min([float(s.get("min_best_r2", -999.0)) for s in profile_summaries], default=-999.0)
    min_config = min([float(s.get("min_config_r2", -999.0)) for s in profile_summaries], default=-999.0)
    ready_count = sum(1 for s in profile_summaries if s.get("formula_ready"))
    total = len(profile_summaries)
    if failures:
        status = "REVIEW"
        rec = "FIX_LOAD_FAILURES_BEFORE_ANY_PATCH_OR_TRAINING"
        blockers = [f"load failures: {len(failures)}"]
        patch_ready = False
    elif total == 0:
        status = "REVIEW"
        rec = "NO_PROFILES_SELECTED"
        blockers = ["No profiles were evaluated"]
        patch_ready = False
    elif ready_count == total and min_best >= float(cfg.get("pass_r2_gate", 0.99)):
        status = "PASS"
        rec = "PROCEED_TO_MINIMAL_P4D_PATCH_USING_IDENTIFIED_FORMULA_FAMILY"
        blockers = []
        patch_ready = True
    elif min_best >= float(cfg.get("review_r2_gate", 0.90)):
        status = "REVIEW"
        rec = "PARTIAL_MATCH_DO_NOT_TRAIN_REVIEW_FORMULA_SUMMARIES"
        blockers = [f"min best formula R2 below pass gate: {min_best:.6g}"]
        patch_ready = False
    else:
        status = "REVIEW"
        rec = "STOP_NO_FORMULA_EQUIVALENCE_DO_NOT_TRAIN_OR_PATCH"
        blockers = [f"min best formula R2 too low: {min_best:.6g}"]
        patch_ready = False
    summary = {
        "protocol": "D17-G6.3_P4D_GENERATOR_FORMULA_FORENSICS",
        "status": status,
        "patch_ready": patch_ready,
        "recommendation": rec,
        "blockers": blockers,
        "selected_profile_count": len(rows),
        "evaluated_profile_count": total,
        "failure_count": len(failures),
        "min_best_formula_r2": min_best,
        "min_config_formula_r2": min_config,
        "ready_profile_count": ready_count,
        "elapsed_s": float(time.perf_counter() - t0),
        "max_time_points": int(args.max_time_points),
        "profile_summaries_json": str(out_dir / "D17_G63_PROFILE_FORMULA_SUMMARIES.json"),
        "candidate_metrics_csv": str(out_dir / "D17_G63_FORMULA_CANDIDATE_METRICS.csv"),
        "failures": failures,
    }
    write_json(summary, out_dir / "D17_G63_P4D_FORMULA_FORENSICS_SUMMARY.json")
    return summary
