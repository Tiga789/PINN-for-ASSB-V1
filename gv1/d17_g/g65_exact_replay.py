from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------- IO helpers -----------------------------

def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def norm_text(s: Any) -> str:
    return str(s or "").replace("\\", "/").lower()


# ----------------------------- metrics -----------------------------

def r2_score(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yp = np.asarray(yp, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yp)
    if int(mask.sum()) < 3:
        return float("nan")
    y = y[mask]
    yp = yp[mask]
    sse = float(np.sum((yp - y) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= 1e-30:
        return float("nan")
    return 1.0 - sse / sst


def mae(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yp = np.asarray(yp, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(yp[mask] - y[mask])))


def rmse(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yp = np.asarray(yp, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yp[mask] - y[mask]) ** 2)))


def bias(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yp = np.asarray(yp, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(yp[mask] - y[mask]))


# ----------------------------- data extraction -----------------------------

def _first_key(z: Any, keys: Sequence[str]) -> Optional[str]:
    files = set(z.files if hasattr(z, "files") else z.keys())
    for k in keys:
        if k in files:
            return k
    return None


def get_1d(z: Any, keys: Sequence[str], n: Optional[int] = None, required: bool = True, fill: float = 0.0) -> np.ndarray:
    k = _first_key(z, keys)
    if k is None:
        if required:
            raise KeyError("Missing keys: " + ",".join(keys))
        if n is None:
            raise ValueError("n must be supplied when missing optional key")
        return np.full(n, fill, dtype=float)
    arr = np.asarray(z[k])
    if arr.dtype.kind in {"U", "S", "O"}:
        if required:
            raise TypeError(f"Key {k!r} is non-numeric dtype={arr.dtype}")
        return np.full(n or arr.size, fill, dtype=float)
    arr = arr.astype(float).reshape(-1)
    if n is not None:
        arr = arr[:n]
    return arr


def radial_weights(z: Any, electrode: str, nr: int) -> np.ndarray:
    keys = [f"radial_volume_weights_{electrode}", f"volume_weights_{electrode}", "radial_volume_weights"]
    k = _first_key(z, keys)
    if k is not None:
        w = np.asarray(z[k], dtype=float).reshape(-1)
        if w.size == nr and np.sum(w) > 0:
            return w / np.sum(w)
    # fallback spherical shell center weights on normalized equal centers
    r = np.linspace(0.5 / nr, 1.0 - 0.5 / nr, nr)
    w = r ** 2
    return w / np.sum(w)


def orient_time_radial(arr: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        if a.shape[0] != n:
            raise ValueError(f"{name}: expected n={n}, got {a.shape}")
        return a.reshape(n, 1).astype(float)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 1D/2D, got shape={a.shape}")
    if a.shape[0] == n:
        return a.astype(float)
    if a.shape[1] == n:
        return a.T.astype(float)
    raise ValueError(f"{name}: cannot orient {a.shape} for n={n}")


def target_theta_mean(z: Any, electrode: str, n: int, prior_csmax: Optional[float]) -> Tuple[np.ndarray, str]:
    # Highest-fidelity mean target is cbar/csmax if both exist.
    cbar_key = _first_key(z, [f"cbar_{electrode}", f"cbar_{electrode}_soft"])
    if cbar_key and prior_csmax and prior_csmax > 0:
        return np.asarray(z[cbar_key], dtype=float).reshape(-1)[:n] / float(prior_csmax), f"{cbar_key}/prior_csmax"
    theta_key = _first_key(z, [f"theta_{electrode}", f"theta_{electrode}_soft"])
    if theta_key:
        th = orient_time_radial(np.asarray(z[theta_key]), n, theta_key)
        if th.shape[1] == 1:
            return th[:, 0], theta_key
        w = radial_weights(z, electrode, th.shape[1])
        return np.sum(th * w.reshape(1, -1), axis=1), f"volume_mean({theta_key})"
    cs_key = _first_key(z, [f"cs_{electrode}", f"cs_{electrode}_soft"])
    if cs_key and prior_csmax and prior_csmax > 0:
        cs = orient_time_radial(np.asarray(z[cs_key]), n, cs_key)
        w = radial_weights(z, electrode, cs.shape[1])
        return np.sum(cs * w.reshape(1, -1), axis=1) / float(prior_csmax), f"volume_mean({cs_key})/prior_csmax"
    raise KeyError(f"Cannot build theta_{electrode} mean target")


def select_indices(n: int, max_points: int) -> np.ndarray:
    if max_points is None or int(max_points) <= 0 or int(max_points) >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))


# ----------------------------- prior/config extraction -----------------------------

def resolve_path(project_root: str | Path, p: str | Path | None) -> Optional[Path]:
    if not p:
        return None
    q = Path(p)
    if q.exists():
        return q
    r = Path(project_root) / q
    return r if r.exists() else q


def load_generation_config(project_root: str | Path, config_path: str | Path | None) -> Dict[str, Any]:
    if not config_path:
        config_path = "configs/d15_p4d_full_remaining14_config.json"
    p = resolve_path(project_root, config_path)
    if p and p.exists():
        cfg = read_json(p)
        cfg["__path__"] = str(p)
        return cfg
    return {"generation": {}, "__path__": str(config_path)}


def load_prior_from_config(project_root: str | Path, cfg: Dict[str, Any], prior_json: Optional[str] = None) -> Dict[str, Any]:
    p = resolve_path(project_root, prior_json or cfg.get("prior_json"))
    if p and p.exists():
        prior = read_json(p)
        prior["__path__"] = str(p)
        return prior
    return {"__path__": str(prior_json or cfg.get("prior_json") or "")}


def prior_windows_and_csmax(prior: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out = {"a": {}, "c": {}}
    try:
        neg = prior.get("electrodes", {}).get("negative", {})
        pos = prior.get("electrodes", {}).get("positive", {})
        out["a"] = {
            "theta_min": float(neg.get("theta_min", 0.0079)),
            "theta_max": float(neg.get("theta_max", 0.8544)),
            "window": float(neg.get("theta_max", 0.8544)) - float(neg.get("theta_min", 0.0079)),
            "csmax": float(neg.get("csmax_mol_m3", np.nan)),
        }
        out["c"] = {
            "theta_min": float(pos.get("theta_min", 0.2535)),
            "theta_max": float(pos.get("theta_max", 0.9149)),
            "window": float(pos.get("theta_max", 0.9149)) - float(pos.get("theta_min", 0.2535)),
            "csmax": float(pos.get("csmax_mol_m3", np.nan)),
        }
    except Exception:
        pass
    return out


# ----------------------------- formula candidates -----------------------------

def safe_dt(t: np.ndarray, mode: str) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if mode == "prepend_t0":
        dt = np.diff(t, prepend=t[0])
    elif mode == "prepend_zero":
        dt = np.diff(t, prepend=0.0)
    elif mode == "prepend_first_interval":
        first = t[1] - t[0] if t.size > 1 else 0.0
        dt = np.diff(t, prepend=t[0] - first)
    else:
        dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt[dt < 0] = 0.0
    return dt


def cumulative_ah(t: np.ndarray, I: np.ndarray, mode: str, cycle: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    I = np.asarray(I, dtype=float).reshape(-1)
    if mode.startswith("trapz"):
        dt_mode = mode.split("__", 1)[1] if "__" in mode else "prepend_t0"
        dt = safe_dt(t, dt_mode)
        Iprev = np.r_[I[0], I[:-1]]
        inc = 0.5 * (I + Iprev) * dt / 3600.0
    else:
        dt_mode = mode.split("__", 1)[1] if "__" in mode else mode
        dt = safe_dt(t, dt_mode)
        inc = I * dt / 3600.0
    if cycle is not None and mode.startswith("cycle_reset"):
        q = np.zeros_like(inc, dtype=float)
        cyc = np.asarray(cycle).reshape(-1)
        acc = 0.0
        prev = cyc[0] if cyc.size else None
        for i, val in enumerate(inc):
            if i == 0 or cyc[i] != prev:
                acc = 0.0
                prev = cyc[i]
            acc += val
            q[i] = acc
        return q
    return np.cumsum(inc)


def fit_affine(q: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    q = np.asarray(q, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(q) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.nanstd(q[mask]) <= 1e-30:
        return float("nan"), float("nan"), np.full_like(y, np.nan)
    A = np.vstack([np.ones(mask.sum()), q[mask]]).T
    coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, a + b * q


def make_formula_candidates(
    y: np.ndarray,
    t: np.ndarray,
    I: np.ndarray,
    cycle: Optional[np.ndarray],
    electrode: str,
    gen: Dict[str, Any],
    prior_e: Dict[str, float],
) -> List[Dict[str, Any]]:
    expected_sign = -1.0 if electrode == "c" else +1.0
    theta0_key = "theta_positive_initial" if electrode == "c" else "theta_negative_initial"
    theta0_cfg = float(gen.get(theta0_key, 0.90 if electrode == "c" else 0.08))
    cap_cfg = float(gen.get("capacity_scale_Ah", 2.0))
    win_cfg = float(prior_e.get("window", np.nan))
    if not np.isfinite(win_cfg) or win_cfg <= 0:
        win_cfg = float(np.nanmax(y) - np.nanmin(y))
    theta0_target = float(y[0]) if y.size else theta0_cfg
    q_modes = [
        "prepend_t0",
        "prepend_zero",
        "prepend_first_interval",
        "trapz__prepend_t0",
        "trapz__prepend_first_interval",
    ]
    if cycle is not None:
        q_modes += ["cycle_reset__prepend_t0", "cycle_reset__prepend_first_interval"]
    signs = [expected_sign, -expected_sign]
    out: List[Dict[str, Any]] = []
    for qmode in q_modes:
        q = cumulative_ah(t, I, qmode, cycle=cycle)
        for isign in [1.0, -1.0]:
            qq = isign * q
            qname = qmode + ("__I" if isign > 0 else "__negI")
            # exact config formula
            for sign in signs:
                pred = theta0_cfg + sign * (qq / max(cap_cfg, 1e-12)) * win_cfg
                out.append({"candidate": f"CONFIG_{qname}_sign{sign:+.0f}_clip", "deployable": True, "uses_target_values": False, "theta0": theta0_cfg, "window": win_cfg, "capacity_Ah": cap_cfg, "pred": np.clip(pred, 0.0, 1.0)})
                out.append({"candidate": f"CONFIG_{qname}_sign{sign:+.0f}_noclip", "deployable": True, "uses_target_values": False, "theta0": theta0_cfg, "window": win_cfg, "capacity_Ah": cap_cfg, "pred": pred})
            # source target first only: diagnostic; if this works config theta0 mismatch is likely
            for sign in signs:
                pred = theta0_target + sign * (qq / max(cap_cfg, 1e-12)) * win_cfg
                out.append({"candidate": f"TARGET_FIRST_{qname}_sign{sign:+.0f}_clip", "deployable": False, "uses_target_values": True, "theta0": theta0_target, "window": win_cfg, "capacity_Ah": cap_cfg, "pred": np.clip(pred, 0.0, 1.0)})
            # affine best possible diagnostic
            a, b, pred = fit_affine(qq, y)
            out.append({"candidate": f"DIAG_AFFINE_{qname}", "deployable": False, "uses_target_values": True, "theta0": a, "slope_per_Ah": b, "pred": pred})
    return out


def summarize_candidates(y: np.ndarray, candidates: List[Dict[str, Any]], electrode: str, profile_label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        yp = np.asarray(c.pop("pred"), dtype=float)
        rows.append({
            "canonical_cell_uid": profile_label,
            "electrode": electrode,
            "candidate": c.get("candidate"),
            "deployable": bool(c.get("deployable")),
            "uses_target_values": bool(c.get("uses_target_values")),
            "r2": r2_score(y, yp),
            "mae": mae(y, yp),
            "rmse": rmse(y, yp),
            "bias": bias(y, yp),
            "target_min": float(np.nanmin(y)),
            "target_max": float(np.nanmax(y)),
            "target_std": float(np.nanstd(y)),
            **{k: v for k, v in c.items() if k not in {"candidate", "deployable", "uses_target_values"}},
        })
    rows.sort(key=lambda r: (-999.0 if not np.isfinite(r.get("r2", np.nan)) else -float(r["r2"])))
    return rows


# ----------------------------- main protocol -----------------------------

def load_g64_profiles(g64_dir: str | Path, profile_contains: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    p = Path(g64_dir) / "D17_G64_PROFILE_PROVENANCE_DETAILS.json"
    if not p.exists():
        raise FileNotFoundError(f"G64 profile details not found: {p}")
    obj = read_json(p)
    profiles = obj.get("profiles", []) if isinstance(obj, dict) else []
    if not profiles:
        raise RuntimeError(f"No profiles in {p}")
    if profile_contains:
        selected = []
        for needle in profile_contains:
            nt = norm_text(needle)
            matches = [r for r in profiles if nt in norm_text(json.dumps(r, ensure_ascii=False))]
            if not matches:
                raise KeyError(f"No G64 profile detail matches {needle!r}")
            selected.append(matches[0])
        return selected
    return profiles


def run_exact_replay(args: Any) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(args.config) if getattr(args, "config", None) and Path(args.config).exists() else {}
    g64_profiles = load_g64_profiles(args.g64_dir, getattr(args, "profile_contains", None))
    gen_cfg = load_generation_config(args.project_root, getattr(args, "d15_p4d_config", None) or cfg.get("d15_p4d_config"))
    prior = load_prior_from_config(args.project_root, gen_cfg, getattr(args, "prior_json", None) or cfg.get("prior_json"))
    gen = dict(gen_cfg.get("generation", {}))
    windows = prior_windows_and_csmax(prior)
    max_points = int(getattr(args, "max_time_points", 4096) or 4096)

    candidate_rows: List[Dict[str, Any]] = []
    profile_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for prof in g64_profiles:
        label = str(prof.get("canonical_cell_uid") or prof.get("cell_uid") or "UNKNOWN")
        paths = prof.get("paths", {})
        soft_npz = Path(paths.get("softlabel_npz") or "")
        if not soft_npz.exists():
            failures.append({"canonical_cell_uid": label, "error": f"softlabel npz missing: {soft_npz}"})
            continue
        try:
            with np.load(soft_npz, allow_pickle=True) as z:
                n = int(np.asarray(z[_first_key(z, ["t_global_s", "time_s", "t"])]).reshape(-1).size)
                idx = select_indices(n, max_points)
                t = get_1d(z, ["t_global_s", "time_s", "t"], n=n)[idx]
                I = get_1d(z, ["I_profile", "current_A", "I"], n=n)[idx]
                cyc = None
                ck = _first_key(z, ["cycle_id", "cycle", "cycle_index"])
                if ck is not None:
                    cyc = np.asarray(z[ck]).reshape(-1)[:n][idx]
                y_a, src_a = target_theta_mean(z, "a", n, windows.get("a", {}).get("csmax"))
                y_c, src_c = target_theta_mean(z, "c", n, windows.get("c", {}).get("csmax"))
                y_a = y_a[idx]
                y_c = y_c[idx]
            rows_a = summarize_candidates(y_a, make_formula_candidates(y_a, t, I, cyc, "a", gen, windows["a"]), "a", label)
            rows_c = summarize_candidates(y_c, make_formula_candidates(y_c, t, I, cyc, "c", gen, windows["c"]), "c", label)
            candidate_rows.extend(rows_a)
            candidate_rows.extend(rows_c)
            best_a = rows_a[0] if rows_a else {}
            best_c = rows_c[0] if rows_c else {}
            best_deploy_a = next((r for r in rows_a if r.get("deployable")), {})
            best_deploy_c = next((r for r in rows_c if r.get("deployable")), {})
            profile_summaries.append({
                "canonical_cell_uid": label,
                "semantic_branch": prof.get("semantic_branch"),
                "softlabel_npz": str(soft_npz),
                "n_total": n,
                "n_evaluated": int(len(idx)),
                "theta_a_target_source": src_a,
                "theta_c_target_source": src_c,
                "best_any_theta_a_candidate": best_a.get("candidate"),
                "best_any_theta_a_r2": best_a.get("r2"),
                "best_any_theta_c_candidate": best_c.get("candidate"),
                "best_any_theta_c_r2": best_c.get("r2"),
                "best_deployable_theta_a_candidate": best_deploy_a.get("candidate"),
                "best_deployable_theta_a_r2": best_deploy_a.get("r2"),
                "best_deployable_theta_c_candidate": best_deploy_c.get("candidate"),
                "best_deployable_theta_c_r2": best_deploy_c.get("r2"),
                "best_deployable_theta_a_bias": best_deploy_a.get("bias"),
                "best_deployable_theta_c_bias": best_deploy_c.get("bias"),
            })
        except Exception as e:
            failures.append({"canonical_cell_uid": label, "softlabel_npz": str(soft_npz), "error": repr(e)})

    write_csv(candidate_rows, out_dir / "D17_G65_FORMULA_CANDIDATE_METRICS.csv")
    write_json({"profiles": profile_summaries, "failures": failures}, out_dir / "D17_G65_PROFILE_FORMULA_SUMMARIES.json")

    def fnum(v: Any, default: float = float("nan")) -> float:
        try:
            return float(v)
        except Exception:
            return default

    deploy_r2 = []
    any_r2 = []
    for ps in profile_summaries:
        deploy_r2 += [fnum(ps.get("best_deployable_theta_a_r2")), fnum(ps.get("best_deployable_theta_c_r2"))]
        any_r2 += [fnum(ps.get("best_any_theta_a_r2")), fnum(ps.get("best_any_theta_c_r2"))]
    deploy_r2 = [x for x in deploy_r2 if np.isfinite(x)]
    any_r2 = [x for x in any_r2 if np.isfinite(x)]
    min_deploy = float(np.min(deploy_r2)) if deploy_r2 else float("nan")
    min_any = float(np.min(any_r2)) if any_r2 else float("nan")
    gate = float(cfg.get("exact_replay_min_r2_gate", 0.999))
    loose_gate = float(cfg.get("diagnostic_affine_min_r2_gate", 0.995))
    blockers = []
    if failures:
        blockers.append(f"profile failures: {len(failures)}")
    if not deploy_r2 or min_deploy < gate:
        blockers.append(f"deployable exact provenance formula min R2 below gate {gate}: {min_deploy:.6g}")
    exact_ready = (not blockers)
    if exact_ready:
        rec = "RUN_G66_DETERMINISTIC_P4D_INVENTORY_OVERRIDE_OR_SELECTED_CYCLE_INFERENCE"
    elif any_r2 and min_any >= loose_gate:
        rec = "STOP_CONFIG_MISMATCH_ONLY_AFFINE_DIAGNOSTIC_WORKS_DO_NOT_DEPLOY_PATCH_YET"
    else:
        rec = "STOP_NO_EXACT_REPLAY_EQUIVALENCE_DO_NOT_TRAIN_OR_PATCH"

    summary = {
        "protocol": "D17-G6.5_EXACT_PROVENANCE_REPLAY_TEST",
        "status": "PASS" if not failures else "REVIEW",
        "exact_replay_ready": bool(exact_ready),
        "patch_ready": bool(exact_ready),
        "recommendation": rec,
        "blockers": blockers,
        "selected_profile_count": len(g64_profiles),
        "evaluated_profile_count": len(profile_summaries),
        "failure_count": len(failures),
        "min_deployable_formula_r2": min_deploy,
        "min_any_formula_r2": min_any,
        "elapsed_s": float(time.perf_counter() - t0),
        "max_time_points": max_points,
        "d15_p4d_config": gen_cfg.get("__path__"),
        "prior_json": prior.get("__path__"),
        "generation_config_used": gen,
        "outputs": {
            "summary_json": str(out_dir / "D17_G65_EXACT_PROVENANCE_REPLAY_SUMMARY.json"),
            "candidate_metrics_csv": str(out_dir / "D17_G65_FORMULA_CANDIDATE_METRICS.csv"),
            "profile_summaries_json": str(out_dir / "D17_G65_PROFILE_FORMULA_SUMMARIES.json"),
        },
        "policy": {
            "training_performed": False,
            "radial_solver_run": False,
            "full_55cell_audit_run": False,
            "uses_softlabel_targets_for_report_only_formula_equivalence": True,
        },
    }
    write_json(summary, out_dir / "D17_G65_EXACT_PROVENANCE_REPLAY_SUMMARY.json")
    return summary
