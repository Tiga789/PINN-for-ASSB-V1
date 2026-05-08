# -*- coding: utf-8 -*-
r"""
ASSB all-cycle soft-label checker.

Default target:
    C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_allcycle\solution.npz

Checks:
  1) NPZ fields/shapes/time/cycles/current/rest/flux signs.
  2) phis_c vs voltage_exp global fixed-B residual, if voltage_exp exists.
  3) SPM spherical-average mass closure for cs_a/cs_c versus j_a/j_c.

The mass-closure check is the important new guard: it catches the previous
failure mode where cs_c/theta_c did not close with I(t)/j_c even though the
file structure looked valid.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import zipfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.lib import format as npfmt

DEFAULT_SOLUTION = r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_allcycle\solution.npz"

TIME_KEYS = ("t_global_s", "t", "time_s", "t_s", "time")
CYCLE_KEYS = ("cycle_id", "cycle", "cycle_index")
I_KEYS = ("I_profile", "I", "current_A", "current")
VEXP_KEYS = ("voltage_exp", "V_exp", "voltage", "V")
POT_KEYS = ("phis_c", "phi_s_c", "V_soft", "voltage_soft")
PHIE_KEYS = ("phie", "phi_e")


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _headers(npz_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(npz_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = Path(name).stem
            with zf.open(name, "r") as fp:
                version = npfmt.read_magic(fp)
                if version == (1, 0):
                    shape, fortran_order, dtype = npfmt.read_array_header_1_0(fp)
                elif version in ((2, 0), (3, 0)):
                    shape, fortran_order, dtype = npfmt.read_array_header_2_0(fp)
                else:
                    raise RuntimeError(f"Unsupported .npy header version {version} for {name}")
            info = zf.getinfo(name)
            out[key] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
                "compressed_size_bytes": int(info.compress_size),
                "uncompressed_size_bytes": int(info.file_size),
            }
    return out


def _find_key(keys: Iterable[str], names: Iterable[str]) -> Optional[str]:
    key_list = list(keys)
    lower = {k.lower(): k for k in key_list}
    for n in names:
        if n in key_list:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _load(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        return z[key]


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    d: Dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        d.update({"finite_count": int(finite.sum()), "nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum())})
        if finite.any():
            v = arr[finite]
            d.update({"min": float(np.min(v)), "max": float(np.max(v)), "mean": float(np.mean(v))})
    else:
        flat = arr.reshape(-1)
        d["sample"] = [str(x) for x in flat[:10]]
    return d


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return None
    aa, bb = a[m], b[m]
    if float(np.std(aa)) <= 0 or float(np.std(bb)) <= 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _as_Nt_Nr(arr: np.ndarray, Nt: int, Nr: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (Nt, Nr):
        return arr.astype(np.float64, copy=False)
    if arr.shape == (Nr, Nt):
        return arr.T.astype(np.float64, copy=False)
    if arr.ndim == 1 and arr.size == Nt * Nr:
        return arr.reshape(Nt, Nr).astype(np.float64, copy=False)
    raise ValueError(f"{name} shape {arr.shape} cannot be interpreted as (Nt,Nr)=({Nt},{Nr})")


def _fv_spherical_weights(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size < 2:
        raise ValueError("radial grid must have at least 2 points")
    R = float(r[-1])
    edges = np.empty(r.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = R
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    w = (edges[1:] ** 3 - edges[:-1] ** 3) / (R ** 3)
    return w / np.sum(w)


def _mass_closure(npz_path: Path, keys: set[str], t: np.ndarray, electrode: str, mae_fail: float, mae_warn: float) -> Dict[str, Any]:
    if electrode == "a":
        rkey = _find_key(keys, ("r_a", "r_an", "r_negative"))
        cskey = _find_key(keys, ("cs_a", "c_s_a", "csa"))
        jkey = _find_key(keys, ("j_a", "J_a", "flux_a"))
    else:
        rkey = _find_key(keys, ("r_c", "r_ca", "r_positive"))
        cskey = _find_key(keys, ("cs_c", "c_s_c", "csc"))
        jkey = _find_key(keys, ("j_c", "J_c", "flux_c"))
    if not (rkey and cskey and jkey):
        return {"status": "SKIP", "reason": f"missing one of r/cs/j for electrode {electrode}", "rkey": rkey, "cskey": cskey, "jkey": jkey}

    r = _load(npz_path, rkey).astype(np.float64).reshape(-1)
    j = _load(npz_path, jkey).astype(np.float64).reshape(-1)
    cs_raw = _load(npz_path, cskey)
    cs = _as_Nt_Nr(cs_raw, t.size, r.size, cskey)
    w = _fv_spherical_weights(r)
    cbar = cs @ w
    dt = np.diff(t)
    rate = -3.0 * j / float(r[-1])

    pred_right = np.empty_like(cbar)
    pred_left = np.empty_like(cbar)
    pred_right[0] = cbar[0]
    pred_left[0] = cbar[0]
    pred_right[1:] = cbar[0] + np.cumsum(rate[1:] * dt)
    pred_left[1:] = cbar[0] + np.cumsum(rate[:-1] * dt)

    def metrics(pred: np.ndarray) -> Dict[str, float | None]:
        e = pred - cbar
        return {
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "maxabs": float(np.max(np.abs(e))),
            "bias": float(np.mean(e)),
            "corr": _corr(pred, cbar),
        }

    m_right = metrics(pred_right)
    m_left = metrics(pred_left)
    best_name = "right_endpoint" if (m_right["mae"] or 1e99) <= (m_left["mae"] or 1e99) else "left_endpoint"
    best = m_right if best_name == "right_endpoint" else m_left
    status = "PASS"
    if best["mae"] is not None and best["mae"] > mae_fail:
        status = "FAIL"
    elif best["mae"] is not None and best["mae"] > mae_warn:
        status = "WARN"
    return {
        "status": status,
        "electrode": electrode,
        "r_key": rkey,
        "cs_key": cskey,
        "j_key": jkey,
        "R_m": float(r[-1]),
        "cbar_start": float(cbar[0]),
        "cbar_end": float(cbar[-1]),
        "best_integrator_endpoint": best_name,
        "right_endpoint": m_right,
        "left_endpoint": m_left,
        "thresholds": {"mae_warn": mae_warn, "mae_fail": mae_fail},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect ASSB all-cycle soft-label solution.npz")
    p.add_argument("--solution", default=DEFAULT_SOLUTION)
    p.add_argument("--expected_cycle_from", type=int, default=5)
    p.add_argument("--expected_cycle_to", type=int, default=522)
    p.add_argument("--expected_nr", type=int, default=64)
    p.add_argument("--quick", action="store_true", help="Skip cs_a/cs_c mass-closure checks")
    p.add_argument("--mass_mae_warn", type=float, default=1.0e-3)
    p.add_argument("--mass_mae_fail", type=float, default=1.0e-2)
    p.add_argument("--write_report", action="store_true", default=True)
    args = p.parse_args()

    sol = Path(args.solution)
    report: Dict[str, Any] = {"solution_path": str(sol), "status": "UNKNOWN", "errors": [], "warnings": []}
    if not sol.exists():
        report["status"] = "FAIL"
        report["errors"].append(f"solution.npz not found: {sol}")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable))
        return 2

    report["file_size_gb"] = round(sol.stat().st_size / (1024 ** 3), 4)
    try:
        hdr = _headers(sol)
    except Exception as e:
        report["status"] = "FAIL"
        report["errors"].append(f"Cannot read npz headers: {type(e).__name__}: {e}")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable))
        return 2

    keys = set(hdr)
    report["array_count"] = len(keys)
    report["array_keys"] = sorted(keys)
    report["arrays_header"] = hdr

    tkey = _find_key(keys, TIME_KEYS)
    ikey = _find_key(keys, I_KEYS)
    ckey = _find_key(keys, CYCLE_KEYS)
    pkey = _find_key(keys, POT_KEYS)
    phiekey = _find_key(keys, PHIE_KEYS)
    vkey = _find_key(keys, VEXP_KEYS)
    report["resolved_keys"] = {"time": tkey, "cycle": ckey, "I": ikey, "phis_c": pkey, "phie": phiekey, "voltage_exp": vkey}

    for label, key in [("time", tkey), ("I", ikey), ("phis_c", pkey), ("phie", phiekey)]:
        if key is None:
            report["errors"].append(f"Missing required {label} array")
    for key in ("r_a", "r_c", "cs_a", "cs_c", "j_a", "j_c"):
        if key not in keys:
            report["errors"].append(f"Missing required array: {key}")

    if report["errors"]:
        report["status"] = "FAIL"
    else:
        t = _load(sol, tkey).astype(np.float64).reshape(-1)  # type: ignore[arg-type]
        I = _load(sol, ikey).astype(np.float64).reshape(-1)  # type: ignore[arg-type]
        report["N_time_points"] = int(t.size)
        report["stats_time"] = _stats(t)
        report["stats_I_profile"] = _stats(I)
        dt = np.diff(t)
        report["time_checks"] = {
            "monotonic_non_decreasing": bool(np.all(dt >= -1e-9)),
            "strictly_increasing_fraction": float(np.mean(dt > 0)) if dt.size else None,
            "dt_min_s": float(np.min(dt)) if dt.size else None,
            "dt_median_s": float(np.median(dt)) if dt.size else None,
            "dt_max_s": float(np.max(dt)) if dt.size else None,
            "t_start_s": float(t[0]) if t.size else None,
            "t_end_s": float(t[-1]) if t.size else None,
        }
        if not report["time_checks"]["monotonic_non_decreasing"]:
            report["errors"].append("time array is not monotonic non-decreasing")

        if ckey is not None:
            cyc_arr = _load(sol, ckey)
            if cyc_arr.size == 1:
                cyc = np.full(t.shape, int(np.asarray(cyc_arr).reshape(-1)[0]), dtype=np.int64)
            else:
                cyc = np.asarray(cyc_arr, dtype=np.int64).reshape(-1)
            if cyc.size != t.size:
                report["errors"].append(f"cycle array length {cyc.size} != time length {t.size}")
            else:
                uniq = np.unique(cyc)
                exp = set(range(args.expected_cycle_from, args.expected_cycle_to + 1))
                got = set(map(int, uniq.tolist()))
                missing = sorted(exp.difference(got))
                extra = sorted(got.difference(exp))
                report["cycle_checks"] = {
                    "cycle_min": int(uniq.min()) if uniq.size else None,
                    "cycle_max": int(uniq.max()) if uniq.size else None,
                    "cycle_count": int(uniq.size),
                    "first_10_cycles": [int(x) for x in uniq[:10]],
                    "last_10_cycles": [int(x) for x in uniq[-10:]],
                    "has_expected_start": bool(uniq.size and int(uniq.min()) == args.expected_cycle_from),
                    "has_expected_end": bool(uniq.size and int(uniq.max()) == args.expected_cycle_to),
                    "missing_cycle_count": len(missing),
                    "missing_cycles_first_20": missing[:20],
                    "extra_cycles": extra,
                }
                if missing:
                    report["warnings"].append(f"missing cycles in expected range: first={missing[:20]}, total={len(missing)}")
                if extra:
                    report["warnings"].append(f"unexpected cycles outside expected range: {extra}")
        else:
            report["warnings"].append("No cycle_id/cycle array; per-cycle evaluation will be limited")

        report["current_checks"] = {
            "has_charge_I_positive": bool(np.nanmax(I) > 0),
            "has_discharge_I_negative": bool(np.nanmin(I) < 0),
            "has_rest_I_zero": bool(np.any(np.isclose(I, 0.0, atol=1e-12))),
            "unique_rounded_first_30_A": [float(x) for x in np.unique(np.round(I, 12))[:30]],
        }
        if not report["current_checks"]["has_charge_I_positive"]:
            report["warnings"].append("I_profile has no positive charging current")
        if not report["current_checks"]["has_discharge_I_negative"]:
            report["warnings"].append("I_profile has no negative discharging current")
        if not report["current_checks"]["has_rest_I_zero"]:
            report["warnings"].append("I_profile has no zero-current rest segment")

        # Shape checks.
        for k in ("r_a", "r_c"):
            if k in hdr:
                shp = hdr[k]["shape"]
                if len(shp) != 1:
                    report["errors"].append(f"{k} should be 1D, got {shp}")
                elif shp[0] != args.expected_nr:
                    report["warnings"].append(f"{k} length {shp[0]} != expected {args.expected_nr}")
        for k in ("cs_a", "cs_c"):
            if k in hdr:
                shp = hdr[k]["shape"]
                if len(shp) != 2 or shp[0] != t.size:
                    report["errors"].append(f"{k} should have shape (N,nr), got {shp}, N={t.size}")
        for k in ("phis_c", "phie", "I_profile", "j_a", "j_c"):
            kk = _find_key(keys, (k,))
            if kk in hdr:
                shp = hdr[kk]["shape"]
                if len(shp) != 1 or shp[0] != t.size:
                    report["errors"].append(f"{kk} should have shape (N,), got {shp}, N={t.size}")

        # Flux sign convention.
        try:
            ja = _load(sol, "j_a").astype(np.float64).reshape(-1)
            jc = _load(sol, "j_c").astype(np.float64).reshape(-1)
            nz = np.abs(I) > max(1e-12, float(np.nanmax(np.abs(I))) * 1e-6)
            if nz.any():
                report["flux_sign_checks"] = {
                    "fraction_j_a_opposite_I": float(np.mean(np.sign(ja[nz]) == -np.sign(I[nz]))),
                    "fraction_j_c_same_I": float(np.mean(np.sign(jc[nz]) == np.sign(I[nz]))),
                    "corr_j_a_I": _corr(ja[nz], I[nz]),
                    "corr_j_c_I": _corr(jc[nz], I[nz]),
                }
                if report["flux_sign_checks"]["fraction_j_a_opposite_I"] < 0.99:
                    report["warnings"].append("j_a sign is not consistently opposite to I_profile")
                if report["flux_sign_checks"]["fraction_j_c_same_I"] < 0.99:
                    report["warnings"].append("j_c sign is not consistently same as I_profile")
        except Exception as e:
            report["errors"].append(f"flux sign check failed: {type(e).__name__}: {e}")

        # Soft-label voltage versus experiment, diagnostic only.
        if vkey and pkey:
            try:
                v = _load(sol, vkey).astype(np.float64).reshape(-1)
                psoft = _load(sol, pkey).astype(np.float64).reshape(-1)
                e = psoft - v
                m = np.isfinite(e)
                report["voltage_fixedB_global_metrics"] = {
                    "mae_V": float(np.mean(np.abs(e[m]))) if m.any() else None,
                    "rmse_V": float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None,
                    "max_abs_V": float(np.max(np.abs(e[m]))) if m.any() else None,
                    "corr": _corr(psoft, v),
                }
            except Exception as e:
                report["warnings"].append(f"voltage metric skipped: {type(e).__name__}: {e}")

        if not args.quick:
            report["mass_closure_checks"] = {}
            for e in ("a", "c"):
                try:
                    mc = _mass_closure(sol, keys, t, e, args.mass_mae_fail, args.mass_mae_warn)
                    report["mass_closure_checks"][e] = mc
                    if mc.get("status") == "FAIL":
                        report["errors"].append(f"mass closure failed for electrode {e}: best MAE={mc.get(mc.get('best_integrator_endpoint', ''), {}).get('mae')}")
                    elif mc.get("status") == "WARN":
                        report["warnings"].append(f"mass closure warning for electrode {e}: best MAE={mc.get(mc.get('best_integrator_endpoint', ''), {}).get('mae')}")
                except Exception as e2:
                    report["errors"].append(f"mass closure check crashed for electrode {e}: {type(e2).__name__}: {e2}")

        for side in ("soft_label_summary.json", "record_profile_summary.json", "metrics_voltage_fixedB_by_cycle.csv"):
            pside = sol.parent / side
            report[f"sidecar_exists_{side}"] = bool(pside.exists())
            if not pside.exists() and side != "metrics_voltage_fixedB_by_cycle.csv":
                report["warnings"].append(f"missing sidecar file: {side}")

        report["status"] = "PASS" if not report["errors"] else "FAIL"

    if args.write_report:
        json_path = sol.parent / "softlabel_integrity_report.json"
        txt_path = sol.parent / "softlabel_integrity_report.txt"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=_jsonable)
            lines = [
                f"Status: {report['status']}",
                f"solution: {sol}",
                f"file_size_gb: {report.get('file_size_gb')}",
                f"array_count: {report.get('array_count')}",
                f"N_time_points: {report.get('N_time_points')}",
                "resolved_keys: " + json.dumps(report.get("resolved_keys"), ensure_ascii=False, default=_jsonable),
            ]
            for name in ("cycle_checks", "time_checks", "current_checks", "flux_sign_checks", "voltage_fixedB_global_metrics", "mass_closure_checks"):
                if name in report:
                    lines.append(name + ": " + json.dumps(report[name], ensure_ascii=False, default=_jsonable))
            if report["warnings"]:
                lines.append("Warnings:")
                lines += ["  - " + w for w in report["warnings"]]
            if report["errors"]:
                lines.append("Errors:")
                lines += ["  - " + e for e in report["errors"]]
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            report["report_json_path"] = str(json_path)
            report["report_txt_path"] = str(txt_path)
        except Exception as e:
            report["warnings"].append(f"failed to write report files: {type(e).__name__}: {e}")

    summary = {
        "status": report.get("status"),
        "solution": str(sol),
        "file_size_gb": report.get("file_size_gb"),
        "array_count": report.get("array_count"),
        "N_time_points": report.get("N_time_points"),
        "cycle_checks": report.get("cycle_checks"),
        "time_checks": report.get("time_checks"),
        "current_checks": report.get("current_checks"),
        "flux_sign_checks": report.get("flux_sign_checks"),
        "voltage_fixedB_global_metrics": report.get("voltage_fixedB_global_metrics"),
        "mass_closure_checks": report.get("mass_closure_checks"),
        "warnings": report.get("warnings"),
        "errors": report.get("errors"),
        "report_json_path": report.get("report_json_path"),
        "report_txt_path": report.get("report_txt_path"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_jsonable))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
