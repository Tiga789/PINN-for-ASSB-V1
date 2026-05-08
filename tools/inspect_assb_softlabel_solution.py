# -*- coding: utf-8 -*-
r"""
ASSB all-cycle soft-label checker.

Default target:
  C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_allcycle\solution.npz

What it checks:
  1) solution.npz exists and can be opened.
  2) Key arrays exist with compatible shapes.
  3) cycle_id covers the expected cycle window, default 5-522.
  4) t_global_s/time is monotonic and I_profile has +, -, and 0 segments.
  5) j_a is opposite to I, j_c is same sign as I.
  6) If --deep is enabled, cs_a/cs_c are finite and their spherical mean is
     checked against dcbar/dt = -3*j/R using two common radial quadrature rules.

Reports written next to solution.npz:
  softlabel_allcycle_check_report.json
  softlabel_allcycle_check_report.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import zipfile

import numpy as np
from numpy.lib import format as npfmt

DEFAULT_SOLUTION = r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_allcycle\solution.npz"

ALIASES = {
    "time": ("t_global_s", "t", "time_s", "t_s", "time"),
    "cycle_id": ("cycle_id", "cycles", "cycle"),
    "I_profile": ("I_profile", "I", "current_A", "I_A", "current"),
    "voltage_exp": ("voltage_exp", "V_exp", "voltage", "V", "voltage_meas"),
    "phis_c": ("phis_c", "phi_s_c", "phis", "voltage_soft", "V_soft"),
    "phie": ("phie", "phi_e", "data_phie"),
    "j_a": ("j_a", "J_a", "flux_a"),
    "j_c": ("j_c", "J_c", "flux_c"),
    "r_a": ("r_a", "ra", "r_an", "r_negative", "r_grid_a", "r_anode"),
    "r_c": ("r_c", "rc", "r_ca", "r_positive", "r_grid_c", "r_cathode"),
    "cs_a": ("cs_a", "c_s_a", "csan", "csa", "cs_a_full", "anode_cs", "c_s_anode"),
    "cs_c": ("cs_c", "c_s_c", "csca", "csc", "cs_c_full", "cathode_cs", "c_s_cathode"),
    "theta_a": ("theta_a", "theta_s_a"),
    "theta_c": ("theta_c", "theta_s_c"),
    "Uocp_a": ("Uocp_a", "U_a", "ocp_a"),
    "Uocp_c": ("Uocp_c", "U_c", "ocp_c"),
    "eta_a": ("eta_a",),
    "eta_c": ("eta_c",),
    "step_id": ("step_id",),
    "step_type": ("step_type",),
}

REQUIRED = ["time", "cycle_id", "I_profile", "phis_c", "phie", "r_a", "r_c", "cs_a", "cs_c"]
RECOMMENDED = ["j_a", "j_c", "voltage_exp", "Uocp_a", "Uocp_c", "eta_a", "eta_c", "step_id", "step_type"]


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def read_npy_header_from_npz(npz_path: Path) -> Dict[str, Dict[str, Any]]:
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
                elif version in {(2, 0), (3, 0)}:
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


def find_key(keys: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    key_list = list(keys)
    lower = {k.lower(): k for k in key_list}
    for a in aliases:
        if a in key_list:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def load(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        return z[key]


def stats_numeric(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out: Dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        out.update({
            "finite_all": bool(finite.all()),
            "finite_count": int(finite.sum()),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
        })
        if finite.any():
            v = arr[finite]
            out.update({"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())})
    else:
        flat = arr.reshape(-1)
        out["sample_first_10"] = [str(x) for x in flat[:10]]
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n < 3:
        return None
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def as_nt_nr(arr: np.ndarray, nt: int, nr: int, label: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (nt, nr):
        return arr.astype(np.float64, copy=False)
    if arr.shape == (nr, nt):
        return arr.T.astype(np.float64, copy=False)
    if arr.ndim == 1 and arr.size == nt * nr:
        return arr.reshape(nt, nr).astype(np.float64, copy=False)
    raise ValueError(f"{label} shape {arr.shape} is not compatible with (Nt,Nr)=({nt},{nr})")


def weights_spherical_trapz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    coeff = np.zeros_like(r)
    dr = np.diff(r)
    coeff[:-1] += 0.5 * dr
    coeff[1:] += 0.5 * dr
    w = coeff * r**2
    return w / np.sum(w)


def weights_fv_shell(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    R = float(r[-1])
    edges = np.empty(r.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = R
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    w = (edges[1:] ** 3 - edges[:-1] ** 3) / (R ** 3)
    return w / np.sum(w)


def closure_metrics(t: np.ndarray, r: np.ndarray, cs: np.ndarray, j: np.ndarray, label: str) -> Dict[str, Any]:
    nt = len(t)
    nr = len(r)
    cs2 = as_nt_nr(cs, nt, nr, label)
    dt = np.diff(t.astype(np.float64))
    R = float(r[-1])
    # Boundary convention D dc/dr = -j gives d<c>/dt = -3*j/R.
    inc_right = (-3.0 * j[1:].astype(np.float64) / R) * dt
    inc_left = (-3.0 * j[:-1].astype(np.float64) / R) * dt
    inc_mid = 0.5 * (inc_left + inc_right)
    out: Dict[str, Any] = {"R": R, "nt": int(nt), "nr": int(nr)}
    best_mae = float("inf")
    best_name = None
    for wname, w in [("finite_volume_shell", weights_fv_shell(r)), ("spherical_trapz", weights_spherical_trapz(r))]:
        cbar = cs2 @ w
        for iname, inc in [("right_endpoint", inc_right), ("left_endpoint", inc_left), ("midpoint", inc_mid)]:
            pred = np.empty_like(cbar)
            pred[0] = cbar[0]
            pred[1:] = cbar[0] + np.cumsum(inc)
            err = pred - cbar
            mask = np.isfinite(err)
            mae = float(np.mean(np.abs(err[mask]))) if mask.any() else float("nan")
            rmse = float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("nan")
            maxabs = float(np.max(np.abs(err[mask]))) if mask.any() else float("nan")
            key = f"{wname}__{iname}"
            out[key] = {
                "mae": mae,
                "rmse": rmse,
                "max_abs": maxabs,
                "bias": float(np.mean(err[mask])) if mask.any() else float("nan"),
                "corr": safe_corr(pred, cbar),
            }
            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best_name = key
    out["best_method"] = best_name
    out["best_mae"] = best_mae
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ASSB all-cycle continuous soft labels.")
    ap.add_argument("--solution", default=DEFAULT_SOLUTION, help="Path to solution.npz")
    ap.add_argument("--expected_cycle_from", type=int, default=5)
    ap.add_argument("--expected_cycle_to", type=int, default=522)
    ap.add_argument("--expected_nr", type=int, default=64)
    ap.add_argument("--deep", action="store_true", help="Load cs_a/cs_c and run finite/mass-closure checks.")
    ap.add_argument("--mass_mae_fail", type=float, default=5e-2, help="Fail if best cbar mass-closure MAE exceeds this value.")
    args = ap.parse_args()

    solution = Path(args.solution)
    out_dir = solution.parent
    report: Dict[str, Any] = {
        "script": "check_assb_allcycle_softlabel.py",
        "solution": str(solution),
        "status": "UNKNOWN",
        "warnings": [],
        "errors": [],
    }

    if not solution.exists():
        report["errors"].append(f"solution.npz not found: {solution}")
        report["status"] = "FAIL"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    report["file_size_gb"] = round(solution.stat().st_size / (1024 ** 3), 4)
    try:
        headers = read_npy_header_from_npz(solution)
    except Exception as e:
        report["errors"].append(f"Cannot read npz headers: {type(e).__name__}: {e}")
        report["status"] = "FAIL"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    keys = set(headers.keys())
    report["array_count"] = len(keys)
    report["array_keys"] = sorted(keys)
    resolved: Dict[str, Optional[str]] = {name: find_key(keys, aliases) for name, aliases in ALIASES.items()}
    report["resolved_keys"] = resolved

    missing_req = [k for k in REQUIRED if resolved.get(k) is None]
    missing_rec = [k for k in RECOMMENDED if resolved.get(k) is None]
    if missing_req:
        report["errors"].append("Missing required logical arrays: " + ", ".join(missing_req))
    if missing_rec:
        report["warnings"].append("Missing recommended logical arrays: " + ", ".join(missing_rec))

    nt: Optional[int] = None
    time_key = resolved.get("time")
    if time_key:
        shp = headers[time_key]["shape"]
        if len(shp) == 1:
            nt = int(shp[0])
            report["N_time_points"] = nt
        else:
            report["errors"].append(f"{time_key} should be 1D; got {shp}")
    if nt is not None:
        for logical in ["cycle_id", "I_profile", "phis_c", "phie", "voltage_exp", "j_a", "j_c", "Uocp_a", "Uocp_c", "eta_a", "eta_c", "step_id", "step_type"]:
            k = resolved.get(logical)
            if k:
                shp = headers[k]["shape"]
                if len(shp) != 1 or int(shp[0]) != nt:
                    report["errors"].append(f"{logical}/{k} should have shape ({nt},); got {shp}")
        for logical_cs, logical_r in [("cs_a", "r_a"), ("cs_c", "r_c")]:
            kcs = resolved.get(logical_cs)
            kr = resolved.get(logical_r)
            if kcs and kr:
                sshape = headers[kcs]["shape"]
                rshape = headers[kr]["shape"]
                if len(rshape) != 1:
                    report["errors"].append(f"{logical_r}/{kr} should be 1D; got {rshape}")
                else:
                    nr = int(rshape[0])
                    if nr != args.expected_nr:
                        report["warnings"].append(f"{logical_r}/{kr} length is {nr}; expected {args.expected_nr}")
                    if len(sshape) != 2 or not ((int(sshape[0]) == nt and int(sshape[1]) == nr) or (int(sshape[0]) == nr and int(sshape[1]) == nt)):
                        report["errors"].append(f"{logical_cs}/{kcs} shape {sshape} is not compatible with Nt={nt}, Nr={nr}")

    try:
        arrays: Dict[str, np.ndarray] = {}
        for logical in ["time", "cycle_id", "I_profile", "phis_c", "phie", "voltage_exp", "j_a", "j_c", "r_a", "r_c"]:
            k = resolved.get(logical)
            if k:
                arrays[logical] = load(solution, k)
                report[f"stats_{logical}"] = stats_numeric(arrays[logical])

        if "time" in arrays:
            t = np.asarray(arrays["time"], dtype=np.float64).reshape(-1)
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
        if "cycle_id" in arrays:
            cyc = np.asarray(arrays["cycle_id"]).astype(int).reshape(-1)
            uniq = np.unique(cyc)
            expected = set(range(args.expected_cycle_from, args.expected_cycle_to + 1))
            got = set(map(int, uniq))
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            report["cycle_checks"] = {
                "cycle_min": int(uniq.min()) if uniq.size else None,
                "cycle_max": int(uniq.max()) if uniq.size else None,
                "cycle_count": int(uniq.size),
                "first_10_cycles": [int(x) for x in uniq[:10]],
                "last_10_cycles": [int(x) for x in uniq[-10:]],
                "missing_cycle_count": len(missing),
                "missing_cycles_first_20": missing[:20],
                "extra_cycles": extra,
            }
            if missing:
                report["warnings"].append(f"Missing expected cycles: first={missing[:20]}, total={len(missing)}")
            if extra:
                report["warnings"].append(f"Unexpected cycle IDs outside expected range: {extra}")
        if "I_profile" in arrays:
            I = np.asarray(arrays["I_profile"], dtype=np.float64).reshape(-1)
            report["current_checks"] = {
                "has_charge_I_positive": bool(np.nanmax(I) > 0),
                "has_discharge_I_negative": bool(np.nanmin(I) < 0),
                "has_rest_I_zero": bool(np.any(np.isclose(I, 0.0, atol=1e-12))),
                "unique_rounded_first_30_A": [float(x) for x in np.unique(np.round(I, 12))[:30]],
            }
            if not report["current_checks"]["has_charge_I_positive"]:
                report["warnings"].append("I_profile has no positive charging segment")
            if not report["current_checks"]["has_discharge_I_negative"]:
                report["warnings"].append("I_profile has no negative discharging segment")
            if not report["current_checks"]["has_rest_I_zero"]:
                report["warnings"].append("I_profile has no zero-current rest segment")
        if all(x in arrays for x in ["I_profile", "j_a", "j_c"]):
            I = np.asarray(arrays["I_profile"], dtype=np.float64).reshape(-1)
            ja = np.asarray(arrays["j_a"], dtype=np.float64).reshape(-1)
            jc = np.asarray(arrays["j_c"], dtype=np.float64).reshape(-1)
            n = min(I.size, ja.size, jc.size)
            I, ja, jc = I[:n], ja[:n], jc[:n]
            nz = np.abs(I) > max(1e-12, float(np.nanmax(np.abs(I))) * 1e-6)
            if np.any(nz):
                report["flux_sign_checks"] = {
                    "fraction_j_a_opposite_I": float(np.mean(np.sign(ja[nz]) == -np.sign(I[nz]))),
                    "fraction_j_c_same_I": float(np.mean(np.sign(jc[nz]) == np.sign(I[nz]))),
                    "corr_j_a_I": safe_corr(ja[nz], I[nz]),
                    "corr_j_c_I": safe_corr(jc[nz], I[nz]),
                }
                if report["flux_sign_checks"]["fraction_j_a_opposite_I"] < 0.99:
                    report["errors"].append("j_a sign is not consistently opposite to I_profile")
                if report["flux_sign_checks"]["fraction_j_c_same_I"] < 0.99:
                    report["errors"].append("j_c sign is not consistently same as I_profile")
        if all(x in arrays for x in ["voltage_exp", "phis_c"]):
            v = np.asarray(arrays["voltage_exp"], dtype=np.float64).reshape(-1)
            p = np.asarray(arrays["phis_c"], dtype=np.float64).reshape(-1)
            n = min(v.size, p.size)
            diff = p[:n] - v[:n]
            mask = np.isfinite(diff)
            report["voltage_soft_vs_exp_metrics"] = {
                "mae_V": float(np.mean(np.abs(diff[mask]))) if mask.any() else None,
                "rmse_V": float(np.sqrt(np.mean(diff[mask] ** 2))) if mask.any() else None,
                "max_abs_V": float(np.max(np.abs(diff[mask]))) if mask.any() else None,
                "corr": safe_corr(p[:n], v[:n]),
                "note": "This is diagnostic only; training/eval reference remains soft-label phis_c.",
            }
    except Exception as e:
        report["errors"].append(f"1D/radial checks failed: {type(e).__name__}: {e}")

    if args.deep and not report["errors"]:
        try:
            t = np.asarray(load(solution, resolved["time"]), dtype=np.float64).reshape(-1)  # type: ignore[index]
            report["deep_checks"] = {}
            for side in ["a", "c"]:
                cs_key = resolved[f"cs_{side}"]
                r_key = resolved[f"r_{side}"]
                j_key = resolved[f"j_{side}"]
                if not cs_key or not r_key or not j_key:
                    report["warnings"].append(f"Skip mass closure for side {side}: missing cs/r/j")
                    continue
                r = np.asarray(load(solution, r_key), dtype=np.float64).reshape(-1)
                j = np.asarray(load(solution, j_key), dtype=np.float64).reshape(-1)
                cs = load(solution, cs_key)
                finite = np.isfinite(cs)
                side_report: Dict[str, Any] = {
                    "cs_key": cs_key,
                    "r_key": r_key,
                    "j_key": j_key,
                    "cs_shape": list(cs.shape),
                    "cs_dtype": str(cs.dtype),
                    "finite_all": bool(finite.all()),
                    "nan_count": int(np.isnan(cs).sum()),
                    "inf_count": int(np.isinf(cs).sum()),
                    "min": float(np.nanmin(cs)),
                    "max": float(np.nanmax(cs)),
                }
                if not finite.all():
                    report["errors"].append(f"{cs_key} contains NaN or Inf")
                side_report["mass_closure"] = closure_metrics(t, r, cs, j, f"cs_{side}")
                best_mae = side_report["mass_closure"].get("best_mae")
                if best_mae is not None and np.isfinite(best_mae) and float(best_mae) > args.mass_mae_fail:
                    report["errors"].append(
                        f"cs_{side} cbar mass-closure best MAE={best_mae:.6g} > threshold {args.mass_mae_fail:g}"
                    )
                report["deep_checks"][f"cs_{side}"] = side_report
                del cs
        except Exception as e:
            report["errors"].append(f"Deep mass-closure checks failed: {type(e).__name__}: {e}")

    # Sidecar report files are useful but not mandatory for a PASS.
    for sidecar in ["soft_label_summary.json", "record_profile_summary.json", "metrics_voltage_fixedB_by_cycle.csv"]:
        p = out_dir / sidecar
        report[f"sidecar_exists_{sidecar}"] = bool(p.exists())
        if not p.exists():
            report["warnings"].append(f"Missing sidecar file: {sidecar}")
    metrics_csv = out_dir / "metrics_voltage_fixedB_by_cycle.csv"
    if metrics_csv.exists():
        try:
            with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            report["metrics_by_cycle_csv"] = {
                "row_count": len(rows),
                "columns": reader.fieldnames,
                "first_row": rows[0] if rows else None,
                "last_row": rows[-1] if rows else None,
            }
        except Exception as e:
            report["warnings"].append(f"Could not parse metrics_voltageFixedB_by_cycle.csv: {type(e).__name__}: {e}")

    report["status"] = "PASS" if not report["errors"] else "FAIL"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "softlabel_allcycle_check_report.json"
    txt_path = out_dir / "softlabel_allcycle_check_report.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=to_jsonable), encoding="utf-8")

    lines = [
        f"Status: {report['status']}",
        f"solution: {solution}",
        f"file_size_gb: {report.get('file_size_gb')}",
        f"array_count: {report.get('array_count')}",
        f"N_time_points: {report.get('N_time_points')}",
        "resolved_keys: " + json.dumps(report.get("resolved_keys"), ensure_ascii=False),
    ]
    for k in ["cycle_checks", "time_checks", "current_checks", "flux_sign_checks", "voltage_soft_vs_exp_metrics"]:
        if k in report:
            lines.append(f"{k}: " + json.dumps(report[k], ensure_ascii=False, default=to_jsonable))
    if "deep_checks" in report:
        for side in ["cs_a", "cs_c"]:
            if side in report["deep_checks"]:
                mc = report["deep_checks"][side].get("mass_closure", {})
                lines.append(f"{side}_mass_closure_best: method={mc.get('best_method')} best_mae={mc.get('best_mae')}")
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend("  - " + w for w in report["warnings"])
    if report["errors"]:
        lines.append("Errors:")
        lines.extend("  - " + e for e in report["errors"])
    lines.append(f"report_json_path: {json_path}")
    lines.append(f"report_txt_path: {txt_path}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "solution": str(solution),
        "file_size_gb": report.get("file_size_gb"),
        "array_count": report.get("array_count"),
        "N_time_points": report.get("N_time_points"),
        "cycle_checks": report.get("cycle_checks"),
        "time_checks": report.get("time_checks"),
        "current_checks": report.get("current_checks"),
        "flux_sign_checks": report.get("flux_sign_checks"),
        "voltage_soft_vs_exp_metrics": report.get("voltage_soft_vs_exp_metrics"),
        "mass_closure_best": {
            side: report.get("deep_checks", {}).get(side, {}).get("mass_closure", {}).get("best_mae")
            for side in ["cs_a", "cs_c"]
        } if "deep_checks" in report else "not_run_use_--deep",
        "warnings": report["warnings"],
        "errors": report["errors"],
        "report_json_path": str(json_path),
        "report_txt_path": str(txt_path),
    }, ensure_ascii=False, indent=2, default=to_jsonable))

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
