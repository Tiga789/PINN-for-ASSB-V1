#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GV1 D9.7 battery-8 outlier/regime diagnosis.

Diagnostic-only script. It reads existing prediction.npz files and an optional
24x40ks scorecard, then writes segment metrics, time-bin metrics, component
health, and plots. It does not train or modify any GV1 model files.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe(r.get(k)) for k in fields})


def pick(d: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in d.files:
            a = np.asarray(d[k])
            if a.size:
                return a.reshape(-1).astype(float)
    return None


def load_solution(path: Path) -> Dict[str, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=False) as d:
        return {
            "t": pick(d, ["t_global_s", "time_s", "t_s", "time", "t"]),
            "v": pick(d, ["voltage_exp", "voltage_V", "voltage", "V"]),
            "i": pick(d, ["I_profile", "current_A", "current", "I"]),
            "T": pick(d, ["temperature_C", "T_C", "temperature"]),
        }


def interp_from_solution(sol: Dict[str, Optional[np.ndarray]], t: np.ndarray, key: str) -> np.ndarray:
    x = sol.get(key)
    st = sol.get("t")
    if x is not None and st is not None and len(st) > 1:
        return np.interp(t, st, x)
    if x is not None and len(x) == len(t):
        return x
    return np.full_like(t, np.nan, dtype=float)


def load_prediction(path: Path, sol: Dict[str, Optional[np.ndarray]]) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as d:
        t = pick(d, ["t_global_s", "time_s", "t_s", "time", "t"])
        yp = pick(d, ["voltage_exp_pred", "voltage_pred", "V_pred", "phis_c_pred"])
        y = pick(d, ["voltage_exp", "voltage_true", "voltage_V", "V_true"])
        i = pick(d, ["I_profile", "current_A", "current", "I"])
        T = pick(d, ["temperature_C", "T_C", "temperature"])
        comps = {}
        for k in d.files:
            if k.startswith("voltage_") and k not in {"voltage_exp", "voltage_true", "voltage_pred", "voltage_exp_pred"}:
                try:
                    comps[k] = np.asarray(d[k]).reshape(-1).astype(float)
                except Exception:
                    pass
    if yp is None:
        raise ValueError(f"No predicted voltage array in {path}")
    if t is None:
        st = sol.get("t")
        t = st[:len(yp)] if st is not None and len(st) >= len(yp) else np.arange(len(yp), dtype=float)
    if y is None:
        y = interp_from_solution(sol, t, "v")
    if i is None:
        i = interp_from_solution(sol, t, "i")
    if T is None:
        T = interp_from_solution(sol, t, "T")
    n = min(len(t), len(y), len(yp), len(i), len(T))
    out = {"t": t[:n], "y": y[:n], "yp": yp[:n], "i": i[:n], "T": T[:n], "components": {}}
    for k, arr in comps.items():
        if len(arr) >= n:
            out["components"][k] = arr[:n]
    return out


def met(y: np.ndarray, yp: np.ndarray, m: np.ndarray) -> Dict[str, Any]:
    m = m & np.isfinite(y) & np.isfinite(yp)
    if not int(m.sum()):
        return {"n": 0, "mae_V": None, "rmse_V": None, "bias_V": None, "corr": None}
    e = yp[m] - y[m]
    corr = None
    if m.sum() > 2 and np.std(y[m]) > 1e-12 and np.std(yp[m]) > 1e-12:
        corr = float(np.corrcoef(y[m], yp[m])[0, 1])
    return {"n": int(m.sum()), "mae_V": float(np.mean(np.abs(e))), "rmse_V": float(np.sqrt(np.mean(e*e))), "bias_V": float(np.mean(e)), "corr": corr}


def diagnose_one(label: str, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    t, y, yp, i, T = data["t"], data["y"], data["yp"], data["i"], data["T"]
    f = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)
    t, y, yp, i, T = t[f], y[f], yp[f], i[f], T[f]
    e = yp - y
    absI = np.abs(i)
    hi_thr = float(np.nanquantile(absI, 0.90)) if np.isfinite(absI).any() else np.nan
    Tmed = float(np.nanmedian(T)) if np.isfinite(T).any() else np.nan
    Tdev = np.abs(T - Tmed) if np.isfinite(Tmed) else np.full_like(T, np.nan)
    Tthr = float(np.nanquantile(Tdev, 0.90)) if np.isfinite(Tdev).any() else np.nan
    regimes = {
        "all": np.ones_like(y, dtype=bool),
        "low_target_le_2p75": y <= 2.75,
        "mid_target_2p75_to_4p10": (y > 2.75) & (y < 4.10),
        "high_target_ge_4p10": y >= 4.10,
        "high_current_absI_ge_q90": absI >= hi_thr if np.isfinite(hi_thr) else np.zeros_like(y, dtype=bool),
        "temperature_event_absdev_ge_q90": Tdev >= Tthr if np.isfinite(Tthr) else np.zeros_like(y, dtype=bool),
        "charge_I_pos": i > 1e-9,
        "discharge_I_neg": i < -1e-9,
        "pred_upper_ge_4p269": yp >= 4.269,
        "pred_overshoot_gt_4p35": yp > 4.35,
        "pred_low_le_2p75": yp <= 2.75,
    }
    reg = {k: met(y, yp, v) for k, v in regimes.items()}
    reg["high_current_absI_ge_q90"]["threshold_A"] = hi_thr if np.isfinite(hi_thr) else None
    reg["temperature_event_absdev_ge_q90"]["threshold_C_absdev"] = Tthr if np.isfinite(Tthr) else None
    s = {
        "label": label,
        **met(y, yp, np.ones_like(y, dtype=bool)),
        "t_start_s": float(np.min(t)),
        "t_end_s": float(np.max(t)),
        "voltage_exp_minmax": [float(np.min(y)), float(np.max(y))],
        "voltage_pred_minmax": [float(np.min(yp)), float(np.max(yp))],
        "voltage_pred_width_V": float(np.max(yp) - np.min(yp)),
        "pred_upper_frac_ge_4p269": float(np.mean(yp >= 4.269)),
        "pred_upper_frac_ge_4p25": float(np.mean(yp >= 4.25)),
        "pred_overshoot_frac_gt_4p35": float(np.mean(yp > 4.35)),
        "pred_low_voltage_frac_le_2p75": float(np.mean(yp <= 2.75)),
        "target_low_voltage_frac_le_2p75": float(np.mean(y <= 2.75)),
        "target_high_voltage_frac_ge_4p10": float(np.mean(y >= 4.10)),
        "regime_metrics": reg,
    }
    bins = np.linspace(float(np.min(t)), float(np.max(t)), 21)
    bin_rows = []
    for b in range(20):
        bm = (t >= bins[b]) & (t <= bins[b+1] if b == 19 else t < bins[b+1])
        row = {"label": label, "bin_index": b, "t0_s": float(bins[b]), "t1_s": float(bins[b+1]), **met(y, yp, bm)}
        row["pred_upper_frac_ge_4p269"] = float(np.mean((yp >= 4.269)[bm])) if bm.sum() else None
        row["target_high_frac_ge_4p10"] = float(np.mean((y >= 4.10)[bm])) if bm.sum() else None
        bin_rows.append(row)
    comp_rows = []
    for k, arr in data["components"].items():
        arr = arr[f]
        if np.isfinite(arr).any():
            comp_rows.append({"label": label, "component": k, "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)), "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "width": float(np.nanmax(arr)-np.nanmin(arr))})
    return s, bin_rows, comp_rows


def plot_one(label: str, data: Dict[str, Any], out: Path, max_points: int) -> None:
    if plt is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    t, y, yp = data["t"], data["y"], data["yp"]
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)
    t, y, yp = t[m], y[m], yp[m]
    idx = np.linspace(0, len(t)-1, min(len(t), max_points)).astype(int)
    th = t[idx] / 3600.0
    fig = plt.figure(figsize=(11,5)); plt.plot(th, y[idx], label="voltage_exp"); plt.plot(th, yp[idx], label="voltage_pred"); plt.xlabel("time / h"); plt.ylabel("V"); plt.title(label); plt.legend(); plt.tight_layout(); fig.savefig(out/f"{label}_voltage.png", dpi=160); plt.close(fig)
    fig = plt.figure(figsize=(11,5)); plt.plot(th, (yp-y)[idx]); plt.axhline(0.0, linewidth=0.8); plt.xlabel("time / h"); plt.ylabel("pred-exp / V"); plt.title(label+" error"); plt.tight_layout(); fig.savefig(out/f"{label}_error.png", dpi=160); plt.close(fig)
    fig = plt.figure(figsize=(5.5,5.5)); plt.scatter(y[idx], yp[idx], s=4, alpha=0.35); lo=min(np.min(y),np.min(yp)); hi=max(np.max(y),np.max(yp)); plt.plot([lo,hi],[lo,hi], linewidth=0.8); plt.xlabel("exp / V"); plt.ylabel("pred / V"); plt.title(label+" parity"); plt.tight_layout(); fig.savefig(out/f"{label}_parity.png", dpi=160); plt.close(fig)


def inspect_scorecard(path: Optional[Path], out: Path) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"ok": False, "reason": "scorecard_json missing"}
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in obj.get("per_run", []):
        rows.append({k: r.get(k) for k in ["run", "protocol", "status", "mae_V", "rmse_V", "corr", "bias_V", "pred_upper_frac_ge_4p269", "pred_low_voltage_frac_le_2p75", "target_low_voltage_frac_le_2p75", "prediction_npz"]})
    write_csv(out/"scorecard_all_runs_d97.csv", rows)
    b1 = [r for r in rows if r.get("protocol") == "2C"]
    b1 = sorted(b1, key=lambda r: (r.get("corr") if isinstance(r.get("corr"), (int,float)) else 999, -(r.get("mae_V") or -999)))
    write_csv(out/"scorecard_B1_2C_worst_first_d97.csv", b1)
    return {"ok": True, "scorecard_json": str(path), "overall_status": obj.get("overall_status"), "status_counts": obj.get("status_counts"), "worst_2C_top3": b1[:3]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scorecard_json")
    ap.add_argument("--prediction_roots", nargs="*", default=[])
    ap.add_argument("--prediction_npz", nargs="*", default=[])
    ap.add_argument("--max_points_plot", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    sol = load_solution(Path(a.solution_npz))
    preds: List[Tuple[str, Path]] = []
    for r in a.prediction_roots:
        p = Path(r)/"prediction.npz"
        if p.exists(): preds.append((Path(r).name.replace("xjtu_batch134_train_conditioned_pinn_", ""), p))
    for p0 in a.prediction_npz:
        p = Path(p0)
        if p.exists(): preds.append((p.parent.name, p))
    seen = set(); uniq=[]
    for lab,p in preds:
        key=str(p.resolve())
        if key not in seen: uniq.append((lab,p)); seen.add(key)
    summaries=[]; bins=[]; comps=[]; missing=[]
    for r in a.prediction_roots:
        if not (Path(r)/"prediction.npz").exists(): missing.append(r)
    for lab,p in uniq:
        try:
            d=load_prediction(p, sol); s,b,c=diagnose_one(lab,d); s["prediction_npz"]=str(p); summaries.append(s); bins += b; comps += c; plot_one(lab,d,out/"plots",a.max_points_plot)
        except Exception as e:
            summaries.append({"label": lab, "prediction_npz": str(p), "ok": False, "error": str(e)})
    score = inspect_scorecard(Path(a.scorecard_json) if a.scorecard_json else None, out)
    interpretation=[]
    for s in summaries:
        width=s.get("voltage_pred_width_V"); upper=s.get("pred_upper_frac_ge_4p269"); over=s.get("pred_overshoot_frac_gt_4p35")
        if isinstance(width,(int,float)) and width < 0.5: interpretation.append(f"{s['label']}: voltage range collapse; reject clamp/guard variant.")
        elif isinstance(over,(int,float)) and over > 0.002: interpretation.append(f"{s['label']}: high-voltage overshoot dominates; inspect high-current and time-bin rows.")
        elif isinstance(upper,(int,float)) and upper > 0.02: interpretation.append(f"{s['label']}: upper-tail saturation dominates.")
        else: interpretation.append(f"{s.get('label')}: no severe global structural failure detected; inspect plots.")
    result={"ok": True, "stage": "GV1 D9.7 battery-8 outlier/regime diagnosis", "solution_npz": a.solution_npz, "output_dir": str(out), "n_predictions_found": len(uniq), "missing_prediction_roots": missing, "prediction_summaries": summaries, "scorecard_summary": score, "interpretation": interpretation, "recommendation": "Diagnostic-only: do not run 24-profile 200ks until this summary and plots are reviewed."}
    write_json(out/"d97_battery8_diagnosis_summary.json", result)
    write_csv(out/"prediction_summary_d97.csv", summaries)
    write_csv(out/"time_bins_all_predictions_d97.csv", bins)
    write_csv(out/"component_health_d97.csv", comps)
    print(json.dumps(_safe(result), indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
