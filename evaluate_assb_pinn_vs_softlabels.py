#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ASSB PINN predictions against v3 soft labels.

Recommended location in project root:
    evaluate_assb_pinn_vs_softlabels.py

Example:
    D:\\Anaconda\\envs\\torchgpu\\python.exe evaluate_assb_pinn_vs_softlabels.py ^
      --model_dir ModelFin_52 ^
      --soft_label_dir Data\\assb_soft_labels_cycles5plus_v3 ^
      --ocp_dir C:\\Users\\Tiga_QJW\\Desktop\\ASSB_Scheme_V1\\ocp_estimation_outputs ^
      --output_dir EvalFin_52_vs_softlabels

This script does NOT train the model. It only loads the saved PINN and compares
its outputs with data_phie.npz / data_phis_c.npz / data_cs_a.npz / data_cs_c.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_paths(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    util_dir = repo_root / "util"
    for p in [repo_root, util_dir]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _find_checkpoint(model_dir: Path) -> str:
    for name in ["best.pt", "last.pt", "lastSGD.pt", "lastLBFGS.pt"]:
        p = model_dir / name
        if p.exists():
            return name
    return "not_found"


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _subsample_indices(n: int, max_points: int, mode: str = "uniform", seed: int = 7) -> np.ndarray:
    n = int(n)
    max_points = int(max_points)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=max_points, replace=False))
    return np.unique(np.linspace(0, n - 1, max_points, dtype=np.int64))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "maxabs": float("nan"),
            "bias_mean": float("nan"),
            "corr": float("nan"),
            "label_min": float("nan"),
            "label_max": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "maxabs": float(np.max(np.abs(err))),
        "bias_mean": float(np.mean(err)),
        "corr": corr,
        "label_min": float(np.min(yt)),
        "label_max": float(np.max(yt)),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
    }


def _write_metrics_text(metrics: Dict, path: Path) -> None:
    lines = []
    lines.append("ASSB PINN vs soft-label evaluation")
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"model_dir: {metrics.get('model_dir')}")
    lines.append(f"checkpoint: {metrics.get('checkpoint')}")
    lines.append(f"soft_label_dir: {metrics.get('soft_label_dir')}")
    lines.append(f"device: {metrics.get('device')}")
    lines.append("")
    for key in ["phie", "phis_c", "cs_a", "cs_c", "theta_a", "theta_c"]:
        if key not in metrics:
            continue
        m = metrics[key]
        lines.append(f"[{key}]")
        for k in ["n", "mae", "rmse", "maxabs", "bias_mean", "corr", "label_min", "label_max", "pred_min", "pred_max"]:
            if k in m:
                lines.append(f"  {k}: {m[k]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _torch_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    x = _as_2d(x)
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def predict_dataset(nn, variable: str, x: np.ndarray, x_params: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict one variable on one soft-label dataset."""
    x = _as_2d(x)
    x_params = _as_2d(x_params)
    n = x.shape[0]
    preds = []

    out_index = {
        "phie": getattr(nn, "ind_phie", 0),
        "phis_c": getattr(nn, "ind_phis_c", 1),
        "cs_a": getattr(nn, "ind_cs_a", 2),
        "cs_c": getattr(nn, "ind_cs_c", 3),
    }[variable]

    ind_deg_i0_a = getattr(nn, "ind_deg_i0_a", 0)
    ind_deg_ds_c = getattr(nn, "ind_deg_ds_c", 1)
    rescale_T = float(nn.params["rescale_T"])
    rescale_R = float(nn.params["rescale_R"])

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = x[start:stop]
        pb = x_params[start:stop]
        t = xb[:, 0:1]
        if xb.shape[1] >= 2:
            r = xb[:, 1:2]
        else:
            r = np.zeros_like(t)

        if pb.shape[1] >= 2:
            deg_i0_a = pb[:, 0:1]
            deg_ds_c = pb[:, 1:2]
        else:
            deg_i0_a = np.ones_like(t)
            deg_ds_c = np.ones_like(t)

        t_t = _torch_tensor(t, nn.device)
        r_t = _torch_tensor(r, nn.device)
        deg_i0_t = _torch_tensor(deg_i0_a, nn.device)
        deg_ds_t = _torch_tensor(deg_ds_c, nn.device)

        with torch.no_grad():
            raw = nn.model(
                [
                    t_t / rescale_T,
                    r_t / rescale_R,
                    nn.rescale_param(deg_i0_t, ind_deg_i0_a),
                    nn.rescale_param(deg_ds_t, ind_deg_ds_c),
                ],
                training=False,
            )[out_index]

            if variable == "phie":
                pred = nn.rescalePhie(raw, t_t, deg_i0_t, deg_ds_t)
            elif variable == "phis_c":
                pred = nn.rescalePhis_c(raw, t_t, deg_i0_t, deg_ds_t)
            elif variable == "cs_a":
                pred = nn.rescaleCs_a(raw, t_t, r_t, deg_i0_t, deg_ds_t)
            elif variable == "cs_c":
                pred = nn.rescaleCs_c(raw, t_t, r_t, deg_i0_t, deg_ds_t)
            else:
                raise ValueError(variable)

        preds.append(pred.detach().cpu().numpy().reshape(-1, 1))
    return np.vstack(preds)


def _try_load_solution(solution_path: Path) -> Dict[str, np.ndarray]:
    if not solution_path.exists():
        return {}
    try:
        data = np.load(solution_path)
        return {k: data[k] for k in data.files}
    except Exception:
        return {}


def _find_voltage_key(sol: Dict[str, np.ndarray]) -> str | None:
    candidates = [
        "V_exp",
        "voltage_exp",
        "voltage_V",
        "V_record",
        "V_meas",
        "voltage_measured",
        "voltage",
    ]
    lower_map = {k.lower(): k for k in sol.keys()}
    for cand in candidates:
        if cand in sol:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def make_plots(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, solution: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib import failed; skip plots. {exc}")
        return

    # Potential time curves
    for var in ["phis_c", "phie"]:
        if var not in results:
            continue
        x = results[var]["x"]
        y = results[var]["label"].reshape(-1)
        p = results[var]["pred"].reshape(-1)
        t = x[:, 0].reshape(-1)
        plt.figure(figsize=(10, 4.8))
        plt.plot(t, y, label=f"soft label {var}")
        plt.plot(t, p, label=f"PINN {var}", alpha=0.85)
        if var == "phis_c":
            vkey = _find_voltage_key(solution)
            if vkey is not None:
                v = np.asarray(solution[vkey]).reshape(-1)
                # If solution time has same length as v, use it; otherwise skip.
                tkey = None
                for cand in ["t", "t_s", "time_s", "time"]:
                    if cand in solution:
                        tkey = cand
                        break
                if tkey is not None:
                    ts = np.asarray(solution[tkey]).reshape(-1)
                    if ts.size == v.size:
                        plt.plot(ts, v, label="experiment voltage", alpha=0.5)
        plt.xlabel("time / s")
        plt.ylabel(f"{var} / V")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"timeseries_{var}.png", dpi=180)
        plt.close()

    # Surface concentrations
    for var, ylabel in [("cs_a", "cs_a"), ("cs_c", "cs_c")]:
        if var not in results:
            continue
        x = results[var]["x"]
        r = x[:, 1].reshape(-1)
        rmax = np.nanmax(r)
        mask = np.isclose(r, rmax, rtol=1e-7, atol=max(1e-12, abs(rmax) * 1e-7))
        if mask.sum() < 2:
            continue
        t = x[mask, 0].reshape(-1)
        y = results[var]["label"].reshape(-1)[mask]
        p = results[var]["pred"].reshape(-1)[mask]
        order = np.argsort(t)
        plt.figure(figsize=(10, 4.8))
        plt.plot(t[order], y[order], label=f"soft label {var} surface")
        plt.plot(t[order], p[order], label=f"PINN {var} surface", alpha=0.85)
        plt.xlabel("time / s")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"surface_timeseries_{var}.png", dpi=180)
        plt.close()

    # Correlation plots
    for var in ["phis_c", "phie", "cs_a", "cs_c"]:
        if var not in results:
            continue
        y = results[var]["label"].reshape(-1)
        p = results[var]["pred"].reshape(-1)
        mask = np.isfinite(y) & np.isfinite(p)
        if mask.sum() < 2:
            continue
        y = y[mask]
        p = p[mask]
        if y.size > 8000:
            idx = _subsample_indices(y.size, 8000, mode="random")
            y_plot = y[idx]
            p_plot = p[idx]
        else:
            y_plot = y
            p_plot = p
        lo = float(min(np.min(y_plot), np.min(p_plot)))
        hi = float(max(np.max(y_plot), np.max(p_plot)))
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(y_plot, p_plot, s=4, alpha=0.35)
        plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
        plt.xlabel(f"soft label {var}")
        plt.ylabel(f"PINN {var}")
        plt.tight_layout()
        plt.savefig(out_dir / f"correlation_{var}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASSB PINN against generated soft labels")
    parser.add_argument("--repo_root", default=None, help="Project root. Default: directory containing this script")
    parser.add_argument("--model_dir", default="ModelFin_52", help="Saved model folder, e.g. ModelFin_52")
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycles5plus_v3", help="Folder with data_*.npz and solution.npz")
    parser.add_argument("--ocp_dir", default=None, help="Optional OCP folder; passed through ASSB_OCP_DIR")
    parser.add_argument("--output_dir", default=None, help="Evaluation output folder")
    parser.add_argument("--max_time_points", type=int, default=3000, help="Max potential points to evaluate")
    parser.add_argument("--max_cs_points", type=int, default=120000, help="Max concentration points to evaluate")
    parser.add_argument("--batch_size", type=int, default=8192, help="Prediction batch size")
    parser.add_argument("--no_plots", action="store_true", help="Only write metrics, no PNG plots")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_script().resolve()
    soft_label_dir = (repo_root / args.soft_label_dir).resolve() if not Path(args.soft_label_dir).is_absolute() else Path(args.soft_label_dir).resolve()
    model_dir = (repo_root / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / f"Eval_{model_dir.name}_vs_{soft_label_dir.name}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _add_repo_paths(repo_root)
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir)
    if args.ocp_dir:
        os.environ["ASSB_OCP_DIR"] = str(Path(args.ocp_dir).resolve())

    from util.load_pinn import load_model

    print(f"INFO: repo_root       = {repo_root}")
    print(f"INFO: model_dir       = {model_dir}")
    print(f"INFO: checkpoint      = {_find_checkpoint(model_dir)}")
    print(f"INFO: soft_label_dir  = {soft_label_dir}")
    print(f"INFO: output_dir      = {out_dir}")

    nn = load_model(utilFolder=str(repo_root / "util"), modelFolder=str(model_dir), localUtilFolder=str(repo_root / "util"))
    nn.model.eval()
    print(f"INFO: loaded model on device = {nn.device}")

    datasets = {
        "phie": _load_npz(soft_label_dir / "data_phie.npz"),
        "phis_c": _load_npz(soft_label_dir / "data_phis_c.npz"),
        "cs_a": _load_npz(soft_label_dir / "data_cs_a.npz"),
        "cs_c": _load_npz(soft_label_dir / "data_cs_c.npz"),
    }

    results: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, object] = {
        "model_dir": str(model_dir),
        "checkpoint": _find_checkpoint(model_dir),
        "soft_label_dir": str(soft_label_dir),
        "output_dir": str(out_dir),
        "device": str(nn.device),
        "params": {
            "csanmax": float(nn.params.get("csanmax", np.nan)),
            "cscamax": float(nn.params.get("cscamax", np.nan)),
            "R_ohm_eff": float(nn.params.get("R_ohm_eff", np.nan)),
            "voltage_alignment_offset_V": float(nn.params.get("voltage_alignment_offset_V", np.nan)),
            "theta_c_bottom": float(nn.params.get("theta_c_bottom", nn.params.get("theta_c_bottom_v3", np.nan))),
            "theta_c_top": float(nn.params.get("theta_c_top", nn.params.get("theta_c_top_v3", np.nan))),
        },
    }

    for var, data in datasets.items():
        x = _as_2d(data["x_train"])
        y = _as_2d(data["y_train"])
        xp = _as_2d(data.get("x_params_train", np.ones((x.shape[0], 2), dtype=np.float64)))
        mode = "uniform" if var in {"phie", "phis_c"} else "random"
        max_points = args.max_time_points if var in {"phie", "phis_c"} else args.max_cs_points
        idx = _subsample_indices(x.shape[0], max_points, mode=mode)
        print(f"INFO: predicting {var}: using {len(idx)} / {x.shape[0]} points")
        pred = predict_dataset(nn, var, x[idx], xp[idx], batch_size=args.batch_size)
        results[var] = {"x": x[idx], "label": y[idx], "pred": pred, "x_params": xp[idx]}
        metrics[var] = _metrics(y[idx], pred)

        if var == "cs_a":
            csmax = float(nn.params.get("csanmax", np.nan))
            if np.isfinite(csmax) and csmax != 0:
                metrics["theta_a"] = _metrics(y[idx] / csmax, pred / csmax)
        if var == "cs_c":
            csmax = float(nn.params.get("cscamax", np.nan))
            if np.isfinite(csmax) and csmax != 0:
                metrics["theta_c"] = _metrics(y[idx] / csmax, pred / csmax)

    solution = _try_load_solution(soft_label_dir / "solution.npz")
    if solution:
        metrics["solution_keys"] = list(solution.keys())

    metrics_json = out_dir / "metrics.json"
    metrics_txt = out_dir / "metrics_summary.txt"
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_metrics_text(metrics, metrics_txt)

    if not args.no_plots:
        make_plots(results, out_dir, solution)

    print("\nEvaluation finished.")
    print(f"Metrics JSON : {metrics_json}")
    print(f"Metrics TXT  : {metrics_txt}")
    print("\nKey metrics:")
    for key in ["phis_c", "phie", "theta_a", "theta_c", "cs_a", "cs_c"]:
        if key in metrics:
            m = metrics[key]
            print(
                f"  {key:8s}  MAE={m['mae']:.6g}  RMSE={m['rmse']:.6g}  "
                f"MAX={m['maxabs']:.6g}  corr={m['corr']:.6g}"
            )


if __name__ == "__main__":
    main()
