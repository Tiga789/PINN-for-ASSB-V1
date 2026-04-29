#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced ASSB cycle5 PINN-vs-soft-label evaluator.

Revision 2026-04-29: fixes per-electrode radial normalization during evaluation.

Recommended location: project root
Example:
  D:\\Anaconda\\envs\\torchgpu\\python.exe evaluate_assb_cycle5_pinn_vs_softlabels_debug.py ^
    --model_dir ModelFin_61 ^
    --soft_label_dir Data\\assb_soft_labels_cycle5_v3 ^
    --ocp_dir C:\\Users\\Tiga_QJW\\Desktop\\ASSB_Scheme_V1\\ocp_estimation_outputs ^
    --output_dir EvalFin_61_cycle5_debug ^
    --debug_print_first_batch

Main differences from the earlier evaluator:
  1. It can force the model parameter builder to use soft_label_dir/soft_label_summary.json.
     This avoids stale train_summary_json paths when changing from cycles5plus to cycle5.
  2. It reports exact checkpoint used, config prior_model, model/data summary paths,
     rescale_T/rescale_R, output indices, data shapes, first rows, and prediction ranges.
  3. It supports exact --checkpoint selection and has robust PyTorch/Keras-style model calls.
  4. It writes metrics, diagnostics, debug JSON, and plots.
  5. It uses per-electrode radial normalization for cs_a and cs_c evaluation.

This script does not train the model and does not enable data loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


REQUIRED_NPZ = {
    "phie": "data_phie.npz",
    "phis_c": "data_phis_c.npz",
    "cs_a": "data_cs_a.npz",
    "cs_c": "data_cs_c.npz",
}


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
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {"n": int(x.size), "min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x))}


def _find_checkpoint(model_dir: Path, checkpoint: Optional[str]) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = model_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Requested checkpoint not found: {p}")
        return p.resolve()
    for name in ["best.pt", "last.pt", "lastLBFGS.pt", "lastSGD.pt", "best.weights.h5"]:
        p = model_dir / name
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def _find_key(data: Dict[str, np.ndarray], candidates: Iterable[str]) -> Optional[str]:
    lower = {k.lower(): k for k in data.keys()}
    for cand in candidates:
        if cand in data:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _subsample_indices(n: int, max_points: int, mode: str, seed: int) -> np.ndarray:
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
            "n": 0, "mae": float("nan"), "rmse": float("nan"), "maxabs": float("nan"),
            "bias_mean": float("nan"), "corr": float("nan"),
            "label_min": float("nan"), "label_max": float("nan"), "label_std": float("nan"),
            "pred_min": float("nan"), "pred_max": float("nan"), "pred_std": float("nan"),
            "std_ratio_pred_over_label": float("nan"),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
    label_std = float(np.std(yt))
    pred_std = float(np.std(yp))
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "maxabs": float(np.max(np.abs(err))),
        "bias_mean": float(np.mean(err)),
        "corr": corr,
        "label_min": float(np.min(yt)),
        "label_max": float(np.max(yt)),
        "label_std": label_std,
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
        "pred_std": pred_std,
        "std_ratio_pred_over_label": float(pred_std / label_std) if label_std > 0 else float("nan"),
    }


def _torch_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_as_2d(x), dtype=torch.float64, device=device)


def _call_model(model: Any, inputs: list[torch.Tensor]) -> Any:
    try:
        return model(inputs, training=False)
    except TypeError:
        return model(inputs)


def _get_raw_output(model_output: Any, index: int) -> torch.Tensor:
    if isinstance(model_output, (list, tuple)):
        return model_output[index]
    if torch.is_tensor(model_output):
        if model_output.ndim == 2 and model_output.shape[1] > index:
            return model_output[:, index:index + 1]
        if index == 0:
            return model_output
    raise RuntimeError(f"Could not extract output index {index} from model output type {type(model_output)}")


def _var_output_index(nn: Any, variable: str) -> int:
    mapping = {
        "phie": getattr(nn, "ind_phie", 0),
        "phis_c": getattr(nn, "ind_phis_c", 1),
        "cs_a": getattr(nn, "ind_cs_a", 2),
        "cs_c": getattr(nn, "ind_cs_c", 3),
    }
    return int(mapping[variable])


def _safe_get_param(nn: Any, key: str, default: float = float("nan")) -> float:
    try:
        val = nn.params.get(key, default)
        if hasattr(val, "item"):
            val = val.item()
        return float(val)
    except Exception:
        return float(default)


def load_model_for_eval(
    repo_root: Path,
    model_dir: Path,
    soft_label_dir: Path,
    checkpoint: Path,
    force_summary_from_soft_labels: bool,
) -> Tuple[Any, Dict[str, Any], str]:
    """Load model while optionally overriding train_summary_json for evaluation.

    The older evaluator relied on util.load_pinn.load_model(), which reconstructs params
    from config.json. If config.json contains a stale train_summary_json from a different
    soft-label folder, t-rescaling and voltage alignment can silently mismatch. This loader
    makes the summary path explicit and records it in debug output.
    """
    _add_repo_paths(repo_root)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path}")
    config = _load_json(config_path)
    original_summary = config.get("train_summary_json")
    soft_summary = soft_label_dir / "soft_label_summary.json"
    chosen_summary = original_summary
    if force_summary_from_soft_labels or not chosen_summary:
        if soft_summary.exists():
            chosen_summary = str(soft_summary.resolve())
            config["train_summary_json"] = chosen_summary
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir.resolve())

    try:
        from util.load_pinn import _make_params  # type: ignore
        from util.init_pinn import initialize_nn_from_params_config, safe_load  # type: ignore
    except ImportError:  # pragma: no cover
        from load_pinn import _make_params  # type: ignore
        from init_pinn import initialize_nn_from_params_config, safe_load  # type: ignore

    simple_model = bool(config.get("simple_model", False))
    prior_model = str(config.get("prior_model", "spm"))
    params = _make_params(simple_model=simple_model, prior_model=prior_model, train_summary_json=chosen_summary)
    nn = initialize_nn_from_params_config(params, config)
    nn = safe_load(nn, str(checkpoint))
    nn.model.eval()
    return nn, config, str(chosen_summary) if chosen_summary else "NONE"


def predict_dataset(
    nn: Any,
    variable: str,
    x: np.ndarray,
    x_params: np.ndarray,
    batch_size: int,
    debug_first: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    x = _as_2d(x)
    x_params = _as_2d(x_params)
    n = x.shape[0]
    preds = []
    debug_payload: Optional[Dict[str, Any]] = None

    out_index = _var_output_index(nn, variable)
    ind_deg_i0_a = getattr(nn, "ind_deg_i0_a", 0)
    ind_deg_ds_c = getattr(nn, "ind_deg_ds_c", 1)
    rescale_T = float(nn.params["rescale_T"])
    rescale_R_global = float(nn.params.get("rescale_R", 1.0))

    # IMPORTANT: after the ASSB per-electrode radial-scaling patch, the trained
    # network sees anode and cathode concentration branches with different
    # normalized radial coordinates:
    #   cs_a: r / rescale_R_a ~= r / Rs_a
    #   cs_c: r / rescale_R_c ~= r / Rs_c
    # Potential branches do not have a physical r dimension in the soft-label
    # data, so they keep the legacy/global dummy radial scale.
    # This mirrors the soft-label generator convention where data_cs_a.r spans
    # [0, Rs_a] and data_cs_c.r spans [0, Rs_c].
    if variable == "cs_a":
        radial_scale = float(nn.params.get("rescale_R_a", nn.params.get("Rs_a", rescale_R_global)))
        radial_scale_key = "rescale_R_a"
    elif variable == "cs_c":
        radial_scale = float(nn.params.get("rescale_R_c", nn.params.get("Rs_c", rescale_R_global)))
        radial_scale_key = "rescale_R_c"
    else:
        radial_scale = rescale_R_global
        radial_scale_key = "rescale_R"
    if not np.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError(f"Invalid radial scale for {variable}: {radial_scale}")

    device = nn.device if hasattr(nn, "device") else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = x[start:stop]
        pb = x_params[start:stop]
        t = xb[:, 0:1]
        r = xb[:, 1:2] if xb.shape[1] >= 2 else np.zeros_like(t)
        if pb.shape[1] >= 2:
            deg_i0_a = pb[:, 0:1]
            deg_ds_c = pb[:, 1:2]
        else:
            deg_i0_a = np.ones_like(t)
            deg_ds_c = np.ones_like(t)

        t_t = _torch_tensor(t, device)
        r_t = _torch_tensor(r, device)
        deg_i0_t = _torch_tensor(deg_i0_a, device)
        deg_ds_t = _torch_tensor(deg_ds_c, device)
        with torch.no_grad():
            model_inputs = [
                t_t / rescale_T,
                r_t / radial_scale,
                nn.rescale_param(deg_i0_t, ind_deg_i0_a),
                nn.rescale_param(deg_ds_t, ind_deg_ds_c),
            ]
            raw_all = _call_model(nn.model, model_inputs)
            raw = _get_raw_output(raw_all, out_index)
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
        if debug_first and debug_payload is None:
            debug_payload = {
                "variable": variable,
                "out_index": out_index,
                "first_x_rows": xb[:5].tolist(),
                "first_x_params_rows": pb[:5].tolist(),
                "radial_scale_key": radial_scale_key,
                "radial_scale_used": radial_scale,
                "global_rescale_R": rescale_R_global,
                "first_t_over_rescale_T": (t[:5] / rescale_T).tolist(),
                "first_r_over_radial_scale": (r[:5] / radial_scale).tolist(),
                "first_raw": raw.detach().cpu().numpy().reshape(-1)[:5].tolist(),
                "first_pred": pred.detach().cpu().numpy().reshape(-1)[:5].tolist(),
            }
        preds.append(pred.detach().cpu().numpy().reshape(-1, 1))
    return np.vstack(preds), debug_payload


def _dataset_debug(name: str, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"keys": list(data.keys())}
    for key in ["x_train", "y_train", "x_params_train"]:
        if key in data:
            arr = np.asarray(data[key])
            rep[f"{key}_shape"] = list(arr.shape)
            if key == "x_train":
                x = _as_2d(arr)
                rep["t_stats"] = _summary_stats(x[:, 0])
                if x.shape[1] > 1:
                    rep["r_stats"] = _summary_stats(x[:, 1])
            elif key == "y_train":
                rep["y_stats"] = _summary_stats(arr)
    return rep


def _radial_consistency_report(nn: Any, datasets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}
    param_values = {
        "rescale_R": _safe_get_param(nn, "rescale_R"),
        "rescale_R_a": _safe_get_param(nn, "rescale_R_a", _safe_get_param(nn, "Rs_a")),
        "rescale_R_c": _safe_get_param(nn, "rescale_R_c", _safe_get_param(nn, "Rs_c")),
        "Rs_a": _safe_get_param(nn, "Rs_a"),
        "Rs_c": _safe_get_param(nn, "Rs_c"),
    }
    rep["model_radial_params"] = param_values
    for var, expected_key in [("cs_a", "rescale_R_a"), ("cs_c", "rescale_R_c")]:
        data = datasets.get(var)
        if not data or "x_train" not in data:
            continue
        x = _as_2d(data["x_train"])
        if x.shape[1] < 2:
            continue
        r = x[:, 1].astype(float)
        rmax_data = float(np.nanmax(r))
        scale = float(param_values.get(expected_key, float("nan")))
        global_scale = float(param_values.get("rescale_R", float("nan")))
        rep[var] = {
            "data_r_min": float(np.nanmin(r)),
            "data_r_max": rmax_data,
            "scale_key_used": expected_key,
            "scale_used": scale,
            "data_rmax_over_scale_used": float(rmax_data / scale) if np.isfinite(scale) and scale > 0 else float("nan"),
            "data_rmax_over_global_rescale_R": float(rmax_data / global_scale) if np.isfinite(global_scale) and global_scale > 0 else float("nan"),
        }
    return rep


def _solution_voltage_metrics(phis_x: np.ndarray, phis_label: np.ndarray, solution: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if not solution:
        return {"available": False, "reason": "solution.npz not found or could not be loaded"}
    tkey = _find_key(solution, ["t", "t_s", "time_s", "time", "t_eval"])
    vkey = _find_key(solution, ["V_exp", "voltage_exp", "voltage_V", "V_record", "V_meas", "voltage_measured", "voltage"])
    if tkey is None or vkey is None:
        return {"available": False, "reason": "compatible time/voltage keys not found", "solution_keys": list(solution.keys())}
    t_label = _as_2d(phis_x)[:, 0].reshape(-1)
    v_label = _as_2d(phis_label).reshape(-1)
    t_exp = np.asarray(solution[tkey], dtype=np.float64).reshape(-1)
    v_exp = np.asarray(solution[vkey], dtype=np.float64).reshape(-1)
    mask = np.isfinite(t_exp) & np.isfinite(v_exp)
    if mask.sum() < 2:
        return {"available": False, "reason": "not enough finite experiment voltage samples"}
    t_exp = t_exp[mask]
    v_exp = v_exp[mask]
    order = np.argsort(t_exp)
    t_exp = t_exp[order]
    v_exp = v_exp[order]
    t_unique, idx_unique = np.unique(t_exp, return_index=True)
    v_unique = v_exp[idx_unique]
    v_interp = np.interp(t_label, t_unique, v_unique)
    return {
        "available": True,
        "time_key": tkey,
        "voltage_key": vkey,
        "metrics_phis_c_label_vs_experiment": _metrics(v_interp, v_label),
    }


def _write_metrics_text(metrics: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("ASSB cycle5 PINN vs soft-label debug evaluation")
    lines.append("=" * 64)
    lines.append(f"model_dir: {metrics.get('model_dir')}")
    lines.append(f"checkpoint: {metrics.get('checkpoint')}")
    lines.append(f"soft_label_dir: {metrics.get('soft_label_dir')}")
    lines.append(f"output_dir: {metrics.get('output_dir')}")
    lines.append(f"device: {metrics.get('device')}")
    lines.append(f"config_prior_model: {metrics.get('config_prior_model')}")
    lines.append(f"config_train_summary_json_original: {metrics.get('config_train_summary_json_original')}")
    lines.append(f"chosen_train_summary_json: {metrics.get('chosen_train_summary_json')}")
    lines.append("")
    lines.append("[model params]")
    for k, v in metrics.get("model_params", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("[radial consistency]")
    lines.append(json.dumps(metrics.get("radial_consistency", {}), indent=2, ensure_ascii=False))
    lines.append("")
    for key in ["phie", "phis_c", "cs_a", "cs_c", "theta_a", "theta_c"]:
        if key not in metrics:
            continue
        m = metrics[key]
        lines.append(f"[{key}]")
        for kk in ["n", "mae", "rmse", "maxabs", "bias_mean", "corr", "label_min", "label_max", "label_std", "pred_min", "pred_max", "pred_std", "std_ratio_pred_over_label"]:
            if kk in m:
                lines.append(f"  {kk}: {m[kk]}")
        lines.append("")
    lines.append("[soft-label voltage sanity]")
    lines.append(json.dumps(metrics.get("soft_label_voltage_sanity", {}), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("[diagnosis]")
    for item in metrics.get("diagnosis", []):
        lines.append(f"  - {item}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _diagnose(metrics: Dict[str, Any]) -> list[str]:
    diagnosis: list[str] = []
    phis = metrics.get("phis_c", {})
    theta_c = metrics.get("theta_c", {})
    theta_a = metrics.get("theta_a", {})
    if phis:
        corr = phis.get("corr", float("nan"))
        mae = phis.get("mae", float("nan"))
        ratio = phis.get("std_ratio_pred_over_label", float("nan"))
        if np.isfinite(corr) and corr >= 0.85 and np.isfinite(mae) and mae <= 0.08:
            diagnosis.append("phis_c passes the cycle5 debug target: corr >= 0.85 and MAE <= 0.08 V.")
        elif np.isfinite(corr) and corr < 0.2:
            diagnosis.append("phis_c correlation is low. First suspect: training did not learn cycle5 dynamics, stale summary/rescale mismatch, wrong checkpoint, or output index mismatch.")
        if np.isfinite(ratio) and ratio < 0.2:
            diagnosis.append("phis_c prediction has very small variance relative to label. This looks like a constant/trivial solution or severe rescale issue.")
        if np.isfinite(mae) and mae > 0.2:
            diagnosis.append("phis_c MAE is above 0.2 V. Do not open data loss yet; check loading/rescale/boundary loss first.")
    if theta_c:
        corr = theta_c.get("corr", float("nan"))
        if np.isfinite(corr) and corr < 0:
            diagnosis.append("theta_c correlation is negative. Check cathode theta/cs conversion, current sign, and bidirectional rescale mapping.")
    if theta_a:
        ratio = theta_a.get("std_ratio_pred_over_label", float("nan"))
        if np.isfinite(ratio) and ratio < 0.2:
            diagnosis.append("theta_a prediction variance is too small. This supports a boundary-flux/trivial-solution diagnosis.")
    if not diagnosis:
        diagnosis.append("No obvious failure signature from simple thresholds. Inspect PNG plots and debug_model_and_data.json.")
    return diagnosis


def _try_load_solution(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        return _load_npz(path)
    except Exception:
        return {}


def _make_plots(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, solution: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib import failed; skip plots. {exc}")
        return

    # Potential time curves.
    for var in ["phis_c", "phie"]:
        if var not in results:
            continue
        x = results[var]["x"]
        y = results[var]["label"].reshape(-1)
        p = results[var]["pred"].reshape(-1)
        t = x[:, 0].reshape(-1)
        order = np.argsort(t)
        plt.figure(figsize=(10, 4.8))
        plt.plot(t[order], y[order], label=f"soft label {var}")
        plt.plot(t[order], p[order], label=f"PINN {var}", alpha=0.85)
        if var == "phis_c" and solution:
            tkey = _find_key(solution, ["t", "t_s", "time_s", "time", "t_eval"])
            vkey = _find_key(solution, ["V_exp", "voltage_exp", "voltage_V", "V_record", "V_meas", "voltage_measured", "voltage"])
            if tkey is not None and vkey is not None:
                ts = np.asarray(solution[tkey]).reshape(-1)
                vv = np.asarray(solution[vkey]).reshape(-1)
                if ts.size == vv.size and ts.size > 1:
                    plt.plot(ts, vv, label="experiment voltage", alpha=0.5)
        plt.xlabel("time / s")
        plt.ylabel(f"{var} / V")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"timeseries_{var}.png", dpi=180)
        plt.close()

    # Surface concentration time curves.
    for var, ylabel in [("cs_a", "cs_a"), ("cs_c", "cs_c")]:
        if var not in results:
            continue
        x = results[var]["x"]
        if x.shape[1] < 2:
            continue
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

    # Correlation plots.
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
            idx = _subsample_indices(y.size, 8000, mode="random", seed=17)
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
    parser = argparse.ArgumentParser(description="Evaluate ASSB cycle5 PINN against soft labels with debug checks and per-electrode radial scaling")
    parser.add_argument("--repo_root", default=None, help="Project root. Default: script directory")
    parser.add_argument("--model_dir", required=True, help="Saved model folder, e.g. ModelFin_61")
    parser.add_argument("--checkpoint", default=None, help="Exact checkpoint file name/path. Default priority: best.pt, last.pt, lastLBFGS.pt, lastSGD.pt")
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3", help="Folder with data_*.npz and solution.npz")
    parser.add_argument("--ocp_dir", default=None, help="Optional OCP folder; exported as ASSB_OCP_DIR before loading")
    parser.add_argument("--output_dir", default=None, help="Evaluation output folder")
    parser.add_argument("--max_time_points", type=int, default=5000, help="Max potential points to evaluate; <=0 means all")
    parser.add_argument("--max_cs_points", type=int, default=150000, help="Max concentration points to evaluate; <=0 means all")
    parser.add_argument("--batch_size", type=int, default=8192, help="Prediction batch size")
    parser.add_argument("--no_plots", action="store_true", help="Only write metrics, no PNG plots")
    parser.add_argument("--debug_print_first_batch", action="store_true", help="Print and save first-row inputs/raw outputs/predictions")
    parser.add_argument("--no_force_summary_from_soft_labels", action="store_true", help="Use config.json train_summary_json as-is. Not recommended for cycle5 debugging.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_script().resolve()
    soft_label_dir = Path(args.soft_label_dir)
    if not soft_label_dir.is_absolute():
        soft_label_dir = repo_root / soft_label_dir
    soft_label_dir = soft_label_dir.resolve()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    model_dir = model_dir.resolve()
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / f"Eval_{model_dir.name}_vs_{soft_label_dir.name}_debug"
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ocp_dir:
        os.environ["ASSB_OCP_DIR"] = str(Path(args.ocp_dir).resolve())
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir)

    checkpoint = _find_checkpoint(model_dir, args.checkpoint)
    config_original = _load_json(model_dir / "config.json")
    nn, config_used, chosen_summary = load_model_for_eval(
        repo_root=repo_root,
        model_dir=model_dir,
        soft_label_dir=soft_label_dir,
        checkpoint=checkpoint,
        force_summary_from_soft_labels=(not args.no_force_summary_from_soft_labels),
    )

    print(f"INFO: repo_root = {repo_root}")
    print(f"INFO: model_dir = {model_dir}")
    print(f"INFO: checkpoint = {checkpoint}")
    print(f"INFO: soft_label_dir = {soft_label_dir}")
    print(f"INFO: chosen_train_summary_json = {chosen_summary}")
    print(f"INFO: output_dir = {out_dir}")
    print(f"INFO: model device = {getattr(nn, 'device', 'unknown')}")

    datasets = {var: _load_npz(soft_label_dir / fname) for var, fname in REQUIRED_NPZ.items()}
    results: Dict[str, Dict[str, np.ndarray]] = {}
    debug_payloads: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "soft_label_dir": str(soft_label_dir),
        "output_dir": str(out_dir),
        "device": str(getattr(nn, "device", "unknown")),
        "config_prior_model": config_used.get("prior_model"),
        "config_train_summary_json_original": config_original.get("train_summary_json"),
        "chosen_train_summary_json": chosen_summary,
        "model_params": {
            "rescale_T": _safe_get_param(nn, "rescale_T"),
            "rescale_R": _safe_get_param(nn, "rescale_R"),
            "rescale_R_a": _safe_get_param(nn, "rescale_R_a", _safe_get_param(nn, "Rs_a")),
            "rescale_R_c": _safe_get_param(nn, "rescale_R_c", _safe_get_param(nn, "Rs_c")),
            "radial_rescale_mode": str(getattr(nn, "params", {}).get("radial_rescale_mode", "unknown")),
            "use_per_electrode_rescale_R": str(getattr(nn, "params", {}).get("use_per_electrode_rescale_R", "unknown")),
            "csanmax": _safe_get_param(nn, "csanmax"),
            "cscamax": _safe_get_param(nn, "cscamax"),
            "R_ohm_eff": _safe_get_param(nn, "R_ohm_eff"),
            "voltage_alignment_offset_V": _safe_get_param(nn, "voltage_alignment_offset_V", _safe_get_param(nn, "voltage_alignment_offset")),
            "theta_c_bottom": _safe_get_param(nn, "theta_c_bottom", _safe_get_param(nn, "theta_c_bottom_v3")),
            "theta_c_top": _safe_get_param(nn, "theta_c_top", _safe_get_param(nn, "theta_c_top_v3")),
            "Rs_a": _safe_get_param(nn, "Rs_a"),
            "Rs_c": _safe_get_param(nn, "Rs_c"),
        },
        "output_indices": {var: _var_output_index(nn, var) for var in REQUIRED_NPZ.keys()},
        "datasets": {var: _dataset_debug(var, data) for var, data in datasets.items()},
    }
    metrics["radial_consistency"] = _radial_consistency_report(nn, datasets)

    for var, data in datasets.items():
        x = _as_2d(data["x_train"])
        y = _as_2d(data["y_train"])
        xp = _as_2d(data.get("x_params_train", np.ones((x.shape[0], 2), dtype=np.float64)))
        mode = "uniform" if var in {"phie", "phis_c"} else "random"
        max_points = args.max_time_points if var in {"phie", "phis_c"} else args.max_cs_points
        idx = _subsample_indices(x.shape[0], max_points, mode=mode, seed=7)
        print(f"INFO: predicting {var}: using {len(idx)} / {x.shape[0]} points")
        pred, dbg = predict_dataset(
            nn,
            var,
            x[idx],
            xp[idx],
            batch_size=args.batch_size,
            debug_first=args.debug_print_first_batch,
        )
        results[var] = {"x": x[idx], "label": y[idx], "pred": pred, "x_params": xp[idx]}
        metrics[var] = _metrics(y[idx], pred)
        if dbg is not None:
            dbg["first_label"] = y[idx][:5].reshape(-1).tolist()
            debug_payloads[var] = dbg
            print(f"DEBUG {var}: first_x_rows={dbg['first_x_rows']}")
            print(f"DEBUG {var}: first_label={dbg['first_label']}")
            print(f"DEBUG {var}: first_pred={dbg['first_pred']}")
        if var == "cs_a":
            csmax = metrics["model_params"]["csanmax"]
            if np.isfinite(csmax) and csmax != 0:
                metrics["theta_a"] = _metrics(y[idx] / csmax, pred / csmax)
        elif var == "cs_c":
            csmax = metrics["model_params"]["cscamax"]
            if np.isfinite(csmax) and csmax != 0:
                metrics["theta_c"] = _metrics(y[idx] / csmax, pred / csmax)

    solution = _try_load_solution(soft_label_dir / "solution.npz")
    metrics["solution_keys"] = list(solution.keys()) if solution else []
    metrics["soft_label_voltage_sanity"] = _solution_voltage_metrics(
        results["phis_c"]["x"],
        results["phis_c"]["label"],
        solution,
    )
    metrics["diagnosis"] = _diagnose(metrics)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_metrics_text(metrics, out_dir / "metrics_summary.txt")
    (out_dir / "debug_model_and_data.json").write_text(json.dumps({"debug_first_batch": debug_payloads, "datasets": metrics["datasets"], "model_params": metrics["model_params"], "output_indices": metrics["output_indices"], "radial_consistency": metrics.get("radial_consistency", {})}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_plots:
        _make_plots(results, out_dir, solution)

    print("\nEvaluation finished.")
    print(f"Metrics TXT : {out_dir / 'metrics_summary.txt'}")
    print(f"Metrics JSON: {out_dir / 'metrics.json'}")
    print(f"Debug JSON  : {out_dir / 'debug_model_and_data.json'}")
    print("\nKey metrics:")
    for key in ["phis_c", "phie", "theta_a", "theta_c", "cs_a", "cs_c"]:
        if key in metrics:
            m = metrics[key]
            print(f"  {key:8s} MAE={m['mae']:.6g} RMSE={m['rmse']:.6g} MAX={m['maxabs']:.6g} corr={m['corr']:.6g} std_ratio={m['std_ratio_pred_over_label']:.6g}")
    print("\nDiagnosis:")
    for item in metrics["diagnosis"]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
