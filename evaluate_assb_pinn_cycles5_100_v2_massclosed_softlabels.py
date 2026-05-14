#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ModelFin_103 against the new v2 mass-closed candidate soft labels.

Default task
------------
Model:      ModelFin_103
Soft labels:C:\\Users\\Tiga_QJW\\Desktop\\ASSB_Scheme_V1\\assb_soft_labels_cycle5_522_v2_massclosed_candidate
Cycles:     5-100, inclusive
Reference:  soft labels only; voltage_exp is intentionally ignored.

The script keeps the original continuous t_global_s coordinate after cycle
filtering. Time is NOT reset at cycle 5 or at cycle boundaries.

Outputs
-------
EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only/
  metrics_global.json
  metrics_by_cycle.csv
  debug_model_and_data.json
  eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz
  plots_softlabel_only/*.png

This is a patched v2 evaluator for the D4 mass-closure check. It follows the
same model-loading and output-rescaling logic as the existing cycle5-100
soft-label-only evaluator, but changes the default soft-label directory and
forces the runtime summary environment to the new v2 dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ASSB_COMPARE_EXPERIMENT_VOLTAGE", "False")
os.environ.setdefault("ASSB_EVAL_REFERENCE", "soft_labels_only")

import numpy as np
import torch

SCRIPT_VERSION = "ModelFin103-cycle5-100-v2-massclosed-softlabel-only-v1"

DEFAULT_SOFT_LABEL_DIR = Path(
    r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
)
DEFAULT_OCP_DIR = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs")
DEFAULT_OUTPUT_DIR = Path("EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only")

POTENTIAL_KEYS = {
    "phie": ("phie", "phi_e", "data_phie"),
    "phis_c": ("phis_c", "phi_s_c", "phis", "voltage_soft", "V_soft"),
}
CONCENTRATION_KEYS = {
    "cs_a": ("cs_a", "c_s_a", "csan", "csa"),
    "cs_c": ("cs_c", "c_s_c", "csca", "csc"),
}
THETA_KEYS = {
    "theta_a": ("theta_a", "theta_an", "theta_negative"),
    "theta_c": ("theta_c", "theta_ca", "theta_positive"),
}
R_KEYS = {
    "cs_a": ("r_a", "ra", "r_an", "r_negative"),
    "cs_c": ("r_c", "rc", "r_ca", "r_positive"),
}


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_paths(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    util_dir = repo_root / "util"
    for p in (repo_root, util_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _find_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_list = list(keys)
    lower = {k.lower(): k for k in key_list}
    for cand in candidates:
        if cand in key_list:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _finite_1d(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    return np.asarray(arr, dtype=dtype).reshape(-1)


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
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "maxabs": float("nan"),
            "bias_mean": float("nan"),
            "corr": float("nan"),
            "r2": float("nan"),
            "label_min": float("nan"),
            "label_max": float("nan"),
            "label_range": float("nan"),
            "label_std": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "pred_std": float("nan"),
            "std_ratio_pred_over_label": float("nan"),
            "nmae": float("nan"),
            "nrmse": float("nan"),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    label_range = float(np.max(yt) - np.min(yt))
    label_std = float(np.std(yt))
    pred_std = float(np.std(yp))
    corr = (
        float(np.corrcoef(yt, yp)[0, 1])
        if yt.size > 1 and label_std > 0 and pred_std > 0
        else float("nan")
    )
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "n": int(yt.size),
        "mae": mae,
        "rmse": rmse,
        "maxabs": float(np.max(np.abs(err))),
        "bias_mean": float(np.mean(err)),
        "corr": corr,
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "label_min": float(np.min(yt)),
        "label_max": float(np.max(yt)),
        "label_range": label_range,
        "label_std": label_std,
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
        "pred_std": pred_std,
        "std_ratio_pred_over_label": float(pred_std / label_std) if label_std > 0 else float("nan"),
        "nmae": float(mae / label_range) if label_range > 0 else float("nan"),
        "nrmse": float(rmse / label_range) if label_range > 0 else float("nan"),
    }


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _torch_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_as_2d(x), dtype=torch.float64, device=device)


def _find_checkpoint(model_dir: Path, checkpoint: Optional[str]) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = model_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Requested checkpoint not found: {p}")
        return p.resolve()
    for name in ("best.pt", "last.pt", "lastLBFGS.pt", "lastSGD.pt", "best.weights.h5"):
        p = model_dir / name
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


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


def _prepare_runtime_environment(soft_label_dir: Path, ocp_dir: Optional[Path]) -> str:
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir.resolve())
    os.environ["ASSB_COMPARE_EXPERIMENT_VOLTAGE"] = "False"
    os.environ["ASSB_EVAL_REFERENCE"] = "soft_labels_only"
    if ocp_dir is not None:
        os.environ["ASSB_OCP_DIR"] = str(ocp_dir.resolve())

    # Important D4 safeguard: do not let an old cycle5_v4 or v1 summary path
    # override the v2 mass-closed candidate dataset during _make_params().
    soft_summary = soft_label_dir / "soft_label_summary.json"
    if soft_summary.exists():
        os.environ["ASSB_SOFT_LABEL_SUMMARY"] = str(soft_summary.resolve())
        return str(soft_summary.resolve())
    os.environ.pop("ASSB_SOFT_LABEL_SUMMARY", None)
    return "NONE"


def load_model_for_eval(
    *,
    repo_root: Path,
    model_dir: Path,
    soft_label_dir: Path,
    checkpoint: Path,
    ocp_dir: Optional[Path],
) -> Tuple[Any, Dict[str, Any], str]:
    _add_repo_paths(repo_root)
    chosen_summary = _prepare_runtime_environment(soft_label_dir, ocp_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path}")
    config = _load_json(config_path)

    try:
        from util.load_pinn import _make_params  # type: ignore
        from util.init_pinn import initialize_nn_from_params_config, safe_load  # type: ignore
    except ImportError:  # pragma: no cover
        from load_pinn import _make_params  # type: ignore
        from init_pinn import initialize_nn_from_params_config, safe_load  # type: ignore

    simple_model = bool(config.get("simple_model", False))
    prior_model = str(config.get("prior_model", "assb_discharge"))

    # Prefer explicit summary injection. Keep fallback for older local versions.
    try:
        params = _make_params(simple_model=simple_model, prior_model=prior_model, train_summary_json=chosen_summary)
    except TypeError:
        params = _make_params(simple_model=simple_model, prior_model=prior_model)

    nn = initialize_nn_from_params_config(params, config)
    nn = safe_load(nn, str(checkpoint))
    nn.model.eval()
    return nn, config, chosen_summary


def predict_dataset(
    *,
    nn: Any,
    variable: str,
    t_s: np.ndarray,
    r_m: Optional[np.ndarray],
    batch_size: int,
    debug_first: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1, 1)
    if r_m is None:
        r_m = np.zeros_like(t_s)
    else:
        r_m = np.asarray(r_m, dtype=np.float64).reshape(-1, 1)

    if t_s.shape[0] != r_m.shape[0]:
        raise ValueError(f"t and r length mismatch for {variable}: {t_s.shape[0]} vs {r_m.shape[0]}")

    n = t_s.shape[0]
    pred_chunks = []
    debug_payload: Optional[Dict[str, Any]] = None

    out_index = _var_output_index(nn, variable)
    ind_deg_i0_a = getattr(nn, "ind_deg_i0_a", 0)
    ind_deg_ds_c = getattr(nn, "ind_deg_ds_c", 1)

    rescale_T = float(nn.params["rescale_T"])
    rescale_R_global = float(nn.params.get("rescale_R", 1.0))
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
        tb = t_s[start:stop]
        rb = r_m[start:stop]
        deg_i0 = np.ones_like(tb)
        deg_ds = np.ones_like(tb)

        t_t = _torch_tensor(tb, device)
        r_t = _torch_tensor(rb, device)
        deg_i0_t = _torch_tensor(deg_i0, device)
        deg_ds_t = _torch_tensor(deg_ds, device)

        with torch.no_grad():
            inputs = [
                t_t / rescale_T,
                r_t / radial_scale,
                nn.rescale_param(deg_i0_t, ind_deg_i0_a),
                nn.rescale_param(deg_ds_t, ind_deg_ds_c),
            ]
            raw_all = _call_model(nn.model, inputs)
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
                "radial_scale_key": radial_scale_key,
                "radial_scale_used": radial_scale,
                "rescale_T": rescale_T,
                "first_t_s": tb[:5].reshape(-1).tolist(),
                "first_r_m": rb[:5].reshape(-1).tolist(),
                "first_t_over_rescale_T": (tb[:5] / rescale_T).reshape(-1).tolist(),
                "first_r_over_radial_scale": (rb[:5] / radial_scale).reshape(-1).tolist(),
                "first_raw": raw.detach().cpu().numpy().reshape(-1)[:5].tolist(),
                "first_pred": pred.detach().cpu().numpy().reshape(-1)[:5].tolist(),
            }

        pred_chunks.append(pred.detach().cpu().numpy().reshape(-1))

    return np.concatenate(pred_chunks), debug_payload


def _load_solution(soft_label_dir: Path) -> Dict[str, Any]:
    path = soft_label_dir / "solution.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing continuous soft-label solution: {path}")

    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        t_key = _find_key(keys, ("t_global_s", "t_global", "time_global_s", "t", "time_s", "time"))
        cycle_key = _find_key(keys, ("cycle_id", "cycle", "cycle_index"))
        if t_key is None:
            raise KeyError(f"solution.npz lacks a compatible time key. Keys={keys}")
        t = _finite_1d(data[t_key])

        out: Dict[str, Any] = {"path": str(path.resolve()), "keys": keys, "t_key": t_key, "t": t}
        if cycle_key is not None:
            out["cycle_key"] = cycle_key
            out["cycle_id"] = _finite_1d(data[cycle_key], dtype=np.int64)

        for var, candidates in POTENTIAL_KEYS.items():
            key = _find_key(keys, candidates)
            if key is None:
                raise KeyError(f"solution.npz lacks required key for {var}. Tried {candidates}. Keys={keys}")
            out[f"{var}_key"] = key
            out[var] = _finite_1d(data[key])

        for var, candidates in CONCENTRATION_KEYS.items():
            key = _find_key(keys, candidates)
            if key is None:
                raise KeyError(f"solution.npz lacks required key for {var}. Tried {candidates}. Keys={keys}")
            out[f"{var}_key"] = key
            out[var] = np.asarray(data[key])

        for theta_var, candidates in THETA_KEYS.items():
            key = _find_key(keys, candidates)
            if key is not None:
                out[f"{theta_var}_key"] = key
                out[theta_var] = np.asarray(data[key])

        for var, candidates in R_KEYS.items():
            key = _find_key(keys, candidates)
            if key is None:
                raise KeyError(f"solution.npz lacks required radial grid key for {var}. Tried {candidates}. Keys={keys}")
            out[f"{var}_r_key"] = key
            out[f"r_{var[-1]}"] = _finite_1d(data[key])

        # Keep voltage_exp provenance out of metric calculations by design.
        vkey = _find_key(keys, ("voltage_exp", "V_exp", "voltage_measured", "V_record"))
        if vkey is not None:
            out["voltage_exp_key_ignored"] = vkey

    n = len(out["t"])
    for key in ("phie", "phis_c"):
        if len(out[key]) != n:
            raise ValueError(f"{key} length {len(out[key])} != time length {n}")
    for key in ("cs_a", "cs_c"):
        arr = np.asarray(out[key])
        if arr.shape[0] != n:
            raise ValueError(f"{key} shape {arr.shape} has first dim != time length {n}")
    for key in ("theta_a", "theta_c"):
        if key in out:
            arr = np.asarray(out[key])
            if arr.shape[0] != n:
                raise ValueError(f"{key} shape {arr.shape} has first dim != time length {n}")
    if "cycle_id" in out and len(out["cycle_id"]) != n:
        raise ValueError(f"cycle_id length {len(out['cycle_id'])} != time length {n}")
    return out


def _filter_solution_by_cycle(sol: Dict[str, Any], cycle_from: int, cycle_to: int) -> Dict[str, Any]:
    """Filter solution arrays by inclusive cycle range while keeping t_global_s unchanged."""
    if "cycle_id" not in sol:
        raise KeyError("solution.npz has no cycle_id field; cannot perform cycle-range evaluation")
    if int(cycle_to) < int(cycle_from):
        raise ValueError(f"Invalid cycle range: cycle_to={cycle_to} < cycle_from={cycle_from}")

    cycle_id = np.asarray(sol["cycle_id"], dtype=np.int64).reshape(-1)
    mask = (cycle_id >= int(cycle_from)) & (cycle_id <= int(cycle_to))
    idx = np.where(mask)[0]
    if idx.size == 0:
        available = np.unique(cycle_id)
        raise ValueError(
            f"No points found for cycle range {cycle_from}-{cycle_to}. "
            f"Available cycle range: {int(available.min())}-{int(available.max())}"
        )

    out = dict(sol)
    out["cycle_filter"] = {
        "cycle_from": int(cycle_from),
        "cycle_to": int(cycle_to),
        "n_time_before_filter": int(cycle_id.size),
        "n_time_after_filter": int(idx.size),
        "cycle_min_after_filter": int(cycle_id[idx].min()),
        "cycle_max_after_filter": int(cycle_id[idx].max()),
        "cycle_count_after_filter": int(np.unique(cycle_id[idx]).size),
        "kept_original_time_values": True,
    }
    out["t"] = np.asarray(sol["t"])[idx]
    out["cycle_id"] = cycle_id[idx]
    for var in ("phie", "phis_c"):
        out[var] = np.asarray(sol[var])[idx]
    for var in ("cs_a", "cs_c", "theta_a", "theta_c"):
        if var in sol:
            out[var] = np.asarray(sol[var])[idx, ...]
    return out


def _build_cs_eval_arrays(sol: Dict[str, Any], var: str, time_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cs = np.asarray(sol[var], dtype=np.float64)
    r = np.asarray(sol[f"r_{var[-1]}"], dtype=np.float64).reshape(-1)
    t = np.asarray(sol["t"], dtype=np.float64).reshape(-1)[time_indices]
    label = cs[time_indices, :]
    T = np.repeat(t, r.size)
    R = np.tile(r, t.size)
    Y = label.reshape(-1)
    return T, R, Y


def _build_theta_label(sol: Dict[str, Any], theta_var: str, cs_var: str, time_indices: np.ndarray, csmax: float) -> np.ndarray:
    """Prefer theta arrays from solution.npz when present; otherwise derive from cs/csmax."""
    if theta_var in sol:
        arr = np.asarray(sol[theta_var], dtype=np.float64)
        if arr.ndim == 1:
            # Surface/mean theta vector: repeat along radial grid to match cs sample size.
            r_size = np.asarray(sol[f"r_{cs_var[-1]}"], dtype=np.float64).size
            return np.repeat(arr[time_indices].reshape(-1), r_size)
        return arr[time_indices, :].reshape(-1)
    cs = np.asarray(sol[cs_var], dtype=np.float64)[time_indices, :]
    return (cs / csmax).reshape(-1)


def _per_cycle_metrics(cycle_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> list[Dict[str, Any]]:
    cycle_id = np.asarray(cycle_id).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rows: list[Dict[str, Any]] = []
    for c in np.unique(cycle_id):
        mask = cycle_id == c
        if mask.sum() < 2:
            continue
        m = _metrics(y_true[mask], y_pred[mask])
        row: Dict[str, Any] = {"cycle_id": int(c)}
        row.update(m)
        rows.append(row)
    return rows


def _write_cycle_csv(path: Path, rows_by_var: Dict[str, list[Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variable",
        "cycle_id",
        "n",
        "mae",
        "rmse",
        "maxabs",
        "bias_mean",
        "corr",
        "r2",
        "nmae",
        "nrmse",
        "label_min",
        "label_max",
        "label_range",
        "label_std",
        "pred_min",
        "pred_max",
        "pred_std",
        "std_ratio_pred_over_label",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for var, rows in rows_by_var.items():
            for row in rows:
                payload = {k: row.get(k, "") for k in fieldnames}
                payload["variable"] = var
                writer.writerow(payload)


def _make_plots(out_dir: Path, results: Dict[str, Dict[str, np.ndarray]], cycle_rows: Dict[str, list[Dict[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; skip plots. {exc}")
        return

    plot_dir = out_dir / "plots_softlabel_only"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for var in ("phis_c", "phie"):
        if var not in results:
            continue
        t = results[var]["t"]
        y = results[var]["label"]
        p = results[var]["pred"]
        order = np.argsort(t)
        plt.figure(figsize=(11.0, 4.8))
        plt.plot(t[order], y[order], label=f"soft label {var}", linewidth=1.0)
        plt.plot(t[order], p[order], label=f"PINN {var}", linewidth=1.0, alpha=0.85)
        plt.xlabel("t_global_s / s")
        plt.ylabel(f"{var}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"timeseries_{var}_softlabel_only.png", dpi=180)
        plt.close()

    for var in ("phis_c", "phie", "theta_a", "theta_c", "cs_a", "cs_c"):
        rows = cycle_rows.get(var, [])
        if not rows:
            continue
        cyc = np.asarray([r["cycle_id"] for r in rows], dtype=float)
        mae = np.asarray([r["mae"] for r in rows], dtype=float)
        plt.figure(figsize=(10.5, 4.6))
        plt.plot(cyc, mae, marker=".", linewidth=0.8)
        plt.xlabel("cycle_id")
        plt.ylabel(f"{var} MAE vs soft label")
        plt.tight_layout()
        plt.savefig(plot_dir / f"per_cycle_mae_{var}_softlabel_only.png", dpi=180)
        plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ModelFin_103 against v2 mass-closed candidate continuous ASSB soft labels, cycle5-100 by default."
    )
    parser.add_argument("--repo_root", type=Path, default=None, help="Project root. Default: script directory")
    parser.add_argument("--model_dir", type=Path, default=Path("ModelFin_103"), help="Model directory. Default: %(default)s")
    parser.add_argument("--checkpoint", default=None, help="Exact checkpoint file name/path. Default: best.pt/last.pt priority")
    parser.add_argument("--soft_label_dir", type=Path, default=DEFAULT_SOFT_LABEL_DIR, help="Continuous soft-label directory. Default: %(default)s")
    parser.add_argument("--ocp_dir", type=Path, default=DEFAULT_OCP_DIR, help="OCP folder exported as ASSB_OCP_DIR. Default: %(default)s")
    parser.add_argument("--cycle_from", type=int, default=5, help="First cycle_id to evaluate, inclusive. Default: %(default)s")
    parser.add_argument("--cycle_to", type=int, default=100, help="Last cycle_id to evaluate, inclusive. Default: %(default)s")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Evaluation output directory. Default: %(default)s")
    parser.add_argument("--max_time_points", type=int, default=0, help="Max sampled potential time points after cycle filtering; <=0 means all. Default: %(default)s")
    parser.add_argument("--max_cs_time_points", type=int, default=12000, help="Max sampled concentration time rows after cycle filtering; <=0 means all. Default: %(default)s")
    parser.add_argument("--batch_size", type=int, default=8192, help="Prediction batch size. Default: %(default)s")
    parser.add_argument("--no_plots", action="store_true", help="Do not write PNG plots")
    parser.add_argument("--debug_print_first_batch", action="store_true", help="Save first-batch debug values")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else _repo_root_from_script().resolve()
    soft_label_dir = args.soft_label_dir if args.soft_label_dir.is_absolute() else repo_root / args.soft_label_dir
    model_dir = args.model_dir if args.model_dir.is_absolute() else repo_root / args.model_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    ocp_dir = args.ocp_dir if args.ocp_dir is not None else None
    if ocp_dir is not None and not ocp_dir.is_absolute():
        ocp_dir = repo_root / ocp_dir

    soft_label_dir = soft_label_dir.resolve()
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sol = _load_solution(soft_label_dir)
    sol = _filter_solution_by_cycle(sol, args.cycle_from, args.cycle_to)
    t = np.asarray(sol["t"], dtype=np.float64).reshape(-1)
    cycle_id = np.asarray(sol.get("cycle_id", np.full(t.shape, -1)), dtype=np.int64).reshape(-1)

    checkpoint = _find_checkpoint(model_dir, args.checkpoint)
    nn, config, chosen_summary = load_model_for_eval(
        repo_root=repo_root,
        model_dir=model_dir,
        soft_label_dir=soft_label_dir,
        checkpoint=checkpoint,
        ocp_dir=ocp_dir,
    )

    print(f"[INFO] script_version = {SCRIPT_VERSION}")
    print(f"[INFO] repo_root = {repo_root}")
    print(f"[INFO] model_dir = {model_dir}")
    print(f"[INFO] checkpoint = {checkpoint}")
    print(f"[INFO] soft_label_dir = {soft_label_dir}")
    print(f"[INFO] runtime_summary = {chosen_summary}")
    print(f"[INFO] output_dir = {output_dir}")
    print("[INFO] reference = soft labels only; voltage_exp is ignored")
    print(f"[INFO] cycle_filter = {args.cycle_from}-{args.cycle_to}; t_global_s is not reset")

    time_idx = _subsample_indices(t.size, args.max_time_points, mode="uniform", seed=11)
    cs_time_idx = _subsample_indices(t.size, args.max_cs_time_points, mode="uniform", seed=13)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    debug: Dict[str, Any] = {}
    metrics_global: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "reference": "soft_labels_only",
        "experiment_voltage_comparison": False,
        "repo_root": str(repo_root),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "soft_label_dir": str(soft_label_dir),
        "solution_npz": sol["path"],
        "chosen_train_summary_json": chosen_summary,
        "solution_keys": sol["keys"],
        "time_key": sol["t_key"],
        "cycle_key": sol.get("cycle_key"),
        "voltage_exp_key_ignored": sol.get("voltage_exp_key_ignored"),
        "n_time_total": int(t.size),
        "n_time_sampled_potential": int(time_idx.size),
        "n_time_sampled_concentration_rows": int(cs_time_idx.size),
        "t_stats": _summary_stats(t),
        "cycle_filter": sol.get("cycle_filter", {}),
        "cycle_stats": {
            "cycle_min": int(np.min(cycle_id)) if cycle_id.size and np.min(cycle_id) >= 0 else None,
            "cycle_max": int(np.max(cycle_id)) if cycle_id.size and np.max(cycle_id) >= 0 else None,
            "cycle_count": int(np.unique(cycle_id[cycle_id >= 0]).size) if cycle_id.size else 0,
        },
        "model_params": {
            "rescale_T": _safe_get_param(nn, "rescale_T"),
            "time_scale_s": _safe_get_param(nn, "time_scale_s", _safe_get_param(nn, "rescale_T")),
            "rescale_R": _safe_get_param(nn, "rescale_R"),
            "rescale_R_a": _safe_get_param(nn, "rescale_R_a", _safe_get_param(nn, "Rs_a")),
            "rescale_R_c": _safe_get_param(nn, "rescale_R_c", _safe_get_param(nn, "Rs_c")),
            "csanmax": _safe_get_param(nn, "csanmax"),
            "cscamax": _safe_get_param(nn, "cscamax"),
            "Rs_a": _safe_get_param(nn, "Rs_a"),
            "Rs_c": _safe_get_param(nn, "Rs_c"),
            "R_ohm_eff": _safe_get_param(nn, "R_ohm_eff"),
        },
        "config_prior_model": config.get("prior_model"),
        "output_indices": {v: _var_output_index(nn, v) for v in ("phie", "phis_c", "cs_a", "cs_c")},
        "theta_labels": {
            "theta_a_source": sol.get("theta_a_key", "derived_from_cs_a_over_csanmax"),
            "theta_c_source": sol.get("theta_c_key", "derived_from_cs_c_over_cscamax"),
        },
    }

    # Potential branches.
    for var in ("phis_c", "phie"):
        label = np.asarray(sol[var], dtype=np.float64).reshape(-1)[time_idx]
        pred, dbg = predict_dataset(
            nn=nn,
            variable=var,
            t_s=t[time_idx],
            r_m=None,
            batch_size=args.batch_size,
            debug_first=args.debug_print_first_batch,
        )
        results[var] = {"t": t[time_idx], "cycle_id": cycle_id[time_idx], "label": label, "pred": pred}
        metrics_global[var] = _metrics(label, pred)
        if dbg is not None:
            dbg["first_label"] = label[:5].reshape(-1).tolist()
            debug[var] = dbg

    # Concentration branches.
    for var in ("cs_a", "cs_c"):
        T, R, label = _build_cs_eval_arrays(sol, var, cs_time_idx)
        pred, dbg = predict_dataset(
            nn=nn,
            variable=var,
            t_s=T,
            r_m=R,
            batch_size=args.batch_size,
            debug_first=args.debug_print_first_batch,
        )
        r_grid = np.asarray(sol[f"r_{var[-1]}"], dtype=np.float64).reshape(-1)
        cyc_rep = np.repeat(cycle_id[cs_time_idx], r_grid.size)
        results[var] = {"t": T, "r": R, "cycle_id": cyc_rep, "label": label, "pred": pred}
        metrics_global[var] = _metrics(label, pred)

        csmax_key = "csanmax" if var == "cs_a" else "cscamax"
        theta_key = "theta_a" if var == "cs_a" else "theta_c"
        csmax = metrics_global["model_params"][csmax_key]
        if np.isfinite(csmax) and csmax != 0:
            theta_label = _build_theta_label(sol, theta_key, var, cs_time_idx, csmax)
            theta_pred = pred / csmax
            results[theta_key] = {"t": T, "r": R, "cycle_id": cyc_rep, "label": theta_label, "pred": theta_pred}
            metrics_global[theta_key] = _metrics(theta_label, theta_pred)

        if dbg is not None:
            dbg["first_label"] = label[:5].reshape(-1).tolist()
            debug[var] = dbg

    cycle_rows: Dict[str, list[Dict[str, Any]]] = {}
    if np.any(cycle_id >= 0):
        for var in ("phis_c", "phie", "cs_a", "cs_c", "theta_a", "theta_c"):
            if var in results:
                cycle_rows[var] = _per_cycle_metrics(results[var]["cycle_id"], results[var]["label"], results[var]["pred"])

    _write_json(output_dir / "metrics_global.json", metrics_global)
    _write_cycle_csv(output_dir / "metrics_by_cycle.csv", cycle_rows)
    _write_json(
        output_dir / "debug_model_and_data.json",
        {
            "debug_first_batch": debug,
            "solution_fields": {
                "time_key": sol["t_key"],
                "cycle_key": sol.get("cycle_key"),
                "phis_c_key": sol.get("phis_c_key"),
                "phie_key": sol.get("phie_key"),
                "cs_a_key": sol.get("cs_a_key"),
                "cs_c_key": sol.get("cs_c_key"),
                "theta_a_key": sol.get("theta_a_key"),
                "theta_c_key": sol.get("theta_c_key"),
                "voltage_exp_key_ignored": sol.get("voltage_exp_key_ignored"),
            },
            "model_params": metrics_global["model_params"],
            "output_indices": metrics_global["output_indices"],
            "sample_sizes": {
                "potential_points": int(time_idx.size),
                "concentration_time_rows": int(cs_time_idx.size),
                "cs_a_points": int(results["cs_a"]["label"].size),
                "cs_c_points": int(results["cs_c"]["label"].size),
            },
            "environment_overrides": {
                "ASSB_SOFT_LABEL_DIR": os.environ.get("ASSB_SOFT_LABEL_DIR"),
                "ASSB_SOFT_LABEL_SUMMARY": os.environ.get("ASSB_SOFT_LABEL_SUMMARY"),
                "ASSB_OCP_DIR": os.environ.get("ASSB_OCP_DIR"),
                "ASSB_EVAL_REFERENCE": os.environ.get("ASSB_EVAL_REFERENCE"),
            },
        },
    )

    # Save sampled arrays for follow-up cbar/radial audits.
    np.savez_compressed(
        output_dir / "eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz",
        t_potential=t[time_idx].astype(np.float64),
        cycle_id_potential=cycle_id[time_idx].astype(np.int32),
        phis_c_true=results["phis_c"]["label"].astype(np.float32),
        phis_c_pred=results["phis_c"]["pred"].astype(np.float32),
        phie_true=results["phie"]["label"].astype(np.float32),
        phie_pred=results["phie"]["pred"].astype(np.float32),
        t_cs=t[cs_time_idx].astype(np.float64),
        cycle_id_cs=cycle_id[cs_time_idx].astype(np.int32),
        r_a=np.asarray(sol["r_a"], dtype=np.float64),
        r_c=np.asarray(sol["r_c"], dtype=np.float64),
        cs_a_true=results["cs_a"]["label"].astype(np.float32),
        cs_a_pred=results["cs_a"]["pred"].astype(np.float32),
        cs_c_true=results["cs_c"]["label"].astype(np.float32),
        cs_c_pred=results["cs_c"]["pred"].astype(np.float32),
        theta_a_true=results.get("theta_a", {}).get("label", np.array([], dtype=np.float32)).astype(np.float32),
        theta_a_pred=results.get("theta_a", {}).get("pred", np.array([], dtype=np.float32)).astype(np.float32),
        theta_c_true=results.get("theta_c", {}).get("label", np.array([], dtype=np.float32)).astype(np.float32),
        theta_c_pred=results.get("theta_c", {}).get("pred", np.array([], dtype=np.float32)).astype(np.float32),
    )

    if not args.no_plots:
        _make_plots(output_dir, results, cycle_rows)

    print("\nEvaluation finished: v2 mass-closed candidate soft-label reference only.")
    print(f"metrics_global: {output_dir / 'metrics_global.json'}")
    print(f"metrics_by_cycle: {output_dir / 'metrics_by_cycle.csv'}")
    for key in ("phis_c", "phie", "theta_a", "theta_c", "cs_a", "cs_c"):
        if key in metrics_global:
            m = metrics_global[key]
            print(
                f"{key:8s} MAE={m['mae']:.6g} RMSE={m['rmse']:.6g} "
                f"R2={m['r2']:.6g} corr={m['corr']:.6g} NMAE={m['nmae']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
