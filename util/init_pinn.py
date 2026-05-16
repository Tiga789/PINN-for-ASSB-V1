# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 initialization.

Complete replacement file for ModelFin_110 aging-fix1. It keeps the original
pointwise data loss closed, loads the cycle table / capacity targets, and
registers the aging mechanism in myNN.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (str(_ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .myNN import myNN
    from .assb_aging_fix1_config import parse_aging_fix1_config
except Exception:  # pragma: no cover
    from myNN import myNN
    from assb_aging_fix1_config import parse_aging_fix1_config

try:
    from prettyPlot.parser import parse_input_file as _pretty_parse_input_file
except Exception:  # pragma: no cover
    _pretty_parse_input_file = None


def parse_input_file(path: str | Path) -> Dict[str, str]:
    if _pretty_parse_input_file is not None:
        return _pretty_parse_input_file(str(path))
    out: Dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
        elif "=" in line:
            k, v = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            k, v = parts
        out[k.strip()] = v.strip()
    return out


def _normalize_path_str(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    val = str(path).strip().strip('"').strip("'")
    if val.upper() in {"NONE", "NULL", ""}:
        return None
    return val


def _abs_path(path: Optional[str]) -> Optional[str]:
    s = _normalize_path_str(path)
    if s is None:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def absolute_path_check(path: Optional[str]) -> None:
    path = _normalize_path_str(path)
    if path is None:
        return
    if not os.path.isabs(path):
        raise SystemExit(f"ERROR: {path} is not absolute")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if s in {"false", "0", "no", "n", "f", "off", "none", "null", ""}:
        return False
    return bool(default)


def _get_bool(inpt: Dict[str, str], key: str, default: bool = False) -> bool:
    return _as_bool(inpt.get(key), default)


def _get_int(inpt: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(inpt[key]))
    except Exception:
        return int(default)


def _get_float(inpt: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(inpt[key])
    except Exception:
        return float(default)


def _get_str(inpt: Dict[str, str], key: str, default: str = "") -> str:
    return str(inpt.get(key, default)).strip()


def _parse_alpha(text: str | None) -> list[float]:
    if text is None:
        vals = [1.0, 1.0, 0.0, 1.0]
    else:
        vals = [float(x) for x in str(text).replace(",", " ").split()]
    while len(vals) < 4:
        vals.append(0.0)
    vals = vals[:4]
    vals[2] = 0.0  # old data loss is always closed in ModelFin_109.
    if abs(vals[3]) < 1.0e-16:
        vals[3] = 1.0
    return vals


def _apply_cbar_baseline_params(params: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    mapping = {
        "USE_I_CBAR_BASELINE": ("use_i_cbar_baseline", _as_bool),
        "USE_I_CBAR_BASELINE_A": ("use_i_cbar_baseline_a", _as_bool),
        "USE_I_CBAR_BASELINE_C": ("use_i_cbar_baseline_c", _as_bool),
        "CBAR_BASELINE_DEVIATION_FRACTION_A": ("cbar_deviation_fraction_a", float),
        "CBAR_BASELINE_DEVIATION_FRACTION_C": ("cbar_deviation_fraction_c", float),
        "USE_ZERO_MEAN_RADIAL_DEVIATION": ("use_zero_mean_radial_deviation", _as_bool),
        "USE_ZERO_MEAN_RADIAL_DEVIATION_A": ("use_zero_mean_radial_deviation_a", _as_bool),
        "USE_ZERO_MEAN_RADIAL_DEVIATION_C": ("use_zero_mean_radial_deviation_c", _as_bool),
        "CBAR_RADIAL_BASIS_MODE": ("cbar_radial_basis_mode", str),
        "USE_CURRENT_POTENTIAL_BASELINE": ("use_current_potential_baseline", _as_bool),
        "USE_CURRENT_POTENTIAL_BASELINE_PHIE": ("use_current_potential_baseline_phie", _as_bool),
        "USE_CURRENT_POTENTIAL_BASELINE_PHIS_C": ("use_current_potential_baseline_phis_c", _as_bool),
        "POTENTIAL_BASELINE_CORRECTION_FRACTION_PHIE": ("potential_baseline_correction_fraction_phie", float),
        "POTENTIAL_BASELINE_CORRECTION_FRACTION_PHIS_C": ("potential_baseline_correction_fraction_phis_c", float),
    }
    for key, (pkey, caster) in mapping.items():
        if key in cfg:
            params[pkey] = caster(cfg[key]) if caster is not _as_bool else _as_bool(cfg[key])
    if "POTENTIAL_BASELINE_CORRECTION_FRACTION" in cfg:
        v = float(cfg["POTENTIAL_BASELINE_CORRECTION_FRACTION"])
        params["potential_baseline_correction_fraction_phie"] = v
        params["potential_baseline_correction_fraction_phis_c"] = v


def _apply_aging_cfg(params: Dict[str, Any], inpt: Dict[str, str]) -> None:
    """Forward ModelFin_110 aging-fix1 settings into params."""
    cfg = parse_aging_fix1_config({**params, **dict(inpt)})
    params.update(cfg.to_dict())
    params["aging_fix1_config"] = cfg.to_dict()
    # Keep uppercase keys visible for existing code paths and config.json.
    params["USE_ASSB_AGING_FIX1"] = bool(cfg.use_assb_aging_fix1)
    params["AGING_STAGE"] = str(cfg.stage)
    params["FREEZE_107A_CORE"] = bool(cfg.freeze_107a_core)
    params["LOAD_AGING_MODEL"] = str(cfg.load_aging_model)
    params["AGING_CYCLE_TABLE_CSV"] = str(cfg.cycle_table_csv)
    params["AGING_CAPACITY_TARGET_CSV"] = str(cfg.capacity_target_csv)
    params["USE_ASSB_AGING_INJECTION_CBAR"] = bool(cfg.use_injection_cbar)
    params["USE_ASSB_AGING_INJECTION_FLUX"] = bool(cfg.use_injection_flux)
    params["USE_ASSB_AGING_INJECTION_THETA_WINDOW"] = bool(cfg.use_injection_theta_window)
    params["USE_ASSB_AGING_INJECTION_ROHM"] = bool(cfg.use_injection_rohm)
    params["LOCK_COMMON_MODE_GAUGE"] = bool(cfg.lock_common_mode_gauge)
    # Compatibility aliases consumed by the Stage-C files.
    params["USE_ASSB_AGING_MECHANISM"] = bool(cfg.use_assb_aging_fix1)
    params["aging_cycle_table_csv"] = str(cfg.cycle_table_csv)
    params["capacity_target_csv"] = str(cfg.capacity_target_csv)
    params["CAPACITY_TARGET_CSV"] = str(cfg.capacity_target_csv)
    params["ASSB_AGING_CYCLE_TABLE"] = str(cfg.cycle_table_csv)
    params["DATA_LOSS"] = False
    params["data_loss"] = False
    params["ALPHA_DATA"] = 0.0
    params["MAX_BATCH_SIZE_DATA"] = 0


def _attach_soft_label_solution(params: Dict[str, Any], inpt_or_params: Dict[str, Any]) -> None:
    soft_dir = _normalize_path_str(inpt_or_params.get("SOFT_LABEL_DIR") or inpt_or_params.get("soft_label_dir") or os.environ.get("ASSB_SOFT_LABEL_DIR"))
    if soft_dir is None:
        return
    soft_path = Path(soft_dir)
    if not soft_path.is_absolute():
        soft_path = Path.cwd() / soft_path
    solution = soft_path / "solution.npz"
    if not solution.exists():
        params["soft_label_dir_runtime"] = str(soft_path)
        params["soft_label_solution_missing"] = str(solution)
        return
    try:
        with np.load(solution, allow_pickle=True) as z:
            names = set(z.files)
            t_key = "t_global_s" if "t_global_s" in names else ("t" if "t" in names else "time_s")
            i_key = "I_profile" if "I_profile" in names else ("current_A" if "current_A" in names else "I")
            t = np.asarray(z[t_key], dtype=np.float64).reshape(-1)
            I = np.asarray(z[i_key], dtype=np.float64).reshape(-1)
            order = np.argsort(t)
            t = t[order]
            I = I[order]
            params["current_profile"] = (t, I)
            params["time_profile"] = t
            params["current_profile_A"] = I
            params["tmax"] = np.float64(float(np.nanmax(t)) if t.size else params.get("tmax", 1.0))
            params["rescale_T"] = np.float64(params["tmax"])
            params["soft_label_dir_runtime"] = str(soft_path)
            params["soft_label_solution_runtime"] = str(solution)
            if "cycle_id" in names:
                params["cycle_id_profile"] = np.asarray(z["cycle_id"], dtype=np.int64).reshape(-1)[order]
            for key, pkey in (("cs_a", "cs_a0"), ("cs_c", "cs_c0")):
                if key in names:
                    arr = np.asarray(z[key], dtype=np.float64)
                    if arr.ndim >= 2:
                        params[pkey] = np.float64(np.mean(arr[0]))
                    elif arr.size:
                        params[pkey] = np.float64(arr[0])
            if "theta_c" in names:
                arr = np.asarray(z["theta_c"], dtype=np.float64)
                try:
                    params.setdefault("theta_c0", np.float64(np.nanmean(arr[0])))
                except Exception:
                    pass
            for key, pkey in (("phie", "phie0"), ("phis_c", "phis_c0")):
                if key in names:
                    arr = np.asarray(z[key], dtype=np.float64).reshape(-1)
                    if arr.size:
                        params[pkey] = np.float64(arr[0])
    except Exception as exc:
        params["soft_label_attach_error"] = str(exc)


def _candidate_state_dicts(payload: Any):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            val = payload.get(key)
            if isinstance(val, dict):
                yield key, val
        if all(isinstance(k, str) for k in payload.keys()):
            yield "raw", payload
    elif hasattr(payload, "state_dict"):
        yield "module", payload.state_dict()


def _strip_prefix_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(state.keys())
    for prefix in ("model.", "module."):
        if keys and sum(k.startswith(prefix) for k in keys) > max(1, len(keys) // 2):
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state


def safe_load(nn: myNN, weight_path: str) -> myNN:
    """Load the 107A core with an explicit key report.

    The load is never silent: missing/unexpected keys are written to
    ``Model/load_report.json`` and critical core-key misses raise immediately.
    Extra wrapper keys are reported but do not stop the Stage-C smoke run.
    """
    if not weight_path:
        return nn
    candidates = [str(weight_path)]
    if os.path.isdir(str(weight_path)):
        candidates.extend([os.path.join(str(weight_path), "best.pt"), os.path.join(str(weight_path), "last.pt")])
    if str(weight_path).endswith(".weights.h5"):
        candidates.append(str(weight_path).replace(".weights.h5", ".pt"))
        candidates.append(str(weight_path).replace("best.weights.h5", "best.pt"))
    seen = set()
    last_exc = None
    for cand in candidates:
        if not cand or cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        try:
            payload = torch.load(cand, map_location=nn.device)
            for name, state in _candidate_state_dicts(payload):
                state = _strip_prefix_if_needed(state)
                result = nn.model.load_state_dict(state, strict=False)
                missing = list(result.missing_keys)
                unexpected = list(result.unexpected_keys)
                critical_missing = [k for k in missing if any(s in k for s in ("base_t", "base_tr", "gp_", "out_"))]
                report = {
                    "source": cand,
                    "payload_name": name,
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                    "critical_missing": critical_missing,
                    "n_loaded_candidate_keys": len(state),
                }
                Path(nn.modelFolder).mkdir(parents=True, exist_ok=True)
                (Path(nn.modelFolder) / "load_report.json").write_text(__import__("json").dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[ASSB-110] loaded prior candidate {cand} ({name}); missing={len(missing)}, unexpected={len(unexpected)}")
                if critical_missing:
                    raise RuntimeError("Critical 107A core keys missing: " + ", ".join(critical_missing[:20]))
                return nn
        except Exception as exc:
            last_exc = exc
            print(f"[ASSB-110] skip incompatible prior weights {cand}: {exc}")
    if last_exc is not None:
        raise SystemExit(f"ERROR: could not load prior model {weight_path}: {last_exc}")
    raise SystemExit(f"ERROR: prior model path not found: {weight_path}")


def initialize_params_from_inpt(inpt: Dict[str, str]) -> Dict[str, Any]:
    if _get_bool(inpt, "DATA_LOSS", False) or _get_float(inpt, "ALPHA_DATA", 0.0) != 0.0 or _get_int(inpt, "MAX_BATCH_SIZE_DATA", 0) != 0:
        raise SystemExit("ERROR: ModelFin_110 aging-fix1 keeps original DATA_LOSS closed. Set DATA_LOSS=False, ALPHA_DATA=0, MAX_BATCH_SIZE_DATA=0.")
    alpha = _parse_alpha(inpt.get("alpha"))
    params: Dict[str, Any] = {
        "seed": _get_int(inpt, "seed", -1),
        "ID": _get_int(inpt, "ID", 110),
        "EPOCHS": _get_int(inpt, "EPOCHS", 300),
        "EPOCHS_LBFGS": _get_int(inpt, "EPOCHS_LBFGS", 0),
        "EPOCHS_START_LBFGS": _get_int(inpt, "EPOCHS_START_LBFGS", 0),
        "alpha": alpha,
        "LEARNING_RATE_WEIGHTS": _get_float(inpt, "LEARNING_RATE_WEIGHTS", 1.0e-2),
        "LEARNING_RATE_WEIGHTS_FINAL": _get_float(inpt, "LEARNING_RATE_WEIGHTS_FINAL", 1.0e-3),
        "LEARNING_RATE_MODEL": _get_float(inpt, "LEARNING_RATE_MODEL", 5.0e-4),
        "LEARNING_RATE_MODEL_FINAL": _get_float(inpt, "LEARNING_RATE_MODEL_FINAL", 5.0e-5),
        "LEARNING_RATE_LBFGS": _get_float(inpt, "LEARNING_RATE_LBFGS", 1.0),
        "GRADIENT_THRESHOLD": _get_float(inpt, "GRADIENT_THRESHOLD", 10.0) if "GRADIENT_THRESHOLD" in inpt else 10.0,
        "HARD_IC_TIMESCALE": _get_float(inpt, "HARD_IC_TIMESCALE", 1.0),
        "RATIO_FIRST_TIME": _get_float(inpt, "RATIO_FIRST_TIME", 1.0),
        "RATIO_T_MIN": _get_float(inpt, "RATIO_T_MIN", 0.0),
        "EXP_LIMITER": _get_float(inpt, "EXP_LIMITER", 10.0),
        "COLLOCATION_MODE": _get_str(inpt, "COLLOCATION_MODE", "fixed"),
        "GRADUAL_TIME_SGD": _get_bool(inpt, "GRADUAL_TIME_SGD", False),
        "GRADUAL_TIME_LBFGS": _get_bool(inpt, "GRADUAL_TIME_LBFGS", False),
        "N_GRADUAL_STEPS_LBFGS": _get_int(inpt, "N_GRADUAL_STEPS_LBFGS", 10),
        "GRADUAL_TIME_MODE_LBFGS": _get_str(inpt, "GRADUAL_TIME_MODE_LBFGS", "exponential"),
        "DYNAMIC_ATTENTION_WEIGHTS": _get_bool(inpt, "DYNAMIC_ATTENTION_WEIGHTS", False),
        "ANNEALING_WEIGHTS": _get_bool(inpt, "ANNEALING_WEIGHTS", False),
        "USE_LOSS_THRESHOLD": _get_bool(inpt, "USE_LOSS_THRESHOLD", False),
        "LOSS_THRESHOLD": _get_float(inpt, "LOSS_THRESHOLD", 2000.0),
        "INNER_EPOCHS": _get_int(inpt, "INNER_EPOCHS", 1),
        "START_WEIGHT_TRAINING_EPOCH": _get_int(inpt, "START_WEIGHT_TRAINING_EPOCH", 50),
        "ACTIVATION": _get_str(inpt, "ACTIVATION", "tanh"),
        "LBFGS": _get_bool(inpt, "LBFGS", False),
        "SGD": _get_bool(inpt, "SGD", True),
        "MERGED": _get_bool(inpt, "MERGED", True),
        "LINEARIZE_J": _get_bool(inpt, "LINEARIZE_J", True),
        "BATCH_SIZE_INT": _get_int(inpt, "BATCH_SIZE_INT", 512),
        "BATCH_SIZE_BOUND": _get_int(inpt, "BATCH_SIZE_BOUND", 512),
        "MAX_BATCH_SIZE_DATA": 0,
        "BATCH_SIZE_DATA": 0,
        "BATCH_SIZE_REG": _get_int(inpt, "BATCH_SIZE_REG", 256),
        "BATCH_SIZE_STRUCT": _get_int(inpt, "BATCH_SIZE_STRUCT", 0),
        "N_BATCH": _get_int(inpt, "N_BATCH", 8),
        "N_BATCH_LBFGS": _get_int(inpt, "N_BATCH_LBFGS", 1),
        "NEURONS_NUM": _get_int(inpt, "NEURONS_NUM", 20),
        "LAYERS_T_NUM": _get_int(inpt, "LAYERS_T_NUM", 1),
        "LAYERS_TR_NUM": _get_int(inpt, "LAYERS_TR_NUM", 1),
        "NUM_GRAD_PATH_LAYERS": _get_int(inpt, "NUM_GRAD_PATH_LAYERS", 3),
        "NUM_GRAD_PATH_UNITS": _get_int(inpt, "NUM_GRAD_PATH_UNITS", 20),
        "NUM_RES_BLOCKS": _get_int(inpt, "NUM_RES_BLOCKS", 0),
        "NUM_RES_BLOCK_LAYERS": _get_int(inpt, "NUM_RES_BLOCK_LAYERS", 1),
        "NUM_RES_BLOCK_UNITS": _get_int(inpt, "NUM_RES_BLOCK_UNITS", 20),
        "LOCAL_utilFolder": _normalize_path_str(inpt.get("LOCAL_utilFolder", str(_THIS_DIR))),
        "HNN_utilFolder": _normalize_path_str(inpt.get("HNN_utilFolder")),
        "HNN_modelFolder": _normalize_path_str(inpt.get("HNN_modelFolder")),
        "HNN_params": None,
        "HNNTIME_utilFolder": _normalize_path_str(inpt.get("HNNTIME_utilFolder")),
        "HNNTIME_modelFolder": _normalize_path_str(inpt.get("HNNTIME_modelFolder")),
        "HNNTIME_val": None,
        "LOAD_MODEL": _normalize_path_str(inpt.get("LOAD_MODEL")),
        "PRIOR_MODEL": _get_str(inpt, "PRIOR_MODEL", "assb_discharge").lower(),
        "SOFT_LABEL_DIR": _normalize_path_str(inpt.get("SOFT_LABEL_DIR")),
        "CAPACITY_TARGET_CSV": _normalize_path_str(inpt.get("AGING_CAPACITY_TARGET_CSV") or inpt.get("CAPACITY_TARGET_CSV")),
        "ASSB_AGING_CYCLE_TABLE": _normalize_path_str(inpt.get("AGING_CYCLE_TABLE_CSV") or inpt.get("ASSB_AGING_CYCLE_TABLE")),
        "DATA_LOSS": False,
        "ALPHA_DATA": 0.0,
        "weights": {
            "phie_int": _get_float(inpt, "w_phie_int", 1.0),
            "phis_c_int": _get_float(inpt, "w_phis_c_int", 1.0),
            "cs_a_int": _get_float(inpt, "w_cs_a_int", 10.0),
            "cs_c_int": _get_float(inpt, "w_cs_c_int", 2.0),
            "cs_a_rmin_bound": _get_float(inpt, "w_cs_a_rmin_bound", 1.0),
            "cs_c_rmin_bound": _get_float(inpt, "w_cs_c_rmin_bound", 1.0),
            "cs_a_rmax_bound": _get_float(inpt, "w_cs_a_rmax_bound", 500.0),
            "cs_c_rmax_bound": _get_float(inpt, "w_cs_c_rmax_bound", 1500.0),
            "phie_dat": 0.0,
            "phis_c_dat": 0.0,
            "cs_a_dat": 0.0,
            "cs_c_dat": 0.0,
        },
    }
    # Copy selected optional fields and translate booleans/numbers below.
    for key, value in inpt.items():
        if key not in params:
            params[key] = value
    _apply_cbar_baseline_params(params, inpt)
    _apply_aging_cfg(params, inpt)
    params["alpha"] = alpha
    params["alpha"][2] = 0.0
    params["DATA_LOSS"] = False
    params["MAX_BATCH_SIZE_DATA"] = 0
    return params


def initialize_params(args) -> Dict[str, Any]:
    inpt = parse_input_file(args.input_file)
    return initialize_params_from_inpt(inpt)


def _choose_param_builder(args, prior_model: str = "assb_discharge"):
    prior_model = str(prior_model or "assb_discharge").strip().lower()
    if getattr(args, "simpleModel", False):
        try:
            from .spm_simpler import makeParams
        except Exception:  # pragma: no cover
            from spm_simpler import makeParams
    elif prior_model in {"assb_discharge", "spm_assb_train_discharge", "assb_train_discharge"}:
        try:
            from .spm_assb_train_discharge import makeParams
        except Exception:  # pragma: no cover
            from spm_assb_train_discharge import makeParams
    else:
        try:
            from .spm import makeParams
        except Exception:  # pragma: no cover
            from spm import makeParams
    return makeParams


def initialize_nn(args, input_params: Dict[str, Any]) -> myNN:
    # Avoid old summary env pollution, which caused long-sequence path mistakes in D3.
    os.environ.pop("ASSB_SOFT_LABEL_SUMMARY", None)
    seed = int(input_params.get("seed", -1))
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    makeParams = _choose_param_builder(args, input_params.get("PRIOR_MODEL", "assb_discharge"))
    params = makeParams()
    _apply_cbar_baseline_params(params, input_params)
    _apply_aging_cfg(params, {k: str(v) for k, v in input_params.items()})
    _attach_soft_label_solution(params, input_params)

    # Forward input/config values into params for config.json visibility.
    for key, val in input_params.items():
        if key != "weights":
            params[key] = val
    params["alpha"] = list(input_params.get("alpha", [1.0, 1.0, 0.0, 1.0]))
    params["alpha"][2] = 0.0
    params["DATA_LOSS"] = False
    params["MAX_BATCH_SIZE_DATA"] = 0

    # Conservative ASSB defaults if makeParams did not define every key.
    params.setdefault("F", np.float64(96485.33212))
    params.setdefault("rescale_T", np.float64(params.get("tmax", 1.0)))
    params.setdefault("rescale_phie", np.float64(1.0))
    params.setdefault("rescale_phis_c", np.float64(1.0))
    params.setdefault("Rs_a", np.float64(50e-6))
    params.setdefault("Rs_c", np.float64(1.8e-6))
    params.setdefault("eps_s_a", np.float64(0.95))
    params.setdefault("eps_s_c", np.float64(0.55))
    params.setdefault("A_a", np.float64(np.pi * (5e-3) ** 2))
    params.setdefault("A_c", np.float64(np.pi * (5e-3) ** 2))
    params.setdefault("L_a", np.float64(100e-6))
    params.setdefault("L_c", np.float64(16e-6))
    params.setdefault("V_a", np.float64(params["A_a"] * params["L_a"]))
    params.setdefault("V_c", np.float64(params["A_c"] * params["L_c"]))
    params.setdefault("cs_a0", np.float64(params.get("csa0", 0.0)))
    params.setdefault("cs_c0", np.float64(params.get("csc0", 0.0)))
    params.setdefault("csanmax", np.float64(max(float(params.get("cs_a0", 0.0)), 6.0)))
    params.setdefault("cscamax", np.float64(max(float(params.get("cs_c0", 0.0)), 51.8)))
    params.setdefault("Ds_a", np.float64(1.0e-13))
    params.setdefault("Ds_c", np.float64(1.0e-14))
    params.setdefault("R_ohm_eff", np.float64(float(params.get("AGING_R_OHM0", 105.0))))
    params.setdefault("theta_c_bottom", np.float64(0.834))
    params.setdefault("theta_c_top", np.float64(0.432))
    params.setdefault("theta_c_window_mid0", np.float64(0.5 * (float(params["theta_c_bottom"]) + float(params["theta_c_top"]))))
    params.setdefault("deg_i0_a_min_eff", np.float64(1.0))
    params.setdefault("deg_i0_a_max_eff", np.float64(1.0))
    params.setdefault("deg_ds_c_min_eff", np.float64(1.0))
    params.setdefault("deg_ds_c_max_eff", np.float64(1.0))

    neurons = int(input_params.get("NEURONS_NUM", 20))
    hidden_t = [neurons] * max(int(input_params.get("LAYERS_T_NUM", 1)), 1)
    hidden_tr = [neurons] * max(int(input_params.get("LAYERS_TR_NUM", 1)), 1)
    nn = myNN(
        params,
        hidden_units_t=hidden_t,
        hidden_units_t_r=hidden_tr,
        hidden_units_phie=[neurons],
        hidden_units_phis_c=[neurons],
        hidden_units_cs_a=[neurons],
        hidden_units_cs_c=[neurons],
        n_hidden_res_blocks=int(input_params.get("NUM_RES_BLOCKS", 0)),
        n_res_block_layers=int(input_params.get("NUM_RES_BLOCK_LAYERS", 1)),
        n_res_block_units=int(input_params.get("NUM_RES_BLOCK_UNITS", 20)),
        n_grad_path_layers=input_params.get("NUM_GRAD_PATH_LAYERS", 3),
        n_grad_path_units=input_params.get("NUM_GRAD_PATH_UNITS", 20),
        alpha=params["alpha"],
        batch_size_int=int(input_params.get("BATCH_SIZE_INT", 0)),
        batch_size_bound=int(input_params.get("BATCH_SIZE_BOUND", 0)),
        max_batch_size_data=0,
        batch_size_reg=int(input_params.get("BATCH_SIZE_REG", 0)),
        batch_size_struct=int(input_params.get("BATCH_SIZE_STRUCT", 0)),
        n_batch=int(input_params.get("N_BATCH", 1)),
        n_batch_lbfgs=int(input_params.get("N_BATCH_LBFGS", 1)),
        nEpochs_start_lbfgs=int(input_params.get("EPOCHS_START_LBFGS", 0)),
        hard_IC_timescale=np.float64(input_params.get("HARD_IC_TIMESCALE", 1.0)),
        exponentialLimiter=np.float64(input_params.get("EXP_LIMITER", 10.0)),
        collocationMode=input_params.get("COLLOCATION_MODE", "fixed"),
        gradualTime_sgd=bool(input_params.get("GRADUAL_TIME_SGD", False)),
        gradualTime_lbfgs=bool(input_params.get("GRADUAL_TIME_LBFGS", False)),
        firstTime=np.float64(input_params.get("RATIO_FIRST_TIME", 1.0)) * np.float64(params.get("rescale_T", 1.0)),
        n_gradual_steps_lbfgs=input_params.get("N_GRADUAL_STEPS_LBFGS"),
        gradualTimeMode_lbfgs=input_params.get("GRADUAL_TIME_MODE_LBFGS"),
        tmin_int_bound=np.float64(input_params.get("RATIO_T_MIN", 0.0)) * np.float64(params.get("rescale_T", 1.0)),
        nEpochs=int(input_params.get("EPOCHS", 300)),
        nEpochs_lbfgs=int(input_params.get("EPOCHS_LBFGS", 0)),
        initialLossThreshold=np.float64(input_params.get("LOSS_THRESHOLD", 2000.0)),
        dynamicAttentionWeights=bool(input_params.get("DYNAMIC_ATTENTION_WEIGHTS", False)),
        annealingWeights=bool(input_params.get("ANNEALING_WEIGHTS", False)),
        useLossThreshold=bool(input_params.get("USE_LOSS_THRESHOLD", False)),
        activation=input_params.get("ACTIVATION", "tanh"),
        linearizeJ=bool(input_params.get("LINEARIZE_J", True)),
        lbfgs=bool(input_params.get("LBFGS", False)),
        sgd=bool(input_params.get("SGD", True)),
        params_max=[params.get("deg_i0_a_max_eff", 1.0), params.get("deg_ds_c_max_eff", 1.0)],
        params_min=[params.get("deg_i0_a_min_eff", 1.0), params.get("deg_ds_c_min_eff", 1.0)],
        xDataList=[],
        x_params_dataList=[],
        yDataList=[],
        logLossFolder="Log",
        modelFolder="Model",
        local_utilFolder=input_params.get("LOCAL_utilFolder"),
        hnn_utilFolder=input_params.get("HNN_utilFolder"),
        hnn_modelFolder=input_params.get("HNN_modelFolder"),
        hnn_params=input_params.get("HNN_params"),
        hnntime_utilFolder=input_params.get("HNNTIME_utilFolder"),
        hnntime_modelFolder=input_params.get("HNNTIME_modelFolder"),
        hnntime_val=input_params.get("HNNTIME_val"),
        verbose=False,
        weights=input_params.get("weights"),
    )
    load_model = input_params.get("LOAD_MODEL")
    if load_model:
        nn = safe_load(nn, load_model)
    return nn


__all__ = ["initialize_params", "initialize_params_from_inpt", "initialize_nn", "safe_load", "absolute_path_check"]
