from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
from typing import Any, Dict, Optional

import numpy as np
import torch

# Original project uses path-based imports. Keep that compatibility.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from . import argument
    from .myNN import myNN
except ImportError:  # pragma: no cover
    import argument
    from myNN import myNN

from prettyPlot.parser import parse_input_file


def _normalize_path_str(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    val = str(path).strip()
    if val.upper() in {"NONE", "NULL", ""}:
        return None
    return val


def absolute_path_check(path: Optional[str]) -> None:
    path = _normalize_path_str(path)
    if path is None:
        return
    if not os.path.isabs(path):
        raise SystemExit(f"ERROR: {path} is not absolute")


def safe_load(nn: myNN, weight_path: str) -> myNN:
    """
    Compatibility loader.

    The TensorFlow project used `best.weights.h5`. This PyTorch port writes `.pt`
    files, but we accept multiple candidate names so existing scripts stay usable.
    """
    candidates = []
    weight_path = str(weight_path)
    candidates.append(weight_path)
    # Common translated filenames.
    if weight_path.endswith(".weights.h5"):
        candidates.append(weight_path.replace(".weights.h5", ".pt"))
        candidates.append(weight_path.replace(".weights.h5", ".pth"))
    if weight_path.endswith("best.weights.h5"):
        candidates.append(weight_path.replace("best.weights.h5", "best.pt"))
    if os.path.isdir(weight_path):
        candidates.extend(
            [
                os.path.join(weight_path, "best.pt"),
                os.path.join(weight_path, "last.pt"),
                os.path.join(weight_path, "best.weights.h5"),
            ]
        )
    seen = set()
    for cand in candidates:
        if cand in seen or cand is None:
            continue
        seen.add(cand)
        if not os.path.exists(cand):
            continue
        try:
            state = torch.load(cand, map_location=nn.device)
            if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
                nn.model.load_state_dict(state)
            else:
                # full module checkpoint fallback
                nn.model = state.to(nn.device)
            return nn
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            continue
    raise SystemExit(f"ERROR: could not load {weight_path}")


_DEF_BOOL_KEYS = {
    "DYNAMIC_ATTENTION_WEIGHTS",
    "ANNEALING_WEIGHTS",
    "USE_LOSS_THRESHOLD",
    "LBFGS",
    "SGD",
    "MERGED",
    "LINEARIZE_J",
    "GRADUAL_TIME_SGD",
    "GRADUAL_TIME_LBFGS",
}


def _get_bool(inpt: Dict[str, str], key: str, default: bool = False) -> bool:
    if key not in inpt:
        return bool(default)
    return str(inpt[key]).strip().lower() == "true"


def _get_int(inpt: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(inpt[key])
    except KeyError:
        return default


def _get_float(inpt: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(inpt[key])
    except KeyError:
        return float(default)


def initialize_params_from_inpt(inpt: Dict[str, str]) -> Dict[str, Any]:
    seed = _get_int(inpt, "seed", -1)
    ID = _get_int(inpt, "ID", 0)
    EPOCHS = int(inpt["EPOCHS"])
    EPOCHS_LBFGS = int(inpt["EPOCHS_LBFGS"])
    EPOCHS_START_LBFGS = _get_int(inpt, "EPOCHS_START_LBFGS", 20)

    alpha = [float(entry) for entry in inpt["alpha"].split()]
    LEARNING_RATE_WEIGHTS = float(inpt["LEARNING_RATE_WEIGHTS"])
    LEARNING_RATE_WEIGHTS_FINAL = float(inpt["LEARNING_RATE_WEIGHTS_FINAL"])
    LEARNING_RATE_MODEL = float(inpt["LEARNING_RATE_MODEL"])
    LEARNING_RATE_MODEL_FINAL = float(inpt["LEARNING_RATE_MODEL_FINAL"])
    LEARNING_RATE_LBFGS = float(inpt["LEARNING_RATE_LBFGS"])
    GRADIENT_THRESHOLD = float(inpt["GRADIENT_THRESHOLD"]) if "GRADIENT_THRESHOLD" in inpt else None

    HARD_IC_TIMESCALE = float(inpt["HARD_IC_TIMESCALE"])
    RATIO_FIRST_TIME = float(inpt["RATIO_FIRST_TIME"])
    RATIO_T_MIN = float(inpt["RATIO_T_MIN"])
    EXP_LIMITER = float(inpt["EXP_LIMITER"])
    COLLOCATION_MODE = inpt["COLLOCATION_MODE"]
    GRADUAL_TIME_SGD = _get_bool(inpt, "GRADUAL_TIME_SGD")
    GRADUAL_TIME_LBFGS = _get_bool(inpt, "GRADUAL_TIME_LBFGS")
    FIRST_TIME_LBFGS = None
    N_GRADUAL_STEPS_LBFGS = None
    GRADUAL_TIME_MODE_LBFGS = None
    if GRADUAL_TIME_LBFGS:
        N_GRADUAL_STEPS_LBFGS = int(inpt["N_GRADUAL_STEPS_LBFGS"])
        GRADUAL_TIME_MODE_LBFGS = inpt.get("GRADUAL_TIME_MODE_LBFGS", "linear")

    DYNAMIC_ATTENTION_WEIGHTS = _get_bool(inpt, "DYNAMIC_ATTENTION_WEIGHTS")
    ANNEALING_WEIGHTS = _get_bool(inpt, "ANNEALING_WEIGHTS")
    USE_LOSS_THRESHOLD = _get_bool(inpt, "USE_LOSS_THRESHOLD")
    LOSS_THRESHOLD = _get_float(inpt, "LOSS_THRESHOLD", 1000.0)
    INNER_EPOCHS = _get_int(inpt, "INNER_EPOCHS", 1)
    START_WEIGHT_TRAINING_EPOCH = _get_int(inpt, "START_WEIGHT_TRAINING_EPOCH", 1)

    LOCAL_utilFolder = _normalize_path_str(inpt.get("LOCAL_utilFolder", str(_THIS_DIR)))
    HNN_utilFolder = _normalize_path_str(inpt.get("HNN_utilFolder"))
    HNN_modelFolder = _normalize_path_str(inpt.get("HNN_modelFolder"))
    try:
        HNN_params = [np.float64(entry) for entry in inpt["HNN_params"].split()]
    except KeyError:
        HNN_params = None
    if (
        HNN_utilFolder is None
        or HNN_modelFolder is None
        or not os.path.isdir(HNN_utilFolder)
        or not os.path.isdir(HNN_modelFolder)
    ):
        HNN_utilFolder = None
        HNN_modelFolder = None
        HNN_params = None

    HNNTIME_utilFolder = _normalize_path_str(inpt.get("HNNTIME_utilFolder"))
    HNNTIME_modelFolder = _normalize_path_str(inpt.get("HNNTIME_modelFolder"))
    try:
        HNNTIME_val = np.float64(inpt.get("HNNTIME_val", inpt.get("HNNTIME_val", inpt.get("HNNTIME_val ", "None"))))
    except Exception:
        HNNTIME_val = None
    if (
        HNNTIME_utilFolder is None
        or HNNTIME_modelFolder is None
        or HNNTIME_val is None
        or not os.path.isdir(HNNTIME_utilFolder)
        or not os.path.isdir(HNNTIME_modelFolder)
    ):
        HNNTIME_utilFolder = None
        HNNTIME_modelFolder = None
        HNNTIME_val = None

    if (HNN_utilFolder is not None) or (HNNTIME_utilFolder is not None):
        if not os.path.isdir(LOCAL_utilFolder):
            print(f"ERROR: {LOCAL_utilFolder} is not a directory")
            sys.exit()
    absolute_path_check(LOCAL_utilFolder)
    absolute_path_check(HNN_utilFolder)
    absolute_path_check(HNN_modelFolder)
    absolute_path_check(HNNTIME_utilFolder)
    absolute_path_check(HNNTIME_modelFolder)

    ACTIVATION = inpt["ACTIVATION"]
    LBFGS = _get_bool(inpt, "LBFGS")
    SGD = _get_bool(inpt, "SGD")
    MERGED = _get_bool(inpt, "MERGED")
    LINEARIZE_J = _get_bool(inpt, "LINEARIZE_J")

    try:
        weights = {
            "phie_int": np.float64(inpt["w_phie_int"]),
            "phis_c_int": np.float64(inpt["w_phis_c_int"]),
            "cs_a_int": np.float64(inpt["w_cs_a_int"]),
            "cs_c_int": np.float64(inpt["w_cs_c_int"]),
            "cs_a_rmin_bound": np.float64(inpt["w_cs_a_rmin_bound"]),
            "cs_a_rmax_bound": np.float64(inpt["w_cs_a_rmax_bound"]),
            "cs_c_rmin_bound": np.float64(inpt["w_cs_c_rmin_bound"]),
            "cs_c_rmax_bound": np.float64(inpt["w_cs_c_rmax_bound"]),
            "phie_dat": np.float64(inpt["w_phie_dat"]),
            "phis_c_dat": np.float64(inpt["w_phis_c_dat"]),
            "cs_a_dat": np.float64(inpt["w_cs_a_dat"]),
            "cs_c_dat": np.float64(inpt["w_cs_c_dat"]),
        }
    except KeyError:
        weights = None

    BATCH_SIZE_INT = int(inpt["BATCH_SIZE_INT"])
    BATCH_SIZE_BOUND = int(inpt["BATCH_SIZE_BOUND"])
    MAX_BATCH_SIZE_DATA = int(inpt["MAX_BATCH_SIZE_DATA"])
    BATCH_SIZE_REG = int(inpt["BATCH_SIZE_REG"])
    BATCH_SIZE_STRUCT = _get_int(inpt, "BATCH_SIZE_STRUCT", 64)

    # ASSB secondary-conservation controls. These keys are optional in the
    # original PINNSTRIPES inputs, so read them defensively. They are forwarded
    # into params and config.json in initialize_nn().
    w_cs_a_mass_reg = _get_float(inpt, "w_cs_a_mass_reg", float(os.environ.get("ASSB_W_CS_A_MASS_REG", 1.0)))
    w_cs_c_mass_reg = _get_float(inpt, "w_cs_c_mass_reg", float(os.environ.get("ASSB_W_CS_C_MASS_REG", 1.0)))
    mass_reg_n_quad = _get_int(inpt, "mass_reg_n_quad", _get_int(inpt, "N_MASS_REG_QUAD", 8))
    N_BATCH = int(inpt["N_BATCH"])
    N_BATCH_LBFGS = int(inpt["N_BATCH_LBFGS"])
    NEURONS_NUM = int(inpt["NEURONS_NUM"])
    LAYERS_T_NUM = int(inpt["LAYERS_T_NUM"])
    LAYERS_TR_NUM = int(inpt["LAYERS_TR_NUM"])
    LAYERS_T_VAR_NUM = int(inpt["LAYERS_T_VAR_NUM"])
    LAYERS_TR_VAR_NUM = int(inpt["LAYERS_TR_VAR_NUM"])
    LAYERS_SPLIT_NUM = int(inpt["LAYERS_SPLIT_NUM"])
    NUM_RES_BLOCKS = _get_int(inpt, "NUM_RES_BLOCKS", 0)
    if NUM_RES_BLOCKS > 0:
        NUM_RES_BLOCK_LAYERS = int(inpt["NUM_RES_BLOCK_LAYERS"])
        NUM_RES_BLOCK_UNITS = int(inpt["NUM_RES_BLOCK_UNITS"])
    else:
        NUM_RES_BLOCK_LAYERS = 1
        NUM_RES_BLOCK_UNITS = 1
    try:
        NUM_GRAD_PATH_LAYERS = int(inpt["NUM_GRAD_PATH_LAYERS"])
    except Exception:
        NUM_GRAD_PATH_LAYERS = None
    if NUM_GRAD_PATH_LAYERS is not None and NUM_GRAD_PATH_LAYERS > 0:
        NUM_GRAD_PATH_UNITS = int(inpt["NUM_GRAD_PATH_UNITS"])
    else:
        NUM_GRAD_PATH_UNITS = None

    LOAD_MODEL = _normalize_path_str(inpt.get("LOAD_MODEL"))
    if LOAD_MODEL is not None and not os.path.isfile(LOAD_MODEL):
        LOAD_MODEL = None

    PRIOR_MODEL = str(inpt.get("PRIOR_MODEL", "spm")).strip().lower()

    return {
        "MERGED": MERGED,
        "NEURONS_NUM": NEURONS_NUM,
        "LAYERS_T_NUM": LAYERS_T_NUM,
        "LAYERS_TR_NUM": LAYERS_TR_NUM,
        "LAYERS_T_VAR_NUM": LAYERS_T_VAR_NUM,
        "LAYERS_TR_VAR_NUM": LAYERS_TR_VAR_NUM,
        "LAYERS_SPLIT_NUM": LAYERS_SPLIT_NUM,
        "seed": seed,
        "ID": ID,
        "EPOCHS": EPOCHS,
        "EPOCHS_LBFGS": EPOCHS_LBFGS,
        "EPOCHS_START_LBFGS": EPOCHS_START_LBFGS,
        "alpha": alpha,
        "LEARNING_RATE_WEIGHTS": LEARNING_RATE_WEIGHTS,
        "LEARNING_RATE_WEIGHTS_FINAL": LEARNING_RATE_WEIGHTS_FINAL,
        "LEARNING_RATE_MODEL": LEARNING_RATE_MODEL,
        "LEARNING_RATE_MODEL_FINAL": LEARNING_RATE_MODEL_FINAL,
        "LEARNING_RATE_LBFGS": LEARNING_RATE_LBFGS,
        "GRADIENT_THRESHOLD": GRADIENT_THRESHOLD,
        "HARD_IC_TIMESCALE": HARD_IC_TIMESCALE,
        "RATIO_FIRST_TIME": RATIO_FIRST_TIME,
        "RATIO_T_MIN": RATIO_T_MIN,
        "EXP_LIMITER": EXP_LIMITER,
        "COLLOCATION_MODE": COLLOCATION_MODE,
        "GRADUAL_TIME_SGD": GRADUAL_TIME_SGD,
        "GRADUAL_TIME_LBFGS": GRADUAL_TIME_LBFGS,
        "FIRST_TIME_LBFGS": FIRST_TIME_LBFGS,
        "N_GRADUAL_STEPS_LBFGS": N_GRADUAL_STEPS_LBFGS,
        "GRADUAL_TIME_MODE_LBFGS": GRADUAL_TIME_MODE_LBFGS,
        "DYNAMIC_ATTENTION_WEIGHTS": DYNAMIC_ATTENTION_WEIGHTS,
        "ANNEALING_WEIGHTS": ANNEALING_WEIGHTS,
        "USE_LOSS_THRESHOLD": USE_LOSS_THRESHOLD,
        "LOSS_THRESHOLD": LOSS_THRESHOLD,
        "INNER_EPOCHS": INNER_EPOCHS,
        "START_WEIGHT_TRAINING_EPOCH": START_WEIGHT_TRAINING_EPOCH,
        "HNN_utilFolder": HNN_utilFolder,
        "HNN_modelFolder": HNN_modelFolder,
        "HNN_params": HNN_params,
        "HNNTIME_utilFolder": HNNTIME_utilFolder,
        "HNNTIME_modelFolder": HNNTIME_modelFolder,
        "HNNTIME_val": HNNTIME_val,
        "ACTIVATION": ACTIVATION,
        "LBFGS": LBFGS,
        "SGD": SGD,
        "LINEARIZE_J": LINEARIZE_J,
        "weights": weights,
        "BATCH_SIZE_INT": BATCH_SIZE_INT,
        "BATCH_SIZE_BOUND": BATCH_SIZE_BOUND,
        "MAX_BATCH_SIZE_DATA": MAX_BATCH_SIZE_DATA,
        "BATCH_SIZE_REG": BATCH_SIZE_REG,
        "BATCH_SIZE_STRUCT": BATCH_SIZE_STRUCT,
        "w_cs_a_mass_reg": w_cs_a_mass_reg,
        "w_cs_c_mass_reg": w_cs_c_mass_reg,
        "mass_reg_n_quad": mass_reg_n_quad,
        "N_BATCH": N_BATCH,
        "N_BATCH_LBFGS": N_BATCH_LBFGS,
        "NUM_RES_BLOCKS": NUM_RES_BLOCKS,
        "NUM_RES_BLOCK_LAYERS": NUM_RES_BLOCK_LAYERS,
        "NUM_RES_BLOCK_UNITS": NUM_RES_BLOCK_UNITS,
        "NUM_GRAD_PATH_LAYERS": NUM_GRAD_PATH_LAYERS,
        "NUM_GRAD_PATH_UNITS": NUM_GRAD_PATH_UNITS,
        "LOAD_MODEL": LOAD_MODEL,
        "LOCAL_utilFolder": LOCAL_utilFolder,
        "PRIOR_MODEL": PRIOR_MODEL,
    }


def initialize_params(args):
    inpt = parse_input_file(args.input_file)
    return initialize_params_from_inpt(inpt)


def _choose_param_builder(args, prior_model: str = "spm"):
    prior_model = str(prior_model or "spm").strip().lower()
    if getattr(args, "simpleModel", False):
        try:
            from .spm_simpler import makeParams
        except ImportError:  # pragma: no cover
            from spm_simpler import makeParams
    elif prior_model in {"assb_discharge", "spm_assb_train_discharge", "assb_train_discharge"}:
        try:
            from .spm_assb_train_discharge import makeParams
        except ImportError:  # pragma: no cover
            from spm_assb_train_discharge import makeParams
    elif prior_model in {"assb_v0", "assb_voltage_anchored", "spm_assb_v0"}:
        try:
            from .spm_assb_v0 import makeParams
        except ImportError:  # pragma: no cover
            from spm_assb_v0 import makeParams
    else:
        try:
            from .spm import makeParams
        except ImportError:  # pragma: no cover
            from spm import makeParams
    return makeParams


def initialize_nn(args, input_params: Dict[str, Any]) -> myNN:
    seed = input_params["seed"]
    PRIOR_MODEL = input_params.get("PRIOR_MODEL", "spm")
    makeParams = _choose_param_builder(args, prior_model=PRIOR_MODEL)
    params = makeParams()
    dataFolder = args.dataFolder

    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    NEURONS_NUM = input_params["NEURONS_NUM"]
    LAYERS_T_NUM = input_params["LAYERS_T_NUM"]
    LAYERS_TR_NUM = input_params["LAYERS_TR_NUM"]
    LAYERS_T_VAR_NUM = input_params["LAYERS_T_VAR_NUM"]
    LAYERS_TR_VAR_NUM = input_params["LAYERS_TR_VAR_NUM"]
    LAYERS_SPLIT_NUM = input_params["LAYERS_SPLIT_NUM"]
    alpha = input_params["alpha"]
    N_BATCH = input_params["N_BATCH"]
    NUM_RES_BLOCKS = input_params["NUM_RES_BLOCKS"]
    NUM_RES_BLOCK_LAYERS = input_params["NUM_RES_BLOCK_LAYERS"]
    NUM_RES_BLOCK_UNITS = input_params["NUM_RES_BLOCK_UNITS"]
    NUM_GRAD_PATH_LAYERS = input_params["NUM_GRAD_PATH_LAYERS"]
    NUM_GRAD_PATH_UNITS = input_params["NUM_GRAD_PATH_UNITS"]
    BATCH_SIZE_INT = input_params["BATCH_SIZE_INT"]
    BATCH_SIZE_BOUND = input_params["BATCH_SIZE_BOUND"]
    BATCH_SIZE_REG = input_params["BATCH_SIZE_REG"]
    BATCH_SIZE_STRUCT = input_params["BATCH_SIZE_STRUCT"]
    w_cs_a_mass_reg = float(input_params.get("w_cs_a_mass_reg", os.environ.get("ASSB_W_CS_A_MASS_REG", 1.0)))
    w_cs_c_mass_reg = float(input_params.get("w_cs_c_mass_reg", os.environ.get("ASSB_W_CS_C_MASS_REG", 1.0)))
    mass_reg_n_quad = int(input_params.get("mass_reg_n_quad", 8))
    MAX_BATCH_SIZE_DATA = input_params["MAX_BATCH_SIZE_DATA"]
    N_BATCH_LBFGS = input_params["N_BATCH_LBFGS"]
    HARD_IC_TIMESCALE = input_params["HARD_IC_TIMESCALE"]
    EXP_LIMITER = input_params["EXP_LIMITER"]
    COLLOCATION_MODE = input_params["COLLOCATION_MODE"]
    GRADUAL_TIME_SGD = input_params["GRADUAL_TIME_SGD"]
    GRADUAL_TIME_LBFGS = input_params["GRADUAL_TIME_LBFGS"]
    GRADUAL_TIME_MODE_LBFGS = input_params["GRADUAL_TIME_MODE_LBFGS"]
    RATIO_FIRST_TIME = input_params["RATIO_FIRST_TIME"]
    N_GRADUAL_STEPS_LBFGS = input_params["N_GRADUAL_STEPS_LBFGS"]
    RATIO_T_MIN = input_params["RATIO_T_MIN"]
    EPOCHS = input_params["EPOCHS"]
    EPOCHS_LBFGS = input_params["EPOCHS_LBFGS"]
    EPOCHS_START_LBFGS = input_params["EPOCHS_START_LBFGS"]
    LOSS_THRESHOLD = input_params["LOSS_THRESHOLD"]
    DYNAMIC_ATTENTION_WEIGHTS = input_params["DYNAMIC_ATTENTION_WEIGHTS"]
    ANNEALING_WEIGHTS = input_params["ANNEALING_WEIGHTS"]
    USE_LOSS_THRESHOLD = input_params["USE_LOSS_THRESHOLD"]
    ACTIVATION = input_params["ACTIVATION"]
    LBFGS = input_params["LBFGS"]
    SGD = input_params["SGD"]
    LINEARIZE_J = input_params["LINEARIZE_J"]
    LOAD_MODEL = input_params["LOAD_MODEL"]
    MERGED = input_params["MERGED"]
    ID = input_params["ID"]
    LOCAL_utilFolder = input_params["LOCAL_utilFolder"]
    HNN_modelFolder = input_params["HNN_modelFolder"]
    HNN_utilFolder = input_params["HNN_utilFolder"]
    HNN_params = input_params["HNN_params"]
    HNNTIME_modelFolder = input_params["HNNTIME_modelFolder"]
    HNNTIME_utilFolder = input_params["HNNTIME_utilFolder"]
    HNNTIME_val = input_params["HNNTIME_val"]
    weights = input_params["weights"]

    # Forward ASSB secondary-conservation controls into the physical parameter
    # dictionary before myNN.__init__ calls setResidualRescaling(). This avoids
    # relying on environment variables and makes the values visible in
    # ModelFin_*/config.json.
    params["w_cs_a_mass_reg"] = np.float64(w_cs_a_mass_reg)
    params["w_cs_c_mass_reg"] = np.float64(w_cs_c_mass_reg)
    params["mass_reg_n_quad"] = int(mass_reg_n_quad)
    os.environ["ASSB_W_CS_A_MASS_REG"] = str(w_cs_a_mass_reg)
    os.environ["ASSB_W_CS_C_MASS_REG"] = str(w_cs_c_mass_reg)
    os.environ["ASSB_MASS_REG_N_QUAD"] = str(mass_reg_n_quad)

    if MERGED:
        hidden_units_t = [NEURONS_NUM] * LAYERS_T_NUM
        hidden_units_t_r = [NEURONS_NUM] * LAYERS_TR_NUM
        hidden_units_cs_a = [NEURONS_NUM] * LAYERS_TR_VAR_NUM
        hidden_units_cs_c = [NEURONS_NUM] * LAYERS_TR_VAR_NUM
        hidden_units_phie = [NEURONS_NUM] * LAYERS_T_VAR_NUM
        hidden_units_phis_c = [NEURONS_NUM] * LAYERS_T_VAR_NUM
    else:
        hidden_units_t = None
        hidden_units_t_r = None
        hidden_units_cs_a = [NEURONS_NUM] * LAYERS_SPLIT_NUM
        hidden_units_cs_c = [NEURONS_NUM] * LAYERS_SPLIT_NUM
        hidden_units_phie = [NEURONS_NUM] * LAYERS_SPLIT_NUM
        hidden_units_phis_c = [NEURONS_NUM] * LAYERS_SPLIT_NUM

    if dataFolder is not None and os.path.isdir(dataFolder) and alpha[2] > 0:
        try:
            data_phie = np.load(os.path.join(dataFolder, "data_phie_multi.npz"))
            use_multi = True
            print("INFO: LOADING MULTI DATASETS")
        except Exception:
            data_phie = np.load(os.path.join(dataFolder, "data_phie.npz"))
            use_multi = False
            print("INFO: LOADING SINGLE DATASETS")

        def _load(name: str):
            fname = f"{name}_multi.npz" if use_multi else f"{name}.npz"
            return np.load(os.path.join(dataFolder, fname))

        xTrain_phie = data_phie["x_train"].astype("float64")
        yTrain_phie = data_phie["y_train"].astype("float64")
        x_params_train_phie = data_phie["x_params_train"].astype("float64")

        data_phis_c = _load("data_phis_c")
        xTrain_phis_c = data_phis_c["x_train"].astype("float64")
        yTrain_phis_c = data_phis_c["y_train"].astype("float64")
        x_params_train_phis_c = data_phis_c["x_params_train"].astype("float64")

        data_cs_a = _load("data_cs_a")
        xTrain_cs_a = data_cs_a["x_train"].astype("float64")
        yTrain_cs_a = data_cs_a["y_train"].astype("float64")
        x_params_train_cs_a = data_cs_a["x_params_train"].astype("float64")

        data_cs_c = _load("data_cs_c")
        xTrain_cs_c = data_cs_c["x_train"].astype("float64")
        yTrain_cs_c = data_cs_c["y_train"].astype("float64")
        x_params_train_cs_c = data_cs_c["x_params_train"].astype("float64")
    else:
        nParams = 2
        print("INFO: LOADING DUMMY DATA")
        xTrain_phie = np.zeros((N_BATCH, 1), dtype="float64")
        yTrain_phie = np.zeros((N_BATCH, 1), dtype="float64")
        x_params_train_phie = np.zeros((N_BATCH, nParams), dtype="float64")
        xTrain_phis_c = np.zeros((N_BATCH, 1), dtype="float64")
        yTrain_phis_c = np.zeros((N_BATCH, 1), dtype="float64")
        x_params_train_phis_c = np.zeros((N_BATCH, nParams), dtype="float64")
        xTrain_cs_a = np.zeros((N_BATCH, 2), dtype="float64")
        yTrain_cs_a = np.zeros((N_BATCH, 1), dtype="float64")
        x_params_train_cs_a = np.zeros((N_BATCH, nParams), dtype="float64")
        xTrain_cs_c = np.zeros((N_BATCH, 2), dtype="float64")
        yTrain_cs_c = np.zeros((N_BATCH, 1), dtype="float64")
        x_params_train_cs_c = np.zeros((N_BATCH, nParams), dtype="float64")

    nn = myNN(
        params=params,
        hidden_units_t=hidden_units_t,
        hidden_units_t_r=hidden_units_t_r,
        hidden_units_phie=hidden_units_phie,
        hidden_units_phis_c=hidden_units_phis_c,
        hidden_units_cs_a=hidden_units_cs_a,
        hidden_units_cs_c=hidden_units_cs_c,
        n_hidden_res_blocks=NUM_RES_BLOCKS,
        n_res_block_layers=NUM_RES_BLOCK_LAYERS,
        n_res_block_units=NUM_RES_BLOCK_UNITS,
        n_grad_path_layers=NUM_GRAD_PATH_LAYERS,
        n_grad_path_units=NUM_GRAD_PATH_UNITS,
        alpha=alpha,
        batch_size_int=BATCH_SIZE_INT,
        batch_size_bound=BATCH_SIZE_BOUND,
        batch_size_reg=BATCH_SIZE_REG,
        batch_size_struct=BATCH_SIZE_STRUCT,
        max_batch_size_data=MAX_BATCH_SIZE_DATA,
        n_batch=N_BATCH,
        n_batch_lbfgs=N_BATCH_LBFGS,
        hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
        exponentialLimiter=EXP_LIMITER,
        collocationMode=COLLOCATION_MODE,
        gradualTime_sgd=GRADUAL_TIME_SGD,
        gradualTime_lbfgs=GRADUAL_TIME_LBFGS,
        gradualTimeMode_lbfgs=GRADUAL_TIME_MODE_LBFGS,
        firstTime=np.float64(HARD_IC_TIMESCALE * RATIO_FIRST_TIME),
        n_gradual_steps_lbfgs=N_GRADUAL_STEPS_LBFGS,
        tmin_int_bound=np.float64(HARD_IC_TIMESCALE * RATIO_T_MIN),
        nEpochs=EPOCHS,
        nEpochs_lbfgs=EPOCHS_LBFGS,
        nEpochs_start_lbfgs=EPOCHS_START_LBFGS,
        initialLossThreshold=np.float64(LOSS_THRESHOLD),
        dynamicAttentionWeights=DYNAMIC_ATTENTION_WEIGHTS,
        annealingWeights=ANNEALING_WEIGHTS,
        useLossThreshold=USE_LOSS_THRESHOLD,
        activation=ACTIVATION,
        lbfgs=LBFGS,
        sgd=SGD,
        linearizeJ=LINEARIZE_J,
        params_min=[params["deg_i0_a_min"], params["deg_ds_c_min"]],
        params_max=[params["deg_i0_a_max"], params["deg_ds_c_max"]],
        xDataList=[xTrain_phie, xTrain_phis_c, xTrain_cs_a, xTrain_cs_c],
        x_params_dataList=[x_params_train_phie, x_params_train_phis_c, x_params_train_cs_a, x_params_train_cs_c],
        yDataList=[yTrain_phie, yTrain_phis_c, yTrain_cs_a, yTrain_cs_c],
        logLossFolder=f"Log_{ID}",
        modelFolder=f"Model_{ID}",
        local_utilFolder=LOCAL_utilFolder,
        hnn_utilFolder=HNN_utilFolder,
        hnn_modelFolder=HNN_modelFolder,
        hnn_params=HNN_params,
        hnntime_utilFolder=HNNTIME_utilFolder,
        hnntime_modelFolder=HNNTIME_modelFolder,
        hnntime_val=HNNTIME_val,
        weights=weights,
        verbose=True,
    )

    nn.configDict["prior_model"] = str(PRIOR_MODEL)
    # ASSB training/provenance diagnostics. These fields are intentionally
    # duplicated in config.json so a finished ModelFin_* can prove whether
    # regularization was active and which electrode was weighted.
    nn.configDict["ID"] = int(ID)
    nn.configDict["alpha"] = [float(x) for x in alpha]
    nn.configDict["BATCH_SIZE_INT"] = int(BATCH_SIZE_INT)
    nn.configDict["BATCH_SIZE_BOUND"] = int(BATCH_SIZE_BOUND)
    nn.configDict["BATCH_SIZE_REG"] = int(BATCH_SIZE_REG)
    nn.configDict["MAX_BATCH_SIZE_DATA"] = int(MAX_BATCH_SIZE_DATA)
    nn.configDict["N_BATCH"] = int(N_BATCH)
    nn.configDict["w_cs_a_mass_reg"] = float(w_cs_a_mass_reg)
    nn.configDict["w_cs_c_mass_reg"] = float(w_cs_c_mass_reg)
    nn.configDict["mass_reg_n_quad"] = int(mass_reg_n_quad)
    nn.configDict["activeReg_runtime"] = bool(getattr(nn, "activeReg", False))
    nn.configDict["regTerms_rescale"] = [float(x) for x in getattr(nn, "regTerms_rescale", [])]
    nn.configDict["regTerms_rescale_unweighted"] = [float(x) for x in getattr(nn, "regTerms_rescale_unweighted", [])]
    nn.configDict["w_cs_a_mass_reg_effective"] = float(params.get("w_cs_a_mass_reg_effective", params.get("w_cs_a_mass_reg", w_cs_a_mass_reg)))
    nn.configDict["w_cs_c_mass_reg_effective"] = float(params.get("w_cs_c_mass_reg_effective", params.get("w_cs_c_mass_reg", w_cs_c_mass_reg)))
    nn.configDict["ASSB_SOFT_LABEL_DIR"] = os.environ.get("ASSB_SOFT_LABEL_DIR", "")
    nn.configDict["ASSB_SOFT_LABEL_SUMMARY"] = os.environ.get("ASSB_SOFT_LABEL_SUMMARY", "")
    nn.configDict["ASSB_OCP_DIR"] = os.environ.get("ASSB_OCP_DIR", "")
    if "train_summary_json" in params:
        nn.configDict["train_summary_json"] = params["train_summary_json"]

    print(
        "INFO: ASSB reg diagnostics | "
        f"alpha={nn.configDict['alpha']} | "
        f"BATCH_SIZE_REG={BATCH_SIZE_REG} | "
        f"activeReg={nn.configDict['activeReg_runtime']} | "
        f"w_a={nn.configDict['w_cs_a_mass_reg']} | "
        f"w_c={nn.configDict['w_cs_c_mass_reg']} | "
        f"reg_rescale={nn.configDict['regTerms_rescale']}"
    )

    if LOAD_MODEL is not None:
        print(f"INFO: Loading model {LOAD_MODEL}")
        nn = safe_load(nn, LOAD_MODEL)
    return nn


def initialize_nn_from_params_config(params, configDict: Dict[str, Any]) -> myNN:
    hidden_units_t = configDict.get("hidden_units_t")
    hidden_units_t_r = configDict.get("hidden_units_t_r")
    hidden_units_phie = configDict.get("hidden_units_phie")
    hidden_units_phis_c = configDict.get("hidden_units_phis_c")
    hidden_units_cs_a = configDict.get("hidden_units_cs_a")
    hidden_units_cs_c = configDict.get("hidden_units_cs_c")
    n_hidden_res_blocks = int(configDict.get("n_hidden_res_blocks", 0) or 0)
    if n_hidden_res_blocks > 0:
        n_res_block_layers = int(configDict["n_res_block_layers"])
        n_res_block_units = int(configDict["n_res_block_units"])
    else:
        n_res_block_layers = 1
        n_res_block_units = 1
    n_grad_path_layers = configDict.get("n_grad_path_layers")
    if n_grad_path_layers is not None and int(n_grad_path_layers) > 0:
        n_grad_path_layers = int(n_grad_path_layers)
        n_grad_path_units = int(configDict["n_grad_path_units"])
    else:
        n_grad_path_layers = None
        n_grad_path_units = None

    HARD_IC_TIMESCALE = configDict["hard_IC_timescale"]
    EXP_LIMITER = configDict["exponentialLimiter"]
    ACTIVATION = configDict["activation"]
    LINEARIZE_J = configDict.get("linearizeJ", True)
    params_min = configDict.get("params_min", [params["deg_i0_a_min"], params["deg_ds_c_min"]])
    params_max = configDict.get("params_max", [params["deg_i0_a_max"], params["deg_ds_c_max"]])
    local_utilFolder = configDict.get("local_utilFolder")
    hnn_utilFolder = configDict.get("hnn_utilFolder")
    hnn_modelFolder = configDict.get("hnn_modelFolder")
    hnn_params = configDict.get("hnn_params")
    hnntime_utilFolder = configDict.get("hnntime_utilFolder")
    hnntime_modelFolder = configDict.get("hnntime_modelFolder")
    hnntime_val = configDict.get("hnntime_val")

    nn = myNN(
        params=params,
        hidden_units_t=hidden_units_t,
        hidden_units_t_r=hidden_units_t_r,
        hidden_units_phie=hidden_units_phie,
        hidden_units_phis_c=hidden_units_phis_c,
        hidden_units_cs_a=hidden_units_cs_a,
        hidden_units_cs_c=hidden_units_cs_c,
        n_hidden_res_blocks=n_hidden_res_blocks,
        n_res_block_layers=n_res_block_layers,
        n_res_block_units=n_res_block_units,
        n_grad_path_layers=n_grad_path_layers,
        n_grad_path_units=n_grad_path_units,
        hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
        exponentialLimiter=EXP_LIMITER,
        activation=ACTIVATION,
        dynamicAttentionWeights=bool(configDict.get("dynamicAttentionWeights", False)),
        annealingWeights=bool(configDict.get("annealingWeights", False)),
        linearizeJ=LINEARIZE_J,
        params_min=params_min,
        params_max=params_max,
        local_utilFolder=local_utilFolder,
        hnn_utilFolder=hnn_utilFolder,
        hnn_modelFolder=hnn_modelFolder,
        hnn_params=hnn_params,
        hnntime_utilFolder=hnntime_utilFolder,
        hnntime_modelFolder=hnntime_modelFolder,
        hnntime_val=hnntime_val,
        verbose=True,
    )
    return nn
