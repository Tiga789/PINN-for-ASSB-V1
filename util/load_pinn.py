from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
from typing import Optional

try:
    from . import argument
    from .init_pinn import initialize_nn_from_params_config, safe_load
except ImportError:  # pragma: no cover
    import argument
    from init_pinn import initialize_nn_from_params_config, safe_load

args = argument.initArg()


def reload(utilFolder, localUtilFolder, params_loaded, checkRescale: bool = False):
    """
    Compatibility stub.

    The original TensorFlow loader dynamically re-imported helper modules from the
    saved util folder. In this PyTorch port, the runtime modules are already loaded
    from the current package, so there is nothing extra to do here.
    """
    return None


def _make_params(simple_model: bool, prior_model: str = "spm", train_summary_json=None):
    prior_model = str(prior_model or "spm").strip().lower()
    if simple_model:
        try:
            from .spm_simpler import makeParams
        except ImportError:  # pragma: no cover
            from spm_simpler import makeParams
        return makeParams()
    if prior_model in {"assb_discharge", "spm_assb_train_discharge", "assb_train_discharge"}:
        try:
            from .spm_assb_train_discharge import makeParams
        except ImportError:  # pragma: no cover
            from spm_assb_train_discharge import makeParams
        return makeParams(summary_json=train_summary_json)
    if prior_model in {"assb_v0", "assb_voltage_anchored", "spm_assb_v0"}:
        try:
            from .spm_assb_v0 import makeParams
        except ImportError:  # pragma: no cover
            from spm_assb_v0 import makeParams
        return makeParams()
    try:
        from .spm import makeParams
    except ImportError:  # pragma: no cover
        from spm import makeParams
    return makeParams()


def load_model(utilFolder, modelFolder, localUtilFolder=None, loadDep: bool = False, checkRescale: bool = False):
    modelFolder = Path(modelFolder)
    utilFolder = Path(utilFolder)
    if localUtilFolder is None:
        localUtilFolder = str(utilFolder)

    config_path = modelFolder / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Could not find {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        configDict = json.load(f)

    simple_model = bool(configDict.get('simple_model', False) or getattr(args, 'simpleModel', False))
    prior_model = configDict.get("prior_model", "spm")
    train_summary_json = configDict.get("train_summary_json")
    params_loaded = _make_params(simple_model=simple_model, prior_model=prior_model, train_summary_json=train_summary_json)
    nn = initialize_nn_from_params_config(params_loaded, configDict)

    for candidate in [
        modelFolder / 'best.pt',
        modelFolder / 'last.pt',
        modelFolder / 'best.weights.h5',
        modelFolder / 'lastLBFGS.pt',
        modelFolder / 'lastSGD.pt',
    ]:
        if candidate.exists():
            nn = safe_load(nn, str(candidate))
            break
    else:
        raise FileNotFoundError(f'No checkpoint found in {modelFolder}')

    reload(str(utilFolder), str(localUtilFolder), params_loaded, checkRescale=checkRescale)
    return nn
