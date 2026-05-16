# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 neural network and training driver.

Complete replacement file for ModelFin_110 aging-fix1.
This version keeps the 107A-style four-branch PINN architecture, registers the
low-dimensional aging mechanism head, and keeps the original pointwise data
loss closed.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
import torch
import torch.nn as torch_nn

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (str(_ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from . import _rescale as rescale_mod
    from . import _losses as losses_mod
    from .assb_aging_fix1_config import AgingFix1Config, parse_aging_fix1_config, validate_aging_config
    from .assb_aging_mechanism import AgingMechanismHead, profiles_to_numpy, save_profiles_csv
    from .assb_aging_capacity import load_capacity_targets, q_ref_from_targets, capacity_metrics, save_json
except Exception:  # pragma: no cover
    import _rescale as rescale_mod
    import _losses as losses_mod
    from assb_aging_fix1_config import AgingFix1Config, parse_aging_fix1_config, validate_aging_config
    from assb_aging_mechanism import AgingMechanismHead, profiles_to_numpy, save_profiles_csv
    from assb_aging_capacity import load_capacity_targets, q_ref_from_targets, capacity_metrics, save_json


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


def _activation(name: str):
    name = str(name or "tanh").lower()
    if name == "relu":
        return torch_nn.ReLU()
    if name in {"silu", "swish"}:
        return torch_nn.SiLU()
    if name == "gelu":
        return torch_nn.GELU()
    if name == "elu":
        return torch_nn.ELU()
    return torch_nn.Tanh()


def _init_linear(layer: torch_nn.Linear) -> None:
    torch_nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        torch_nn.init.zeros_(layer.bias)


class DenseAct(torch_nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        super().__init__()
        self.linear = torch_nn.Linear(int(in_dim), int(out_dim))
        _init_linear(self.linear)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class BaseNet(torch_nn.Module):
    def __init__(self, in_dim: int, hidden_units: Sequence[int], activation: str = "tanh"):
        super().__init__()
        hidden_units = list(hidden_units or [20])
        layers: List[torch_nn.Module] = []
        prev = int(in_dim)
        for h in hidden_units:
            layers.append(DenseAct(prev, int(h), activation))
            prev = int(h)
        self.layers = torch_nn.ModuleList(layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class GradPath(torch_nn.Module):
    """Small gradient-pathology block used in the 101/107A line."""
    def __init__(self, in_dim: int, n_blocks: int, n_units: int, activation: str):
        super().__init__()
        self.U = DenseAct(in_dim, n_units, activation)
        self.V = DenseAct(in_dim, n_units, activation)
        self.H0 = DenseAct(in_dim, n_units, activation)
        self.Z_layers = torch_nn.ModuleList([DenseAct(n_units, n_units, activation) for _ in range(max(int(n_blocks) - 1, 0))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.U(x)
        V = self.V(x)
        H = self.H0(x)
        for z_layer in self.Z_layers:
            Z = z_layer(H)
            H = (1.0 - Z) * U + Z * V
        return H


class ASSBGradPathPINN(torch_nn.Module):
    """107A-style four-output PINN core.

    The module names intentionally follow the 101/107A branch structure
    (base_t/base_tr/gp_*/out_*).  That keeps core checkpoint keys compatible
    and avoids replacing the trained electrochemical surrogate with a new
    simplified network.
    """
    def __init__(self, parent: "myNN"):
        super().__init__()
        self.parent = parent
        self.base_t = BaseNet(3, parent.hidden_units_t, parent.activation)
        self.base_tr = BaseNet(self.base_t.out_dim + 1, parent.hidden_units_t_r, parent.activation)
        gp_layers = int(parent.n_grad_path_layers or 3)
        gp_units = int(parent.n_grad_path_units or parent.n_res_block_units or 20)
        self.gp_phie = GradPath(self.base_t.out_dim, gp_layers, gp_units, parent.activation)
        self.gp_phis = GradPath(self.base_t.out_dim, gp_layers, gp_units, parent.activation)
        self.gp_csa = GradPath(self.base_tr.out_dim, gp_layers, gp_units, parent.activation)
        self.gp_csc = GradPath(self.base_tr.out_dim, gp_layers, gp_units, parent.activation)
        self.out_phie = torch_nn.Linear(gp_units, 1)
        self.out_phis = torch_nn.Linear(gp_units, 1)
        self.out_csa = torch_nn.Linear(gp_units, 1, bias=False)
        self.out_csc = torch_nn.Linear(gp_units, 1, bias=False)
        for layer in (self.out_phie, self.out_phis, self.out_csa, self.out_csc):
            _init_linear(layer)

    def _prep(self, inputs):
        t, r, deg_i0_a, deg_ds_c = inputs
        dev = next(self.parameters()).device
        vals = []
        for x in (t, r, deg_i0_a, deg_ds_c):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float64, device=dev)
            else:
                x = x.to(dtype=torch.float64, device=dev)
            if x.ndim == 0:
                x = x.reshape(1, 1)
            elif x.ndim == 1:
                x = x.reshape(-1, 1)
            vals.append(x)
        return vals

    def forward(self, inputs, training: bool = False):
        t, r, deg_i0_a, deg_ds_c = self._prep(inputs)
        t_par = torch.cat([t, deg_i0_a, deg_ds_c], dim=1)
        h_t = self.base_t(t_par)
        h_tr = self.base_tr(torch.cat([h_t, r], dim=1))
        return [
            self.out_phie(self.gp_phie(h_t)),
            self.out_phis(self.gp_phis(h_t)),
            self.out_csa(self.gp_csa(h_tr)),
            self.out_csc(self.gp_csc(h_tr)),
        ]

def safe_save_state_dict(model: torch_nn.Module, weight_path: str | Path) -> None:
    weight_path = Path(weight_path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)


class myNN:
    def __init__(
        self,
        params,
        hidden_units_t=None,
        hidden_units_t_r=None,
        hidden_units_phie=None,
        hidden_units_phis_c=None,
        hidden_units_cs_a=None,
        hidden_units_cs_c=None,
        n_hidden_res_blocks=0,
        n_res_block_layers=1,
        n_res_block_units=1,
        n_grad_path_layers=None,
        n_grad_path_units=None,
        alpha=[0, 0, 0, 0],
        batch_size_int=0,
        batch_size_bound=0,
        max_batch_size_data=0,
        batch_size_reg=0,
        batch_size_struct=64,
        n_batch=0,
        n_batch_lbfgs=0,
        nEpochs_start_lbfgs=10,
        hard_IC_timescale=np.float64(1.0),
        exponentialLimiter=np.float64(10.0),
        collocationMode="fixed",
        gradualTime_sgd=False,
        gradualTime_lbfgs=False,
        firstTime=np.float64(0.1),
        n_gradual_steps_lbfgs=None,
        gradualTimeMode_lbfgs=None,
        tmin_int_bound=np.float64(0.1),
        nEpochs=60,
        nEpochs_lbfgs=60,
        initialLossThreshold=np.float64(100),
        dynamicAttentionWeights=False,
        annealingWeights=False,
        useLossThreshold=True,
        activation="tanh",
        linearizeJ=False,
        lbfgs=False,
        sgd=True,
        params_max=None,
        params_min=None,
        xDataList=None,
        x_params_dataList=None,
        yDataList=None,
        logLossFolder=None,
        modelFolder=None,
        local_utilFolder=None,
        hnn_utilFolder=None,
        hnn_modelFolder=None,
        hnn_params=None,
        hnntime_utilFolder=None,
        hnntime_modelFolder=None,
        hnntime_val=None,
        verbose=False,
        weights=None,
    ):
        self.verbose = bool(verbose)
        self.freq = 1
        self.logLossFolder = "Log" if logLossFolder is None else str(logLossFolder)
        self.modelFolder = "Model" if modelFolder is None else str(modelFolder)
        Path(self.logLossFolder).mkdir(parents=True, exist_ok=True)
        Path(self.modelFolder).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.params = dict(params or {})
        self.weights = weights or {}
        self.alpha = list(alpha or [1.0, 1.0, 0.0, 1.0])[:4]
        while len(self.alpha) < 4:
            self.alpha.append(0.0)
        # Hard guard: old pointwise soft-label data loss must remain off.
        self.alpha[2] = 0.0
        self.params["DATA_LOSS"] = False
        self.params["data_loss"] = False
        self.params["MAX_BATCH_SIZE_DATA"] = 0
        self.params["max_batch_size_data"] = 0

        self.batch_size_int = int(batch_size_int or 0)
        self.batch_size_bound = int(batch_size_bound or 0)
        self.batch_size_data = 0
        self.batch_size_reg = int(batch_size_reg or 0)
        self.batch_size_struct = int(batch_size_struct or 0)
        self.n_batch = int(n_batch or 1)
        self.n_batch_lbfgs = int(n_batch_lbfgs or 1)
        self.nEpochs_start_lbfgs = int(nEpochs_start_lbfgs or 0)
        self.nEpochs = int(nEpochs or 0)
        self.nEpochs_lbfgs = int(nEpochs_lbfgs or 0)
        self.hard_IC_timescale = np.float64(hard_IC_timescale)
        self.exponentialLimiter = np.float64(exponentialLimiter)
        self.collocationMode = str(collocationMode or "fixed")
        self.gradualTime_sgd = bool(gradualTime_sgd)
        self.gradualTime_lbfgs = bool(gradualTime_lbfgs)
        self.firstTime = np.float64(firstTime)
        self.tmin_int_bound = np.float64(tmin_int_bound)
        self.activation = activation
        self.linearizeJ = bool(linearizeJ)
        self.run_LBFGS = bool(lbfgs)
        self.run_SGD = bool(sgd)
        self.dynamicAttentionWeights = bool(dynamicAttentionWeights)
        self.annealingWeights = bool(annealingWeights)
        self.useLossThreshold = bool(useLossThreshold)
        self.initialLossThreshold = np.float64(initialLossThreshold)

        # Indices expected by some legacy evaluators.
        self.ind_phie = 0
        self.ind_phis_c = 1
        self.ind_cs_a = 2
        self.ind_cs_c = 3
        self.ind_t = 0
        self.ind_r = 1
        self.ind_deg_i0_a = 0
        self.ind_deg_ds_c = 1
        self.ind_phie_data = 0
        self.ind_phis_c_data = 1
        self.ind_cs_a_data = 2
        self.ind_cs_c_data = 3
        self.ind_cs_offset_data = 2

        self.activeInt = bool(self.alpha[0] and self.batch_size_int > 0)
        self.activeBound = bool(self.alpha[1] and self.batch_size_bound > 0)
        self.activeData = False
        self.activeReg = bool(self.alpha[3])

        self.tmin = np.float64(0.0)
        self.tmax = np.float64(self.params.get("tmax", self.params.get("rescale_T", 1.0)))
        self.rmin = np.float64(0.0)
        self.rmax_a = np.float64(self.params.get("Rs_a", 50e-6))
        self.rmax_c = np.float64(self.params.get("Rs_c", 1.8e-6))
        self.params_max = params_max or [self.params.get("deg_i0_a_max_eff", 1.0), self.params.get("deg_ds_c_max_eff", 1.0)]
        self.params_min = params_min or [self.params.get("deg_i0_a_min_eff", 1.0), self.params.get("deg_ds_c_min_eff", 1.0)]

        self.hidden_units_t = list(hidden_units_t or [int(self.params.get("NEURONS_NUM", 20))])
        self.hidden_units_t_r = list(hidden_units_t_r or [int(self.params.get("NEURONS_NUM", 20))])
        self.hidden_units_phie = list(hidden_units_phie or [int(self.params.get("NEURONS_NUM", 20))])
        self.hidden_units_phis_c = list(hidden_units_phis_c or [int(self.params.get("NEURONS_NUM", 20))])
        self.hidden_units_cs_a = list(hidden_units_cs_a or [int(self.params.get("NEURONS_NUM", 20))])
        self.hidden_units_cs_c = list(hidden_units_cs_c or [int(self.params.get("NEURONS_NUM", 20))])
        self.n_hidden_res_blocks = int(n_hidden_res_blocks or 0)
        self.n_res_block_layers = int(n_res_block_layers or 1)
        self.n_res_block_units = int(n_res_block_units or self.params.get("NUM_GRAD_PATH_UNITS", 20) or 20)
        self.n_grad_path_layers = int(n_grad_path_layers or self.params.get("NUM_GRAD_PATH_LAYERS", 3) or 3)
        self.n_grad_path_units = int(n_grad_path_units or self.params.get("NUM_GRAD_PATH_UNITS", 20) or 20)
        self.model = ASSBGradPathPINN(self).to(device=self.device, dtype=torch.float64)

        # Bind rescale/loss helpers as instance methods.
        self.rescale_param = rescale_mod.rescale_param.__get__(self, self.__class__)
        self.unrescale_param = rescale_mod.unrescale_param.__get__(self, self.__class__)
        self.fix_param = rescale_mod.fix_param.__get__(self, self.__class__)
        self.rescaleCs_a = rescale_mod.rescaleCs_a.__get__(self, self.__class__)
        self.rescaleCs_c = rescale_mod.rescaleCs_c.__get__(self, self.__class__)
        self.rescalePhie = rescale_mod.rescalePhie.__get__(self, self.__class__)
        self.rescalePhis_c = rescale_mod.rescalePhis_c.__get__(self, self.__class__)
        self.interior_loss = losses_mod.interior_loss.__get__(self, self.__class__)
        self.boundary_loss = losses_mod.boundary_loss.__get__(self, self.__class__)
        self.data_loss = losses_mod.data_loss.__get__(self, self.__class__)
        self.regularization_loss = losses_mod.regularization_loss.__get__(self, self.__class__)
        self.aging_mechanism_loss = losses_mod.aging_mechanism_loss.__get__(self, self.__class__)
        self.get_unweighted_loss = losses_mod.get_unweighted_loss.__get__(self, self.__class__)
        self.setResidualRescaling = losses_mod.setResidualRescaling.__get__(self, self.__class__)
        self.setResidualRescaling(weights)

        self.aging_head: Optional[AgingMechanismHead] = None
        self.cycle_features: Optional[torch.Tensor] = None
        self.capacity_target_frame = None
        self.capacity_target_batch: Optional[Dict[str, torch.Tensor]] = None
        self.q_ref_ah = float(self.params.get("q_ref_Ah", self.params.get("Q_ref_Ah", 1.0)))
        self._attach_aging_components()
        self._write_config()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _as_abs(self, path_value: Any) -> Optional[Path]:
        if path_value is None:
            return None
        s = str(path_value).strip().strip('"').strip("'")
        if not s or s.upper() in {"NONE", "NULL"}:
            return None
        p = Path(s)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    def _load_cycle_table_frame(self, csv_path: Path):
        if pd is None:
            raise RuntimeError("pandas is required to read the aging cycle table")
        frame = pd.read_csv(csv_path)
        if "cycle_id" not in frame.columns:
            raise KeyError("cycle_table.csv must contain cycle_id")
        frame = frame.sort_values("cycle_id").reset_index(drop=True)
        return frame

    def _feature_columns_from_frame(self, frame) -> List[str]:
        default = [
            "cycle_norm", "throughput_norm", "duration_norm", "I_abs_mean_norm",
            "I_abs_max_norm", "q_charge_norm", "q_discharge_norm", "rest_fraction_norm",
        ]
        explicit = self.params.get("AGING_FEATURE_COLUMNS", None)
        if explicit:
            cols = [c.strip() for c in str(explicit).replace(";", ",").split(",") if c.strip()]
        elif "feature_columns" in frame.columns and frame["feature_columns"].notna().any():
            first = str(frame["feature_columns"].dropna().iloc[0])
            cols = [c.strip() for c in first.replace(";", ",").split(",") if c.strip()]
        else:
            cols = default
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"cycle_table.csv missing feature columns: {missing}")
        return cols

    def _attach_aging_components(self) -> None:
        cfg = parse_aging_fix1_config(self.params)
        self.aging_cfg = cfg
        self.params["aging_fix1_config"] = cfg.to_dict()
        if not cfg.use_assb_aging_fix1:
            return

        cycle_csv = self._as_abs(cfg.cycle_table_csv)
        if cycle_csv is None or not cycle_csv.exists():
            print(f"[ASSB-110] cycle table not found: {cycle_csv}")
            return
        try:
            frame = self._load_cycle_table_frame(cycle_csv)
            feature_cols = self._feature_columns_from_frame(frame)
            self.cycle_table_frame = frame
            self.cycle_feature_columns = feature_cols
            self.cycle_features = torch.as_tensor(frame[feature_cols].to_numpy(dtype=float), dtype=torch.float64, device=self.device)
            self.cycle_id_tensor = torch.as_tensor(frame["cycle_id"].to_numpy(dtype=int), dtype=torch.long, device=self.device)
            self.params["cycle_t_start_s"] = frame.get("t_start_s", frame.get("t_start", frame["cycle_id"] * 0.0)).to_numpy(dtype=float)
            self.params["cycle_id_profile"] = frame["cycle_id"].to_numpy(dtype=int)
            if "q_net_cycle_C" in frame.columns:
                self.params["q_net_cycle_C"] = frame["q_net_cycle_C"].to_numpy(dtype=float)
            elif "q_net_cycle" in frame.columns:
                self.params["q_net_cycle_C"] = frame["q_net_cycle"].to_numpy(dtype=float)
            if "throughput_cycle_C" in frame.columns:
                self.params["throughput_cycle_C"] = frame["throughput_cycle_C"].to_numpy(dtype=float)
            self.params["cycle_table_csv_runtime"] = str(cycle_csv)
            self.params["AGING_FEATURE_DIM_RUNTIME"] = int(self.cycle_features.shape[1])
            cfg.feature_dim = int(self.cycle_features.shape[1])
        except Exception as exc:
            print(f"[ASSB-110] failed to load cycle table {cycle_csv}: {exc}")
            self.cycle_table_frame = None
            self.cycle_feature_columns = []
            self.cycle_features = torch.zeros((1, int(cfg.feature_dim)), dtype=torch.float64, device=self.device)
            self.cycle_id_tensor = torch.arange(1, 2, dtype=torch.long, device=self.device)

        # Capacity/SOH observations are expected to be merged into the cycle table,
        # but keep a fallback load for older tables.
        frame = getattr(self, "cycle_table_frame", None)
        if frame is not None and {"Q_obs_Ah", "SOH_obs"}.issubset(set(frame.columns)):
            cap_frame = frame.copy()
        else:
            cap_csv = self._as_abs(cfg.capacity_target_csv)
            cap_frame = None
            if cap_csv is not None and cap_csv.exists():
                try:
                    cap_frame = load_capacity_targets(cap_csv)
                    if frame is not None:
                        cap_frame = frame[["cycle_id", "split"]].merge(cap_frame, on="cycle_id", how="left")
                except Exception as exc:
                    print(f"[ASSB-110] failed to load capacity targets {cap_csv}: {exc}")
        if cap_frame is not None:
            cap_frame = cap_frame.sort_values("cycle_id").reset_index(drop=True)
            if "split" not in cap_frame.columns:
                cap_frame["split"] = "train"
            split_code = cap_frame["split"].astype(str).str.lower().map({"train": 0, "val": 1, "valid": 1, "test": 2}).fillna(0).astype(int).to_numpy()
            complete = cap_frame["complete_cycle"].astype(bool).to_numpy() if "complete_cycle" in cap_frame.columns else np.ones(len(cap_frame), dtype=bool)
            q_col = "Q_obs_Ah" if "Q_obs_Ah" in cap_frame.columns else ("Q_dis_Ah" if "Q_dis_Ah" in cap_frame.columns else "Q_discharge_Ah")
            soh_col = "SOH_obs" if "SOH_obs" in cap_frame.columns else "SOH"
            self.capacity_target_frame = cap_frame
            self.q_ref_ah = float(q_ref_from_targets(cap_frame))
            self.capacity_target_batch = {
                "cycle_id": torch.as_tensor(cap_frame["cycle_id"].to_numpy(dtype=int), dtype=torch.long, device=self.device),
                "Q_obs_Ah": torch.as_tensor(cap_frame[q_col].to_numpy(dtype=float), dtype=torch.float64, device=self.device),
                "SOH_obs": torch.as_tensor(cap_frame[soh_col].to_numpy(dtype=float), dtype=torch.float64, device=self.device),
                "split_code": torch.as_tensor(split_code, dtype=torch.long, device=self.device),
                "complete": torch.as_tensor(complete, dtype=torch.bool, device=self.device),
            }
            self.params["capacity_target_n_cycles"] = int(len(cap_frame))
            self.params["q_ref_Ah"] = float(self.q_ref_ah)

        load_aging = self._as_abs(cfg.load_aging_model)
        self.aging_head = None
        if load_aging is not None and load_aging.exists():
            try:
                self.aging_head = AgingMechanismHead.load(load_aging, map_location=self.device).to(device=self.device, dtype=torch.float64)
                # Keep runtime feature dimension consistent with the table.
                if self.aging_head.cfg.feature_dim != int(self.cycle_features.shape[1]):
                    print(f"[ASSB-110] Stage-B feature_dim={self.aging_head.cfg.feature_dim} does not match table={self.cycle_features.shape[1]}; creating a runtime head from input config.")
                    self.aging_head = AgingMechanismHead(validate_aging_config(cfg)).to(device=self.device, dtype=torch.float64)
            except Exception as exc:
                print(f"[ASSB-110] failed to load Stage-B aging model {load_aging}: {exc}; creating a new head")
                self.aging_head = AgingMechanismHead(validate_aging_config(cfg)).to(device=self.device, dtype=torch.float64)
        else:
            self.aging_head = AgingMechanismHead(validate_aging_config(cfg)).to(device=self.device, dtype=torch.float64)

        if cfg.freeze_107a_core:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.params["core_frozen_for_aging_fix1"] = True
        print(f"[ASSB-110] aging-fix1 enabled; cycles={int(self.cycle_features.shape[0])}, features={int(self.cycle_features.shape[1])}, q_ref={float(self.q_ref_ah):.8g} Ah")

    def get_aging_profiles(self):
        if self.aging_head is None:
            return None
        if self.cycle_features is None:
            self.cycle_features = torch.zeros((1, int(self.aging_head.cfg.feature_dim)), dtype=torch.float64, device=self.device)
        cycle_id = getattr(self, "cycle_id_tensor", None)
        return self.aging_head(self.cycle_features, cycle_id=cycle_id, q_ref_ah=float(self.q_ref_ah))

    def parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.aging_head is not None:
            params.extend([p for p in self.aging_head.parameters() if p.requires_grad])
        return iter(params)

    # ------------------------------------------------------------------
    # Saving / logging
    # ------------------------------------------------------------------
    def _clean_value(self, v: Any):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        if isinstance(v, Path):
            return str(v)
        if callable(v):
            return getattr(v, "__name__", str(v))
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    def _write_config(self) -> None:
        cfg = dict(self.params)
        cfg.update({
            "alpha": list(self.alpha),
            "DATA_LOSS": False,
            "MAX_BATCH_SIZE_DATA": 0,
            "USE_ASSB_AGING_MECHANISM": bool(self.aging_head is not None),
            "aging_head_enabled": self.aging_head is not None,
            "modelFolder": self.modelFolder,
            "logLossFolder": self.logLossFolder,
        })
        if self.aging_head is not None:
            cfg["aging_config"] = self.aging_head.cfg.to_dict()
        cfg = {str(k): self._clean_value(v) for k, v in cfg.items()}
        self.configDict = cfg
        self.config = cfg
        Path(self.modelFolder).mkdir(parents=True, exist_ok=True)
        (Path(self.modelFolder) / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_aging_state(self) -> None:
        if self.aging_head is None:
            return
        out_dir = Path(self.modelFolder)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.aging_head.save(out_dir, extra={"feature_columns": getattr(self, "cycle_feature_columns", []), "q_ref_ah": float(self.q_ref_ah)})
            profiles = self.get_aging_profiles()
            npy = profiles_to_numpy(profiles)
            np.savez_compressed(out_dir / "aging_profiles.npz", **npy)
            extra = {}
            frame = getattr(self, "cycle_table_frame", None)
            if frame is not None and "split" in frame.columns:
                extra["split"] = frame["split"].astype(str).to_numpy()
            save_profiles_csv(profiles, out_dir / "mechanism_by_cycle.csv", extra_columns=extra)
        except Exception as exc:
            print(f"[ASSB-110] warning: failed to save aging state/profile tables: {exc}")

    def _save_checkpoint(self, filename: str = "best.pt") -> None:
        safe_save_state_dict(self.model, Path(self.modelFolder) / filename)
        if self.aging_head is not None:
            self._save_aging_state()
        self._write_config()

    def logTraining(self, rows: List[dict]) -> None:
        Path(self.logLossFolder).mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        log_path = Path(self.logLossFolder) / "log.csv"
        fields = list(rows[0].keys())
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _scalarize_terms(self, nested_terms) -> torch.Tensor:
        device = self.device
        total = torch.tensor(0.0, dtype=torch.float64, device=device)
        for group in nested_terms or []:
            for term in group or []:
                total = total + torch.mean(rescale_mod._to_tensor(term, like=term).square())
        return total

    def train(
        self,
        learningRateModel=1.0e-3,
        learningRateModelFinal=None,
        lrSchedulerModel=None,
        learningRateWeights=1.0e-3,
        learningRateWeightsFinal=None,
        lrSchedulerWeights=None,
        learningRateLBFGS=1.0,
        inner_epochs=1,
        start_weight_training_epoch=0,
        gradient_threshold=None,
    ):
        rows: List[dict] = []
        best_loss = float("inf")
        best_model_state = None
        best_aging_state = None
        params_to_train = list(self.parameters())
        if not params_to_train:
            raise RuntimeError("No trainable parameters: check FREEZE_107A_CORE and aging head registration")
        opt = torch.optim.Adam(params_to_train, lr=float(learningRateModel))
        n_epochs = max(int(self.nEpochs), 1)
        grad_clip = float(gradient_threshold) if gradient_threshold is not None else 10.0
        for epoch in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            int_loss = self._scalarize_terms(self.interior_loss()) if self.activeInt else torch.tensor(0.0, dtype=torch.float64, device=self.device)
            bound_loss = self._scalarize_terms(self.boundary_loss()) if self.activeBound else torch.tensor(0.0, dtype=torch.float64, device=self.device)
            reg_loss, reg_info = self.aging_mechanism_loss() if self.activeReg else (torch.tensor(0.0, dtype=torch.float64, device=self.device), {})
            loss = float(self.alpha[0]) * int_loss + float(self.alpha[1]) * bound_loss + float(self.alpha[3]) * reg_loss
            if not torch.isfinite(loss):
                print(f"[ASSB-110] non-finite loss at epoch {epoch}; skip step")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, grad_clip)
            opt.step()
            if lrSchedulerModel is not None:
                for group in opt.param_groups:
                    group["lr"] = float(lrSchedulerModel(epoch, group["lr"]))
            val = float(loss.detach().cpu())
            if val < best_loss:
                best_loss = val
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.aging_head is not None:
                    best_aging_state = {k: v.detach().cpu().clone() for k, v in self.aging_head.state_dict().items()}
                self._save_checkpoint("best.pt")
            if epoch % max(n_epochs // 100, 1) == 0 or epoch == n_epochs - 1:
                row = {
                    "epoch": epoch,
                    "stage": "aging_fix1_stageC",
                    "loss": val,
                    "int_loss": float(int_loss.detach().cpu()),
                    "bound_loss": float(bound_loss.detach().cpu()),
                    "data_loss": 0.0,
                    "reg_loss": float(reg_loss.detach().cpu()),
                    "data_loss_active": False,
                    "cap_loss": float(reg_info.get("cap_loss", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "soh_loss": float(reg_info.get("soh_loss", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "cap_mae_mAh": float(reg_info.get("cap_mae_mAh", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "soh_mae": float(reg_info.get("soh_mae", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "R_ohm_mean": float(reg_info.get("R_ohm_mean", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "f_lam_c_min": float(reg_info.get("f_lam_c_min", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "theta_window_min": float(reg_info.get("theta_window_min", torch.tensor(float("nan"))).detach().cpu()) if reg_info else float("nan"),
                    "lr": float(opt.param_groups[0]["lr"]),
                }
                rows.append(row)
                print(
                    f"[ASSB-110] epoch={epoch:05d} loss={val:.6e} int={row['int_loss']:.3e} "
                    f"bound={row['bound_loss']:.3e} reg={row['reg_loss']:.3e} "
                    f"Q_MAE={row['cap_mae_mAh']:.6f}mAh SOH_MAE={row['soh_mae']:.6f}"
                )
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state, strict=False)
        if self.aging_head is not None and best_aging_state is not None:
            self.aging_head.load_state_dict(best_aging_state, strict=False)
        self._save_checkpoint("best.pt")
        self._save_checkpoint("last.pt")
        self.logTraining(rows)
        return float(best_loss if np.isfinite(best_loss) else 0.0)

    # ------------------------------------------------------------------
    # Evaluator compatibility helpers
    # ------------------------------------------------------------------
    def stretchT(self, t, tmin_old, tmax_old, tmin_new, tmax_new):
        t = rescale_mod._to_tensor(t, device=self.device)
        return tmin_new + (t - tmin_old) * (tmax_new - tmin_new) / max(float(tmax_old - tmin_old), 1.0e-12)

    def predict_raw(self, t, r):
        t = rescale_mod._to_tensor(t, device=self.device)
        r = rescale_mod._to_tensor(r, device=self.device)
        one = torch.ones_like(t)
        return self.model([t, r, one, one], training=False)


__all__ = ["myNN", "ASSBGradPathPINN", "safe_save_state_dict"]
