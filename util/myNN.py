from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as torch_nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

try:
    from ._losses import (
        boundary_loss,
        data_loss,
        get_unweighted_loss,
        interior_loss,
        loss_fn,
        regularization_loss,
        setResidualRescaling,
    )
    from ._rescale import (
        fix_param,
        get_cs_a_hnn,
        get_cs_a_hnntime,
        get_cs_c_hnn,
        get_cs_c_hnntime,
        get_phie0,
        get_phie_hnn,
        get_phie_hnntime,
        get_phis_c0,
        get_phis_c_hnn,
        get_phis_c_hnntime,
        rescale_param,
        rescaleCs_a,
        rescaleCs_c,
        rescalePhie,
        rescalePhis_c,
        unrescale_param,
    )
except ImportError:  # pragma: no cover
    from _losses import (
        boundary_loss,
        data_loss,
        get_unweighted_loss,
        interior_loss,
        loss_fn,
        regularization_loss,
        setResidualRescaling,
    )
    from _rescale import (
        fix_param,
        get_cs_a_hnn,
        get_cs_a_hnntime,
        get_cs_c_hnn,
        get_cs_c_hnntime,
        get_phie0,
        get_phie_hnn,
        get_phie_hnntime,
        get_phis_c0,
        get_phis_c_hnn,
        get_phis_c_hnntime,
        rescale_param,
        rescaleCs_a,
        rescaleCs_c,
        rescalePhie,
        rescalePhis_c,
        unrescale_param,
    )


def _as_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def _to_tensor(x, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x.to(dtype=torch.float64)
        if device is not None:
            out = out.to(device)
    else:
        out = torch.as_tensor(x, dtype=torch.float64, device=device)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _check_data_shape(xData, x_params_data, yData):
    if len(xData.shape) != 2:
        raise SystemExit(f"Expected rank-2 xData, got shape={xData.shape}")
    if len(x_params_data.shape) != 2:
        raise SystemExit(f"Expected rank-2 x_params_data, got shape={x_params_data.shape}")
    if len(yData.shape) != 2:
        raise SystemExit(f"Expected rank-2 yData, got shape={yData.shape}")


def _complete_dataset(xDataList, x_params_dataList, yDataList):
    nList = [x.shape[0] for x in xDataList]
    maxN = max(nList)
    toFillList = [max(maxN - n, 0) for n in nList]
    for i, toFill in enumerate(toFillList):
        if toFill < nList[i]:
            xDataList[i] = np.vstack((xDataList[i], xDataList[i][:toFill, :]))
            x_params_dataList[i] = np.vstack((x_params_dataList[i], x_params_dataList[i][:toFill, :]))
            yDataList[i] = np.vstack((yDataList[i], yDataList[i][:toFill, :]))
        if toFill > nList[i]:
            nrep = toFill // nList[i]
            nrep_res = toFill - nList[i] * nrep
            x_tmp = xDataList[i].copy()
            x_params_tmp = x_params_dataList[i].copy()
            y_tmp = yDataList[i].copy()
            for _ in range(nrep):
                xDataList[i] = np.vstack((xDataList[i], x_tmp))
                x_params_dataList[i] = np.vstack((x_params_dataList[i], x_params_tmp))
                yDataList[i] = np.vstack((yDataList[i], y_tmp))
            xDataList[i] = np.vstack((xDataList[i], x_tmp[:nrep_res, :]))
            x_params_dataList[i] = np.vstack((x_params_dataList[i], x_params_tmp[:nrep_res, :]))
            yDataList[i] = np.vstack((yDataList[i], y_tmp[:nrep_res, :]))
    return maxN


def _activation(name: str):
    name = str(name).lower()
    if name == "tanh":
        return torch_nn.Tanh()
    if name == "sigmoid":
        return torch_nn.Sigmoid()
    if name == "elu":
        return torch_nn.ELU()
    if name == "selu":
        return torch_nn.SELU()
    if name == "gelu":
        return torch_nn.GELU()
    if name == "swish":
        return torch_nn.SiLU()
    raise SystemExit(f"ABORTING: Activation {name} unrecognized")


def _init_linear(linear: torch_nn.Linear):
    torch_nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    if linear.bias is not None:
        torch_nn.init.zeros_(linear.bias)


class DenseAct(torch_nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        super().__init__()
        self.linear = torch_nn.Linear(in_dim, out_dim)
        _init_linear(self.linear)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ResidualBlock(torch_nn.Module):
    def __init__(self, n_units: int, n_layers: int, activation: str):
        super().__init__()
        layers: list[torch_nn.Module] = []
        for _ in range(n_layers):
            layers.append(DenseAct(n_units, n_units, activation))
        self.layers = torch_nn.ModuleList(layers)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.act(x + h)


class HeadNet(torch_nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int] | None,
        n_hidden_res_blocks: int,
        n_res_block_layers: int,
        n_res_block_units: int,
        activation: str,
        zero_bias_out: bool = False,
    ):
        super().__init__()
        hidden_units = list(hidden_units or [])
        self.pre_layers = torch_nn.ModuleList()
        curr_dim = input_dim
        for unit in hidden_units:
            self.pre_layers.append(DenseAct(curr_dim, int(unit), activation))
            curr_dim = int(unit)
        self.has_pre_res = True
        if curr_dim != int(n_res_block_units):
            self.pre_res = DenseAct(curr_dim, int(n_res_block_units), activation)
        else:
            self.pre_res = torch_nn.Identity()
        curr_dim = int(n_res_block_units)
        self.res_blocks = torch_nn.ModuleList(
            [ResidualBlock(curr_dim, int(n_res_block_layers), activation) for _ in range(int(n_hidden_res_blocks))]
        )
        self.out = torch_nn.Linear(curr_dim, 1, bias=not zero_bias_out)
        _init_linear(self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.pre_layers:
            h = layer(h)
        h = self.pre_res(h)
        for block in self.res_blocks:
            h = block(h)
        return self.out(h)


class BaseNet(torch_nn.Module):
    def __init__(self, input_dim: int, hidden_units: Sequence[int] | None, activation: str):
        super().__init__()
        hidden_units = list(hidden_units or [])
        self.layers = torch_nn.ModuleList()
        curr_dim = input_dim
        for unit in hidden_units:
            self.layers.append(DenseAct(curr_dim, int(unit), activation))
            curr_dim = int(unit)
        self.out_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class GradPath(torch_nn.Module):
    def __init__(self, in_dim: int, n_blocks: int, n_units: int, activation: str):
        super().__init__()
        self.U = DenseAct(in_dim, n_units, activation)
        self.V = DenseAct(in_dim, n_units, activation)
        self.H0 = DenseAct(in_dim, n_units, activation)
        self.Z_layers = torch_nn.ModuleList([DenseAct(n_units, n_units, activation) for _ in range(max(n_blocks - 1, 0))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.U(x)
        V = self.V(x)
        H = self.H0(x)
        for z_layer in self.Z_layers:
            Z = z_layer(H)
            H = (1.0 - Z) * U + Z * V
        return H


class SplitPINN(torch_nn.Module):
    def __init__(self, parent: "myNN"):
        super().__init__()
        self.parent = parent
        self.phie_head = HeadNet(3, parent.hidden_units_phie, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation)
        self.phis_head = HeadNet(3, parent.hidden_units_phis_c, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation)
        self.csa_head = HeadNet(4, parent.hidden_units_cs_a, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation, zero_bias_out=True)
        self.csc_head = HeadNet(4, parent.hidden_units_cs_c, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation, zero_bias_out=True)

    def _prep(self, inputs):
        t, r, deg_i0_a, deg_ds_c = inputs
        dev = next(self.parameters()).device
        t = _to_tensor(t, dev)
        r = _to_tensor(r, dev)
        deg_i0_a = _to_tensor(deg_i0_a, dev)
        deg_ds_c = _to_tensor(deg_ds_c, dev)
        return t, r, deg_i0_a, deg_ds_c

    def forward(self, inputs, training: bool = False):
        t, r, deg_i0_a, deg_ds_c = self._prep(inputs)
        t_par = torch.cat([t, deg_i0_a, deg_ds_c], dim=1)
        cs_in = torch.cat([t_par, r], dim=1)
        return [
            self.phie_head(t_par),
            self.phis_head(t_par),
            self.csa_head(cs_in),
            self.csc_head(cs_in),
        ]


class MergedPINN(torch_nn.Module):
    def __init__(self, parent: "myNN"):
        super().__init__()
        self.parent = parent
        self.base_t = BaseNet(3, parent.hidden_units_t, parent.activation)
        self.base_tr = BaseNet(self.base_t.out_dim + 1, parent.hidden_units_t_r, parent.activation)
        self.phie_head = HeadNet(self.base_t.out_dim, parent.hidden_units_phie, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation)
        self.phis_head = HeadNet(self.base_t.out_dim, parent.hidden_units_phis_c, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation)
        self.csa_head = HeadNet(self.base_tr.out_dim, parent.hidden_units_cs_a, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation, zero_bias_out=True)
        self.csc_head = HeadNet(self.base_tr.out_dim, parent.hidden_units_cs_c, parent.n_hidden_res_blocks, parent.n_res_block_layers, parent.n_res_block_units, parent.activation, zero_bias_out=True)

    def _prep(self, inputs):
        t, r, deg_i0_a, deg_ds_c = inputs
        dev = next(self.parameters()).device
        t = _to_tensor(t, dev)
        r = _to_tensor(r, dev)
        deg_i0_a = _to_tensor(deg_i0_a, dev)
        deg_ds_c = _to_tensor(deg_ds_c, dev)
        return t, r, deg_i0_a, deg_ds_c

    def forward(self, inputs, training: bool = False):
        t, r, deg_i0_a, deg_ds_c = self._prep(inputs)
        t_par = torch.cat([t, deg_i0_a, deg_ds_c], dim=1)
        h_t = self.base_t(t_par)
        h_tr = self.base_tr(torch.cat([h_t, r], dim=1))
        return [
            self.phie_head(h_t),
            self.phis_head(h_t),
            self.csa_head(h_tr),
            self.csc_head(h_tr),
        ]


class GradPathPINN(torch_nn.Module):
    def __init__(self, parent: "myNN"):
        super().__init__()
        self.parent = parent
        self.base_t = BaseNet(3, parent.hidden_units_t, parent.activation)
        self.base_tr = BaseNet(self.base_t.out_dim + 1, parent.hidden_units_t_r, parent.activation)
        self.gp_phie = GradPath(self.base_t.out_dim, int(parent.n_grad_path_layers), int(parent.n_grad_path_units), parent.activation)
        self.gp_phis = GradPath(self.base_t.out_dim, int(parent.n_grad_path_layers), int(parent.n_grad_path_units), parent.activation)
        self.gp_csa = GradPath(self.base_tr.out_dim, int(parent.n_grad_path_layers), int(parent.n_grad_path_units), parent.activation)
        self.gp_csc = GradPath(self.base_tr.out_dim, int(parent.n_grad_path_layers), int(parent.n_grad_path_units), parent.activation)
        self.out_phie = torch_nn.Linear(int(parent.n_grad_path_units), 1)
        self.out_phis = torch_nn.Linear(int(parent.n_grad_path_units), 1)
        self.out_csa = torch_nn.Linear(int(parent.n_grad_path_units), 1, bias=False)
        self.out_csc = torch_nn.Linear(int(parent.n_grad_path_units), 1, bias=False)
        for layer in [self.out_phie, self.out_phis, self.out_csa, self.out_csc]:
            _init_linear(layer)

    def _prep(self, inputs):
        t, r, deg_i0_a, deg_ds_c = inputs
        dev = next(self.parameters()).device
        t = _to_tensor(t, dev)
        r = _to_tensor(r, dev)
        deg_i0_a = _to_tensor(deg_i0_a, dev)
        deg_ds_c = _to_tensor(deg_ds_c, dev)
        return t, r, deg_i0_a, deg_ds_c

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


def safe_save_state_dict(model: torch_nn.Module, weight_path: str):
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
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
        hard_IC_timescale=np.float64(0.81),
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
        params_max=[],
        params_min=[],
        xDataList=[],
        x_params_dataList=[],
        yDataList=[],
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
        self.verbose = verbose
        self.freq = 1
        self.logLossFolder = "Log" if logLossFolder is None else logLossFolder
        self.modelFolder = "Model" if modelFolder is None else modelFolder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.params = params
        self.local_utilFolder = local_utilFolder
        self.hnn_utilFolder = hnn_utilFolder
        self.hnn_modelFolder = hnn_modelFolder
        self.hnn_params = hnn_params
        self.use_hnn = False
        if hnn_utilFolder is not None and hnn_modelFolder is not None:
            self.use_hnn = True
            self.vprint("INFO: LOADING HNN...")
            try:
                from .load_pinn import load_model
            except ImportError:  # pragma: no cover
                from load_pinn import load_model
            self.hnn = load_model(
                utilFolder=hnn_utilFolder,
                modelFolder=hnn_modelFolder,
                localUtilFolder=self.local_utilFolder,
                loadDep=True,
                checkRescale=False,
            )
        self.hnntime_utilFolder = hnntime_utilFolder
        self.hnntime_modelFolder = hnntime_modelFolder
        self.hnntime_val = hnntime_val
        self.use_hnntime = False
        if hnntime_utilFolder is not None and hnntime_modelFolder is not None:
            self.use_hnntime = True
            self.vprint("INFO: LOADING HNN-TIME...")
            try:
                from .load_pinn import load_model
            except ImportError:  # pragma: no cover
                from load_pinn import load_model
            self.hnntime = load_model(
                utilFolder=hnntime_utilFolder,
                modelFolder=hnntime_modelFolder,
                localUtilFolder=self.local_utilFolder,
                loadDep=True,
                checkRescale=False,
            )

        self.hidden_units_t = hidden_units_t
        self.hidden_units_t_r = hidden_units_t_r
        self.hidden_units_phie = hidden_units_phie
        self.hidden_units_phis_c = hidden_units_phis_c
        self.hidden_units_cs_a = hidden_units_cs_a
        self.hidden_units_cs_c = hidden_units_cs_c
        self.n_hidden_res_blocks = int(n_hidden_res_blocks or 0)
        self.n_res_block_layers = int(n_res_block_layers or 1)
        self.n_res_block_units = int(n_res_block_units or 1)
        self.n_grad_path_layers = None if n_grad_path_layers in [None, 0] else int(n_grad_path_layers)
        self.n_grad_path_units = None if n_grad_path_units in [None, 0] else int(n_grad_path_units)
        self.dynamicAttentionWeights = bool(dynamicAttentionWeights)
        self.annealingWeights = bool(annealingWeights)
        if self.dynamicAttentionWeights:
            print("WARNING: PyTorch port currently disables dynamic attention weights. Falling back to fixed loss weights.")
            self.dynamicAttentionWeights = False
        if self.annealingWeights:
            print("WARNING: PyTorch port currently disables annealing weights. Falling back to fixed loss weights.")
            self.annealingWeights = False
        self.useLossThreshold = bool(useLossThreshold)
        self.activation = str(activation).lower()
        _activation(self.activation)
        self.tmin = np.float64(self.params["tmin"])
        self.tmax = np.float64(self.params["tmax"])
        self.rmin = np.float64(self.params["rmin"])
        self.rmax_a = np.float64(self.params["Rs_a"])
        self.rmax_c = np.float64(self.params["Rs_c"])
        self.ind_t = np.int32(0)
        self.ind_r = np.int32(1)
        self.ind_phie = np.int32(0)
        self.ind_phis_c = np.int32(1)
        self.ind_cs_offset = np.int32(2)
        self.ind_cs_a = np.int32(2)
        self.ind_cs_c = np.int32(3)
        self.ind_phie_data = np.int32(0)
        self.ind_phis_c_data = np.int32(1)
        self.ind_cs_offset_data = np.int32(2)
        self.ind_cs_a_data = np.int32(2)
        self.ind_cs_c_data = np.int32(3)
        self.alpha = [np.float64(a) for a in alpha]
        self.alpha_unweighted = [np.float64(1.0) for _ in alpha]
        self.phis_a0 = np.float64(0.0)
        self.ce_0 = self.params["ce0"]
        self.cs_a0 = self.params["cs_a0"]
        self.cs_c0 = self.params["cs_c0"]
        self.ind_deg_i0_a = np.int32(0)
        self.ind_deg_ds_c = np.int32(1)
        self.ind_deg_i0_a_nn = max(self.ind_t, self.ind_r) + self.ind_deg_i0_a
        self.ind_deg_ds_c_nn = max(self.ind_t, self.ind_r) + self.ind_deg_ds_c
        self.dim_params = np.int32(2)
        self.params_min = [float(v) for v in params_min]
        self.params_max = [float(v) for v in params_max]
        self.resc_params = [(min_val + max_val) / 2.0 for (min_val, max_val) in zip(self.params_min, self.params_max)]
        self.hard_IC_timescale = float(hard_IC_timescale)
        self.exponentialLimiter = float(exponentialLimiter)
        self.collocationMode = str(collocationMode).lower()
        self.firstTime = np.float64(firstTime)
        self.tmin_int_bound = np.float64(tmin_int_bound)
        self.dim_inpt = np.int32(2)
        self.nEpochs = int(nEpochs)
        self.nEpochs_lbfgs = int(nEpochs_lbfgs)
        self.linearizeJ = bool(linearizeJ)
        self.gradualTime_sgd = bool(gradualTime_sgd)
        self.gradualTime_lbfgs = bool(gradualTime_lbfgs)
        self.n_gradual_steps_lbfgs = 0 if n_gradual_steps_lbfgs in [None, 0] else int(n_gradual_steps_lbfgs)
        self.gradualTimeMode_lbfgs = gradualTimeMode_lbfgs
        self.reg = 0
        self.n_batch = max(int(n_batch), 1)
        self.initialLossThreshold = float(initialLossThreshold)
        self.batch_size_int = int(batch_size_int)
        self.batch_size_bound = int(batch_size_bound)
        self.batch_size_reg = int(batch_size_reg)
        self.max_batch_size_data = int(max_batch_size_data)
        self.total_step = 0
        self.current_stage = "SGD"

        if self.gradualTime_sgd:
            self.timeIncreaseExponent = -np.log((self.firstTime - np.float64(self.params["tmin"])) / (np.float64(self.params["tmax"]) - np.float64(self.params["tmin"])))

        if (xDataList != []) and (abs(self.alpha[2]) > 1e-16) and (self.max_batch_size_data > 0):
            xDataList = [_as_numpy(x) for x in xDataList]
            x_params_dataList = [_as_numpy(x) for x in x_params_dataList]
            yDataList = [_as_numpy(y) for y in yDataList]
            for i in range(len(xDataList)):
                _check_data_shape(xDataList[i], x_params_dataList[i], yDataList[i])
            ndata = _complete_dataset(xDataList, x_params_dataList, yDataList)
            batch_size_data = min(ndata // self.n_batch, self.max_batch_size_data)
            self.batch_size_data = int(max(batch_size_data, 1))
            self.new_nData = self.n_batch * self.batch_size_data
            self.vprint("new n data =", self.new_nData)
            self.vprint("batch_size_data =", self.batch_size_data)
        else:
            self.batch_size_data = 1
            self.new_nData = self.n_batch

        n_int = self.n_batch * max(self.batch_size_int, 1)
        n_bound = self.n_batch * max(self.batch_size_bound, 1)
        n_reg = self.n_batch * max(self.batch_size_reg, 1)
        tmin_int = self.tmin_int_bound
        tmin_bound = self.tmin_int_bound
        tmin_reg = self.tmin_int_bound

        self.activeInt = True
        self.activeBound = True
        self.activeData = True
        self.activeReg = True
        if self.batch_size_int == 0 or abs(self.alpha[0]) < 1e-12:
            self.vprint("INFO: INT loss is INACTIVE")
            self.activeInt = False
            n_int = self.n_batch
            self.batch_size_int = 1
        else:
            self.vprint("INFO: INT loss is ACTIVE")
        if self.batch_size_bound == 0 or abs(self.alpha[1]) < 1e-12:
            self.vprint("INFO: BOUND loss is INACTIVE")
            self.activeBound = False
            n_bound = self.n_batch
            self.batch_size_bound = 1
        else:
            self.vprint("INFO: BOUND loss is ACTIVE")
        if self.max_batch_size_data == 0 or abs(self.alpha[2]) < 1e-12 or xDataList == []:
            self.vprint("INFO: DATA loss is INACTIVE")
            self.activeData = False
            self.batch_size_data = 1
        else:
            self.vprint("INFO: DATA loss is ACTIVE")
        if self.batch_size_reg == 0 or abs(self.alpha[3]) < 1e-12:
            self.vprint("INFO: REG loss is INACTIVE")
            self.activeReg = False
            n_reg = self.n_batch
            self.batch_size_reg = 1
        else:
            self.vprint("INFO: REG loss is ACTIVE")

        self.setResidualRescaling(weights)

        # Collocation points
        self.r_a_int = torch.empty((n_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.rmin) + 1e-12, float(self.rmax_a))
        self.r_c_int = torch.empty((n_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.rmin) + 1e-12, float(self.rmax_c))
        self.r_maxa_int = float(self.rmax_a) * torch.ones((n_int, 1), dtype=torch.float64, device=self.device)
        self.r_maxc_int = float(self.rmax_c) * torch.ones((n_int, 1), dtype=torch.float64, device=self.device)
        t_int_hi = float(self.firstTime) if self.gradualTime_sgd else float(self.tmax)
        self.t_int = torch.empty((n_int, 1), dtype=torch.float64, device=self.device).uniform_(float(tmin_int), t_int_hi)
        self.deg_i0_a_int = torch.empty((n_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_i0_a_min_eff"]), float(self.params["deg_i0_a_max_eff"]))
        self.deg_ds_c_int = torch.empty((n_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_ds_c_min_eff"]), float(self.params["deg_ds_c_max_eff"]))
        self.ind_int_col_t = np.int32(0)
        self.ind_int_col_r_a = np.int32(1)
        self.ind_int_col_r_c = np.int32(2)
        self.ind_int_col_r_maxa = np.int32(3)
        self.ind_int_col_r_maxc = np.int32(4)
        self.int_col_pts = [self.t_int, self.r_a_int, self.r_c_int, self.r_maxa_int, self.r_maxc_int]
        self.ind_int_col_params_deg_i0_a = np.int32(0)
        self.ind_int_col_params_deg_ds_c = np.int32(1)
        self.int_col_params = [self.deg_i0_a_int, self.deg_ds_c_int]

        self.r_min_bound = torch.zeros((n_bound, 1), dtype=torch.float64, device=self.device)
        self.r_maxa_bound = float(self.rmax_a) * torch.ones((n_bound, 1), dtype=torch.float64, device=self.device)
        self.r_maxc_bound = float(self.rmax_c) * torch.ones((n_bound, 1), dtype=torch.float64, device=self.device)
        self.deg_i0_a_bound = torch.empty((n_bound, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_i0_a_min_eff"]), float(self.params["deg_i0_a_max_eff"]))
        self.deg_ds_c_bound = torch.empty((n_bound, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_ds_c_min_eff"]), float(self.params["deg_ds_c_max_eff"]))
        t_bound_hi = float(self.firstTime) if self.gradualTime_sgd else float(self.tmax)
        self.t_bound = torch.empty((n_bound, 1), dtype=torch.float64, device=self.device).uniform_(float(tmin_bound), t_bound_hi)
        self.ind_bound_col_t = np.int32(0)
        self.ind_bound_col_r_min = np.int32(1)
        self.ind_bound_col_r_maxa = np.int32(2)
        self.ind_bound_col_r_maxc = np.int32(3)
        self.bound_col_pts = [self.t_bound, self.r_min_bound, self.r_maxa_bound, self.r_maxc_bound]
        self.ind_bound_col_params_deg_i0_a = np.int32(0)
        self.ind_bound_col_params_deg_ds_c = np.int32(1)
        self.bound_col_params = [self.deg_i0_a_bound, self.deg_ds_c_bound]

        t_reg_hi = float(self.firstTime) if self.gradualTime_sgd else float(self.tmax)
        self.t_reg = torch.empty((n_reg, 1), dtype=torch.float64, device=self.device).uniform_(float(tmin_reg), t_reg_hi)
        self.deg_i0_a_reg = torch.empty((n_reg, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_i0_a_min_eff"]), float(self.params["deg_i0_a_max_eff"]))
        self.deg_ds_c_reg = torch.empty((n_reg, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_ds_c_min_eff"]), float(self.params["deg_ds_c_max_eff"]))
        self.ind_reg_col_t = np.int32(0)
        self.reg_col_pts = [self.t_reg]
        self.ind_reg_col_params_deg_i0_a = np.int32(0)
        self.ind_reg_col_params_deg_ds_c = np.int32(1)
        self.reg_col_params = [self.deg_i0_a_reg, self.deg_ds_c_reg]

        if self.n_grad_path_layers is not None and self.n_grad_path_layers != 0:
            self.vprint("INFO: MAKING PATHOLOGY GRADIENT MODEL")
            self.model = GradPathPINN(self)
        elif self.hidden_units_t is not None:
            self.vprint("INFO: MAKING MERGED MODEL")
            self.model = MergedPINN(self)
        else:
            self.vprint("INFO: MAKING SPLIT MODEL")
            self.model = SplitPINN(self)
        self.model = self.model.to(self.device).double()
        n_trainable_par = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.vprint("Num trainable param = ", n_trainable_par)
        self.n_trainable_par = int(n_trainable_par)

        if self.activeData:
            self.xDataList_full = [x[: self.new_nData] for x in xDataList]
            self.x_params_dataList_full = [x[: self.new_nData] for x in x_params_dataList]
            self.yDataList_full = [y[: self.new_nData] for y in yDataList]
            ndata_orig = xDataList[0].shape[0]
            if self.new_nData < ndata_orig:
                print("WARNING: Only %.2f percent of the data will be read" % (100 * self.new_nData / ndata_orig))
                print("Adjust N_BATCH and MAX_BATCH_SIZE_DATA to accommodate %d datapoints" % ndata_orig)
        else:
            self.new_nData = self.n_batch
            self.xDataList_full = [
                np.zeros((self.n_batch, self.dim_inpt)).astype("float64") if i in [self.ind_cs_a_data, self.ind_cs_c_data] else np.zeros((self.n_batch, self.dim_inpt - 1)).astype("float64")
                for i in range(self.n_data_terms)
            ]
            self.x_params_dataList_full = [np.zeros((self.n_batch, self.dim_params)).astype("float64") for _ in range(self.n_data_terms)]
            self.yDataList_full = [np.zeros((self.n_batch, 1)).astype("float64") for _ in range(self.n_data_terms)]

        self.n_batch_lbfgs = max(int(n_batch_lbfgs), 1)
        if self.n_batch_lbfgs > self.n_batch:
            raise SystemExit("ERROR: n_batch LBFGS must be smaller or equal to SGD's")
        if self.n_batch % self.n_batch_lbfgs > 0:
            raise SystemExit("ERROR: n_batch SGD must be divisible by LBFGS's")
        self.batch_size_int_lbfgs = int(self.batch_size_int * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_bound_lbfgs = int(self.batch_size_bound * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_data_lbfgs = int(self.batch_size_data * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_reg_lbfgs = int(self.batch_size_reg * self.n_batch / self.n_batch_lbfgs)
        self.lbfgs = bool(lbfgs)
        if self.lbfgs:
            if self.gradualTime_lbfgs and self.n_gradual_steps_lbfgs > 0:
                self.gradualTimeSchedule_lbfgs = []
                if str(self.gradualTimeMode_lbfgs).lower() == "linear":
                    for istep in range(self.n_gradual_steps_lbfgs):
                        stepTime = (self.params["tmax"] - self.firstTime) / self.n_gradual_steps_lbfgs
                        new_time = np.float64(istep) * stepTime + self.firstTime
                        self.gradualTimeSchedule_lbfgs.append(min(new_time, self.params["tmax"]))
                else:
                    constantExp = -1.0
                    timeExponent = np.log(self.params["tmax"] - self.firstTime - constantExp) / float(self.n_gradual_steps_lbfgs)
                    for istep in range(self.n_gradual_steps_lbfgs):
                        new_time = constantExp + np.exp(timeExponent * istep) + self.firstTime
                        self.gradualTimeSchedule_lbfgs.append(min(new_time, self.params["tmax"]))
            self.nEpochs_start_lbfgs = int(nEpochs_start_lbfgs)
            self.nEpochs_lbfgs += self.nEpochs_start_lbfgs
            if self.nEpochs_lbfgs <= self.nEpochs_start_lbfgs:
                self.lbfgs = False
                print("WARNING: Will not use LBFGS based on number of epoch specified")
            else:
                self.vprint("n_batch_lbfgs = ", self.n_batch_lbfgs)
                self.vprint("n_epoch_lbfgs = ", self.nEpochs_lbfgs)
        self.sgd = bool(sgd)
        if self.nEpochs <= 0:
            self.sgd = False
            print("WARNING: Will not use SGD based on number of epoch specified")
        else:
            self.vprint("n_batch_sgd = ", self.n_batch)
            self.vprint("n_epoch_sgd = ", self.nEpochs)

        self.configDict = {
            "hidden_units_t": self.hidden_units_t,
            "hidden_units_t_r": self.hidden_units_t_r,
            "hidden_units_phie": self.hidden_units_phie,
            "hidden_units_phis_c": self.hidden_units_phis_c,
            "hidden_units_cs_a": self.hidden_units_cs_a,
            "hidden_units_cs_c": self.hidden_units_cs_c,
            "n_hidden_res_blocks": self.n_hidden_res_blocks,
            "n_res_block_layers": self.n_res_block_layers,
            "n_res_block_units": self.n_res_block_units,
            "n_grad_path_layers": self.n_grad_path_layers,
            "n_grad_path_units": self.n_grad_path_units,
            "hard_IC_timescale": self.hard_IC_timescale,
            "exponentialLimiter": self.exponentialLimiter,
            "dynamicAttentionWeights": self.dynamicAttentionWeights,
            "annealingWeights": self.annealingWeights,
            "linearizeJ": self.linearizeJ,
            "activation": self.activation,
            "activeInt": self.activeInt,
            "activeBound": self.activeBound,
            "activeData": self.activeData,
            "activeReg": self.activeReg,
            "params_min": self.params_min,
            "params_max": self.params_max,
            "local_utilFolder": self.local_utilFolder,
            "hnn_utilFolder": self.hnn_utilFolder,
            "hnn_modelFolder": self.hnn_modelFolder,
            "hnn_params": self.hnn_params,
            "hnntime_utilFolder": self.hnntime_utilFolder,
            "hnntime_modelFolder": self.hnntime_modelFolder,
            "hnntime_val": self.hnntime_val,
        }
        self.config = self.configDict

    def vprint(self, *kwargs):
        if self.verbose:
            print(*kwargs)

    def loadCol(self, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, w, sizes_w=None, sizes_b=None):  # compatibility stub
        raise NotImplementedError("set_weights is not used in the PyTorch port")

    def get_weights(self, model=None):  # compatibility stub
        model = self.model if model is None else model
        flat = []
        for p in model.parameters():
            flat.append(p.detach().flatten())
        return torch.cat(flat) if flat else torch.empty(0, dtype=torch.float64, device=self.device)

    # imported physics helpers
    fix_param = fix_param
    get_cs_a_hnn = get_cs_a_hnn
    get_cs_a_hnntime = get_cs_a_hnntime
    get_cs_c_hnn = get_cs_c_hnn
    get_cs_c_hnntime = get_cs_c_hnntime
    get_phie0 = get_phie0
    get_phie_hnn = get_phie_hnn
    get_phie_hnntime = get_phie_hnntime
    get_phis_c0 = get_phis_c0
    get_phis_c_hnn = get_phis_c_hnn
    get_phis_c_hnntime = get_phis_c_hnntime
    rescale_param = rescale_param
    rescaleCs_a = rescaleCs_a
    rescaleCs_c = rescaleCs_c
    rescalePhie = rescalePhie
    rescalePhis_c = rescalePhis_c
    unrescale_param = unrescale_param

    def stretchT(self, t, tmin, tmax, tminp, tmaxp):
        return (t - tmin) * (tmaxp - tminp) / (tmax - tmin) + tminp

    interior_loss = interior_loss
    boundary_loss = boundary_loss
    data_loss = data_loss
    regularization_loss = regularization_loss
    setResidualRescaling = setResidualRescaling

    def _slice_tensor(self, tensor, i_batch, batch_size):
        return tensor[i_batch * batch_size : (i_batch + 1) * batch_size]

    def _slice_array(self, arr, i_batch, batch_size):
        return arr[i_batch * batch_size : (i_batch + 1) * batch_size, :]

    def _make_batch_payload(self, i_batch: int, use_lbfgs: bool = False):
        bs_int = self.batch_size_int_lbfgs if use_lbfgs else self.batch_size_int
        bs_bound = self.batch_size_bound_lbfgs if use_lbfgs else self.batch_size_bound
        bs_data = self.batch_size_data_lbfgs if use_lbfgs else self.batch_size_data
        bs_reg = self.batch_size_reg_lbfgs if use_lbfgs else self.batch_size_reg

        int_col_pts = [self._slice_tensor(pts, i_batch, bs_int) for pts in self.int_col_pts]
        int_col_params = [self._slice_tensor(pts, i_batch, bs_int) for pts in self.int_col_params]
        bound_col_pts = [self._slice_tensor(pts, i_batch, bs_bound) for pts in self.bound_col_pts]
        bound_col_params = [self._slice_tensor(pts, i_batch, bs_bound) for pts in self.bound_col_params]
        reg_col_pts = [self._slice_tensor(pts, i_batch, bs_reg) for pts in self.reg_col_pts]
        reg_col_params = [self._slice_tensor(pts, i_batch, bs_reg) for pts in self.reg_col_params]
        x_batch_trainList = [self._slice_array(x, i_batch, bs_data) for x in self.xDataList_full[: self.ind_cs_offset_data]]
        x_cs_batch_trainList = [self._slice_array(x, i_batch, bs_data) for x in self.xDataList_full[self.ind_cs_offset_data :]]
        x_params_batch_trainList = [self._slice_array(x, i_batch, bs_data) for x in self.x_params_dataList_full]
        y_batch_trainList = [self._slice_array(y, i_batch, bs_data) for y in self.yDataList_full]
        return {
            "int_col_pts": int_col_pts,
            "int_col_params": int_col_params,
            "bound_col_pts": bound_col_pts,
            "bound_col_params": bound_col_params,
            "reg_col_pts": reg_col_pts,
            "reg_col_params": reg_col_params,
            "x_batch_trainList": x_batch_trainList,
            "x_cs_batch_trainList": x_cs_batch_trainList,
            "x_params_batch_trainList": x_params_batch_trainList,
            "y_batch_trainList": y_batch_trainList,
        }

    def _compute_loss_info(self, payload, tmax=None):
        interiorTerms = self.interior_loss(payload["int_col_pts"], payload["int_col_params"], tmax)
        boundaryTerms = self.boundary_loss(payload["bound_col_pts"], payload["bound_col_params"], tmax)
        dataTerms = self.data_loss(payload["x_batch_trainList"], payload["x_cs_batch_trainList"], payload["x_params_batch_trainList"], payload["y_batch_trainList"])
        regTerms = self.regularization_loss(payload["reg_col_pts"], tmax)
        interiorTerms_rescaled = [interiorTerm[0] * resc for (interiorTerm, resc) in zip(interiorTerms, self.interiorTerms_rescale)]
        boundaryTerms_rescaled = [boundaryTerm[0] * resc for (boundaryTerm, resc) in zip(boundaryTerms, self.boundaryTerms_rescale)]
        dataTerms_rescaled = [dataTerm[0] * resc for (dataTerm, resc) in zip(dataTerms, self.dataTerms_rescale)]
        regTerms_rescaled = [regTerm[0] * resc for (regTerm, resc) in zip(regTerms, self.regTerms_rescale)]
        loss_value, int_loss, bound_loss, data_loss_val, reg_loss = loss_fn(
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regTerms_rescaled,
            alpha=self.alpha,
        )
        return {
            "loss": loss_value,
            "int_loss": int_loss,
            "bound_loss": bound_loss,
            "data_loss": data_loss_val,
            "reg_loss": reg_loss,
            "interiorTerms": interiorTerms_rescaled,
            "boundaryTerms": boundaryTerms_rescaled,
            "dataTerms": dataTerms_rescaled,
            "regTerms": regTerms_rescaled,
        }

    def _mean_epoch_loss(self, n_batches: int, use_lbfgs: bool = False, tmax=None):
        total = 0.0
        total_int = 0.0
        total_bound = 0.0
        total_data = 0.0
        total_reg = 0.0
        for i_batch in range(n_batches):
            payload = self._make_batch_payload(i_batch, use_lbfgs=use_lbfgs)
            info = self._compute_loss_info(payload, tmax=tmax)
            total += float(info["loss"].detach().cpu())
            total_int += float(info["int_loss"].detach().cpu())
            total_bound += float(info["bound_loss"].detach().cpu())
            total_data += float(info["data_loss"].detach().cpu())
            total_reg += float(info["reg_loss"].detach().cpu())
        div = max(n_batches, 1)
        return total / div, total_int / div, total_bound / div, total_data / div, total_reg / div

    def train(
        self,
        learningRateModel,
        learningRateModelFinal,
        lrSchedulerModel,
        learningRateWeights=None,
        learningRateWeightsFinal=None,
        lrSchedulerWeights=None,
        learningRateLBFGS=None,
        inner_epochs=None,
        start_weight_training_epoch=None,
        gradient_threshold=None,
    ):
        if gradient_threshold is not None:
            print(f"INFO: clipping gradients at {gradient_threshold:.2g}")
        self.prepareLog()
        self.initLearningRateControl(learningRateModel, learningRateWeights)
        self.initLossThresholdControl(self.initialLossThreshold)
        bestLoss = None
        lr_m = float(learningRateModel)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_m)
        print("Using collocation points: " + self.collocationMode)

        final_unweighted = None
        self.run_SGD = True
        self.run_LBFGS = False
        if self.sgd:
            for epoch in range(self.nEpochs):
                self.current_stage = "SGD"
                lr_m, _ = self.dynamic_control_lrm(lr_m, epoch, lrSchedulerModel)
                for g in optimizer.param_groups:
                    g["lr"] = float(lr_m)
                if self.gradualTime_sgd:
                    if self.nEpochs > 3:
                        new_tmax = ((float(self.params["tmax"]) - float(self.params["tmin"])) * float(np.exp(self.timeIncreaseExponent * ((epoch) / (self.nEpochs // 2 - 1) - 1)))) + float(self.params["tmin"])
                        new_tmax = min(new_tmax, float(self.params["tmax"]))
                    else:
                        new_tmax = float(self.params["tmax"])
                else:
                    new_tmax = None

                epoch_loss = 0.0
                epoch_int = 0.0
                epoch_bound = 0.0
                epoch_data = 0.0
                epoch_reg = 0.0
                for step in range(self.n_batch):
                    payload = self._make_batch_payload(step, use_lbfgs=False)
                    optimizer.zero_grad(set_to_none=True)
                    info = self._compute_loss_info(payload, tmax=new_tmax)
                    info["loss"].backward()
                    if gradient_threshold is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_threshold))
                    optimizer.step()
                    epoch_loss += float(info["loss"].detach().cpu())
                    epoch_int += float(info["int_loss"].detach().cpu())
                    epoch_bound += float(info["bound_loss"].detach().cpu())
                    epoch_data += float(info["data_loss"].detach().cpu())
                    epoch_reg += float(info["reg_loss"].detach().cpu())
                    self.total_step += 1
                    self.logLosses(self.total_step, info["interiorTerms"], info["boundaryTerms"], info["dataTerms"], info["regTerms"])
                    print(
                        f"Epoch {epoch+1:4d} Batch {step+1:3d} loss {float(info['loss'].detach().cpu()):12.6g} "
                        f"iL {float(info['int_loss'].detach().cpu()):12.6g} "
                        f"bL {float(info['bound_loss'].detach().cpu()):12.6g} "
                        f"dL {float(info['data_loss'].detach().cpu()):12.6g} "
                        f"rL {float(info['reg_loss'].detach().cpu()):12.6g}"
                    )
                epoch_loss /= self.n_batch
                epoch_int /= self.n_batch
                epoch_bound /= self.n_batch
                epoch_data /= self.n_batch
                epoch_reg /= self.n_batch
                final_unweighted = get_unweighted_loss(
                    self,
                    self.int_col_pts,
                    self.int_col_params,
                    self.bound_col_pts,
                    self.bound_col_params,
                    self.reg_col_pts,
                    self.reg_col_params,
                    self.xDataList_full,
                    self.x_params_dataList_full,
                    self.yDataList_full,
                    n_batch=self.n_batch_lbfgs,
                    tmax=new_tmax,
                )
                bestLoss = self.logTraining(epoch + 1, epoch_loss, bestLoss, mse_unweighted=final_unweighted)
                print(
                    f"Epoch {epoch+1:4d} summary loss {epoch_loss:12.6g} iL {epoch_int:12.6g} bL {epoch_bound:12.6g} dL {epoch_data:12.6g} rL {epoch_reg:12.6g}"
                )

        if self.lbfgs:
            self.run_SGD = False
            self.run_LBFGS = True
            self.current_stage = "LBFGS"
            max_iter = max(int(inner_epochs or 20), 1)
            optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=float(learningRateLBFGS or learningRateModelFinal), max_iter=max_iter, line_search_fn="strong_wolfe")
            for epoch in range(self.nEpochs_lbfgs):
                if self.gradualTime_lbfgs and getattr(self, "gradualTimeSchedule_lbfgs", None):
                    idx = min(epoch // max(1, self.nEpochs_lbfgs // max(len(self.gradualTimeSchedule_lbfgs), 1)), len(self.gradualTimeSchedule_lbfgs) - 1)
                    lbfgs_tmax = float(self.gradualTimeSchedule_lbfgs[idx])
                else:
                    lbfgs_tmax = None

                def closure():
                    optimizer_lbfgs.zero_grad(set_to_none=True)
                    total_loss = torch.zeros(1, dtype=torch.float64, device=self.device).squeeze()
                    for i_batch in range(self.n_batch_lbfgs):
                        payload = self._make_batch_payload(i_batch, use_lbfgs=True)
                        info = self._compute_loss_info(payload, tmax=lbfgs_tmax)
                        total_loss = total_loss + info["loss"]
                    total_loss = total_loss / float(self.n_batch_lbfgs)
                    total_loss.backward()
                    if gradient_threshold is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_threshold))
                    return total_loss

                optimizer_lbfgs.step(closure)
                epoch_loss, epoch_int, epoch_bound, epoch_data, epoch_reg = self._mean_epoch_loss(self.n_batch_lbfgs, use_lbfgs=True, tmax=lbfgs_tmax)
                final_unweighted = get_unweighted_loss(
                    self,
                    self.int_col_pts,
                    self.int_col_params,
                    self.bound_col_pts,
                    self.bound_col_params,
                    self.reg_col_pts,
                    self.reg_col_params,
                    self.xDataList_full,
                    self.x_params_dataList_full,
                    self.yDataList_full,
                    n_batch=self.n_batch_lbfgs,
                    tmax=lbfgs_tmax,
                )
                bestLoss = self.logTraining(self.nEpochs + epoch + 1, epoch_loss, bestLoss, mse_unweighted=final_unweighted)
                print(
                    f"LBFGS Epoch {epoch+1:4d} summary loss {epoch_loss:12.6g} iL {epoch_int:12.6g} bL {epoch_bound:12.6g} dL {epoch_data:12.6g} rL {epoch_reg:12.6g}"
                )

        if final_unweighted is None:
            final_unweighted = 0.0
        return final_unweighted

    def prepareLog(self):
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.logLossFolder, exist_ok=True)
        for name in [
            os.path.join(self.modelFolder, "config.json"),
            os.path.join(self.logLossFolder, "log.csv"),
            os.path.join(self.logLossFolder, "interiorTerms.csv"),
            os.path.join(self.logLossFolder, "boundaryTerms.csv"),
            os.path.join(self.logLossFolder, "dataTerms.csv"),
            os.path.join(self.logLossFolder, "regTerms.csv"),
        ]:
            try:
                os.remove(name)
            except FileNotFoundError:
                pass
        with open(os.path.join(self.modelFolder, "config.json"), "w+", encoding="utf-8") as outfile:
            json.dump(self._json_safe(self.configDict), outfile, indent=4, sort_keys=True)
        with open(os.path.join(self.logLossFolder, "log.csv"), "a+", encoding="utf-8") as f:
            f.write("epoch;step;mseloss\n")
        for name in ["interiorTerms.csv", "boundaryTerms.csv", "dataTerms.csv", "regTerms.csv"]:
            with open(os.path.join(self.logLossFolder, name), "a+", encoding="utf-8") as f:
                f.write("step;lossArray\n")

    def _json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    def initLearningRateControl(self, learningRateModel, learningRateWeights=None):
        with open(os.path.join(self.modelFolder, "learningRateModel"), "w+", encoding="utf-8") as f:
            f.write(str(learningRateModel))
        self.learningRateModelStamp = os.stat(os.path.join(self.modelFolder, "learningRateModel")).st_mtime
        if self.dynamicAttentionWeights:
            with open(os.path.join(self.modelFolder, "learningRateWeights"), "w+", encoding="utf-8") as f:
                f.write(str(learningRateWeights))
            self.learningRateWeightsStamp = os.stat(os.path.join(self.modelFolder, "learningRateWeights")).st_mtime

    def initLossThresholdControl(self, lossThreshold):
        with open(os.path.join(self.modelFolder, "lossThreshold"), "w+", encoding="utf-8") as f:
            f.write(str(lossThreshold))
        self.lossThresholdStamp = os.stat(os.path.join(self.modelFolder, "lossThreshold")).st_mtime

    def update_param_file(self, param, param_old, mode):
        if mode == "lr_m":
            filename = os.path.join(self.modelFolder, "learningRateModel")
        elif mode == "lr_w":
            filename = os.path.join(self.modelFolder, "learningRateWeights")
        else:
            filename = os.path.join(self.modelFolder, "lossThreshold")
        if abs(float(param) - float(param_old)) > 1e-15:
            with open(filename, "w+", encoding="utf-8") as f:
                f.write(str(param))
            stamp = os.stat(filename).st_mtime
            if mode == "lr_m":
                self.learningRateModelStamp = stamp
            elif mode == "lr_w":
                self.learningRateWeightsStamp = stamp
            else:
                self.lossThresholdStamp = stamp

    def read_param_file(self, param, param_old, mode):
        if mode == "lr_m":
            has_changed = self.lrmHasChanged()
            filename = os.path.join(self.modelFolder, "learningRateModel")
            prefix_info = "INFO: LR Model changed from"
            prefix_warning = "WARNING: LR Model has changed but could not be updated. Using LR"
        elif mode == "lr_w":
            has_changed = self.lrwHasChanged()
            filename = os.path.join(self.modelFolder, "learningRateWeights")
            prefix_info = "INFO: LR Weights changed from"
            prefix_warning = "WARNING: LR Weights has changed but could not be updated. Using LR"
        else:
            has_changed = self.ltHasChanged()
            filename = os.path.join(self.modelFolder, "lossThreshold")
            prefix_info = "INFO: LT changed from"
            prefix_warning = "WARNING: LT has changed but could not be updated. Using LT"
        if has_changed:
            param_old = param
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                param = np.float64(lines[0])
                if abs(float(param_old) - float(param)) > 1e-12:
                    print("\n" + prefix_info + " %.3g to %.3g\n" % (param_old, param))
            except Exception:
                print("\n" + prefix_warning + " = %.3g\n" % float(param_old))
            stamp = os.stat(filename).st_mtime
            if mode == "lr_m":
                self.learningRateModelStamp = stamp
            elif mode == "lr_w":
                self.learningRateWeightsStamp = stamp
            else:
                self.lossThresholdStamp = stamp
        return float(param), float(param_old)

    def force_decrease_lrm(self, lr_m, learningRateModelFinal):
        lr_m = max(float(lr_m) * 0.5, float(learningRateModelFinal))
        with open(os.path.join(self.modelFolder, "learningRateModel"), "w+", encoding="utf-8") as f:
            f.write(str(lr_m))

    def force_decrease_lrm_lrw(self, lr_m, learningRateModelFinal, lr_w, learningRateWeightsFinal):
        self.force_decrease_lrm(lr_m, learningRateModelFinal)
        lr_w = max(float(lr_w) * 0.5, float(learningRateWeightsFinal))
        with open(os.path.join(self.modelFolder, "learningRateWeights"), "w+", encoding="utf-8") as f:
            f.write(str(lr_w))

    def dynamic_control_lrm(self, lr_m, epoch, scheduler):
        lr_m_old = lr_m
        lr_m = scheduler(epoch, self.lr_m_epoch_start) if hasattr(self, "lr_m_epoch_start") else scheduler(epoch, lr_m)
        if (self.total_step % self.freq) == 0:
            lr_m, lr_m_old = self.read_param_file(lr_m, lr_m_old, mode="lr_m")
            self.update_param_file(lr_m, lr_m_old, mode="lr_m")
        self.lr_m_epoch_start = lr_m
        return float(lr_m), float(lr_m_old)

    def dynamic_control_lrw(self, lr_w, epoch, scheduler):
        lr_w_old = lr_w
        lr_w = scheduler(epoch, self.lr_w_epoch_start) if hasattr(self, "lr_w_epoch_start") else scheduler(epoch, lr_w)
        if (self.total_step % self.freq) == 0:
            lr_w, lr_w_old = self.read_param_file(lr_w, lr_w_old, mode="lr_w")
            self.update_param_file(lr_w, lr_w_old, mode="lr_w")
        self.lr_w_epoch_start = lr_w
        return float(lr_w), float(lr_w_old)

    def dynamic_control_lt(self, lt):
        lt_old = lt
        if (self.total_step % self.freq) == 0:
            lt, lt_old = self.read_param_file(lt, lt_old, mode="lt")
            self.update_param_file(lt, lt, mode="lt")
        return float(lt), float(lt_old)

    def lrmHasChanged(self):
        return abs(os.stat(os.path.join(self.modelFolder, "learningRateModel")).st_mtime - self.learningRateModelStamp) > 1e-6

    def lrwHasChanged(self):
        return abs(os.stat(os.path.join(self.modelFolder, "learningRateWeights")).st_mtime - self.learningRateWeightsStamp) > 1e-6

    def ltHasChanged(self):
        return abs(os.stat(os.path.join(self.modelFolder, "lossThreshold")).st_mtime - self.lossThresholdStamp) > 1e-6

    def logTraining(self, epoch, mse, bestLoss, mse_unweighted=None):
        with open(os.path.join(self.logLossFolder, "log.csv"), "a+", encoding="utf-8") as f:
            if mse_unweighted is None:
                f.write(f"{int(epoch)};{int(epoch * self.n_batch)};{float(mse)}\n")
            else:
                f.write(f"{int(epoch)};{int(epoch * self.n_batch)};{float(mse)};{float(mse_unweighted)}\n")
        epochLoss = float(mse_unweighted if mse_unweighted is not None else mse)
        safe_save_state_dict(self.model, os.path.join(self.modelFolder, "last.pt"))
        if self.current_stage.upper() == "SGD":
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "lastSGD.pt"))
        else:
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "lastLBFGS.pt"))
        if bestLoss is None or epochLoss < float(bestLoss):
            bestLoss = epochLoss
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "best.pt"))
        return bestLoss

    def logLosses(self, step, interiorTerms, boundaryTerms, dataTerms, regTerms):
        if step % self.freq != 0:
            return
        if self.activeInt:
            interiorTermsArray = [float(torch.mean(term.square()).detach().cpu()) for term in interiorTerms]
        else:
            interiorTermsArray = []
        interiorTermsArrayPercent = [round(term / (1e-16 + sum(interiorTermsArray)), 2) for term in interiorTermsArray] if interiorTermsArray else []
        with open(os.path.join(self.logLossFolder, "interiorTerms.csv"), "a+", encoding="utf-8") as f:
            f.write(f"{int(step)};{interiorTermsArrayPercent}\n")
            f.write(f"{int(step)};{interiorTermsArray}\n")
        if self.activeBound:
            boundaryTermsArray = [float(torch.mean(term.square()).detach().cpu()) for term in boundaryTerms]
        else:
            boundaryTermsArray = []
        boundaryTermsArrayPercent = [round(term / (1e-16 + sum(boundaryTermsArray)), 2) for term in boundaryTermsArray] if boundaryTermsArray else []
        with open(os.path.join(self.logLossFolder, "boundaryTerms.csv"), "a+", encoding="utf-8") as f:
            f.write(f"{int(step)};{boundaryTermsArrayPercent}\n")
            f.write(f"{int(step)};{boundaryTermsArray}\n")
        if self.activeData:
            dataTermsArray = [float(torch.mean(term.square()).detach().cpu()) for term in dataTerms]
        else:
            dataTermsArray = []
        dataTermsArrayPercent = [round(term / (1e-16 + sum(dataTermsArray)), 2) for term in dataTermsArray] if dataTermsArray else []
        with open(os.path.join(self.logLossFolder, "dataTerms.csv"), "a+", encoding="utf-8") as f:
            f.write(f"{int(step)};{dataTermsArrayPercent}\n")
            f.write(f"{int(step)};{dataTermsArray}\n")
        if self.activeReg:
            regTermsArray = [float(torch.mean(term.square()).detach().cpu()) for term in regTerms]
        else:
            regTermsArray = []
        regTermsArrayPercent = [round(term / (1e-16 + sum(regTermsArray)), 2) for term in regTermsArray] if regTermsArray else []
        with open(os.path.join(self.logLossFolder, "regTerms.csv"), "a+", encoding="utf-8") as f:
            f.write(f"{int(step)};{regTermsArrayPercent}\n")
            f.write(f"{int(step)};{regTermsArray}\n")
