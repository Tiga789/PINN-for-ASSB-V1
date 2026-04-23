from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))


def _to_tensor(x, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
        if like is not None:
            out = out.to(dtype=like.dtype, device=like.device)
        else:
            out = out.to(dtype=torch.float64)
    else:
        dtype = torch.float64 if like is None else like.dtype
        device = None if like is None else like.device
        out = torch.as_tensor(x, dtype=dtype, device=device)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _ones_like(x) -> torch.Tensor:
    return torch.ones_like(_to_tensor(x))


def rescalePhie(self, phie, t, deg_i0_a, deg_ds_c):
    resc_phie = float(self.params["rescale_phie"])
    phie = _to_tensor(phie)
    t_reshape = _to_tensor(t, phie)
    deg_i0_a_reshape = _to_tensor(deg_i0_a, phie)
    deg_ds_c_reshape = _to_tensor(deg_ds_c, phie)

    if self.use_hnntime:
        phie_start = self.get_phie_hnntime(deg_i0_a_reshape, deg_ds_c_reshape)
        timeDistance = 1.0 - torch.exp(
            -(t_reshape - float(self.hnntime_val)) / float(self.hard_IC_timescale)
        )
    else:
        phie_start = self.get_phie0(deg_i0_a_reshape)
        timeDistance = 1.0 - torch.exp(-(t_reshape) / float(self.hard_IC_timescale))

    offset = torch.zeros_like(phie)
    phie_nn = phie
    if self.use_hnn:
        phie_hnn = self.get_phie_hnn(t_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        offset = phie_hnn - phie_start
        resc_phie *= 0.1

    return (resc_phie * phie_nn + offset) * timeDistance + phie_start


def rescalePhis_c(self, phis_c, t, deg_i0_a, deg_ds_c):
    resc_phis_c = float(self.params["rescale_phis_c"])
    phis_c = _to_tensor(phis_c)
    t_reshape = _to_tensor(t, phis_c)
    deg_i0_a_reshape = _to_tensor(deg_i0_a, phis_c)
    deg_ds_c_reshape = _to_tensor(deg_ds_c, phis_c)

    if self.use_hnntime:
        phis_c_start = self.get_phis_c_hnntime(deg_i0_a_reshape, deg_ds_c_reshape)
        timeDistance = 1.0 - torch.exp(
            -(t_reshape - float(self.hnntime_val)) / float(self.hard_IC_timescale)
        )
    else:
        phis_c_start = self.get_phis_c0(deg_i0_a_reshape)
        timeDistance = 1.0 - torch.exp(-(t_reshape) / float(self.hard_IC_timescale))

    offset = torch.zeros_like(phis_c)
    phis_c_nn = phis_c
    if self.use_hnn:
        phis_c_hnn = self.get_phis_c_hnn(t_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        offset = phis_c_hnn - phis_c_start
        resc_phis_c *= 0.1

    return (resc_phis_c * phis_c_nn + offset) * timeDistance + phis_c_start


def rescaleCs_a(self, cs_a, t, r, deg_i0_a, deg_ds_c, clip: bool = True):
    cs_a = _to_tensor(cs_a)
    t_reshape = _to_tensor(t, cs_a)
    r_reshape = _to_tensor(r, cs_a)
    deg_i0_a_reshape = _to_tensor(deg_i0_a, cs_a)
    deg_ds_c_reshape = _to_tensor(deg_ds_c, cs_a)

    if self.use_hnntime:
        cs_a_start = self.get_cs_a_hnntime(r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        timeDistance = 1.0 - torch.exp(
            -(t_reshape - float(self.hnntime_val)) / float(self.hard_IC_timescale)
        )
    else:
        cs_a_start = torch.full_like(cs_a, float(self.cs_a0))
        timeDistance = 1.0 - torch.exp(-(t_reshape) / float(self.hard_IC_timescale))
    resc_cs_a = -cs_a_start

    offset = torch.zeros_like(cs_a)
    if self.use_hnn:
        cs_a_hnn = self.get_cs_a_hnn(t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        offset = cs_a_hnn - cs_a_start
        cs_a_nn = cs_a * 0.01
    else:
        cs_a_nn = torch.sigmoid(cs_a)

    out = (resc_cs_a * cs_a_nn + offset) * timeDistance + cs_a_start
    if clip:
        out = torch.clamp(out, 0.0, float(self.params["csanmax"]))
    return out


def rescaleCs_c(self, cs_c, t, r, deg_i0_a, deg_ds_c, clip: bool = True):
    cs_c = _to_tensor(cs_c)
    t_reshape = _to_tensor(t, cs_c)
    r_reshape = _to_tensor(r, cs_c)
    deg_i0_a_reshape = _to_tensor(deg_i0_a, cs_c)
    deg_ds_c_reshape = _to_tensor(deg_ds_c, cs_c)

    if self.use_hnntime:
        cs_c_start = self.get_cs_c_hnntime(r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        timeDistance = 1.0 - torch.exp(
            -(t_reshape - float(self.hnntime_val)) / float(self.hard_IC_timescale)
        )
    else:
        cs_c_start = torch.full_like(cs_c, float(self.cs_c0))
        timeDistance = 1.0 - torch.exp(-(t_reshape) / float(self.hard_IC_timescale))
    resc_cs_c = float(self.params["cscamax"]) - cs_c_start

    offset = torch.zeros_like(cs_c)
    if self.use_hnn:
        cs_c_hnn = self.get_cs_c_hnn(t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)
        offset = cs_c_hnn - cs_c_start
        cs_c_nn = cs_c * 0.01
    else:
        cs_c_nn = torch.sigmoid(cs_c)

    out = (resc_cs_c * cs_c_nn + offset) * timeDistance + cs_c_start
    if clip:
        out = torch.clamp(out, 0.0, float(self.params["cscamax"]))
    return out


def get_phie0(self, deg_i0_a):
    deg_i0_a = _to_tensor(deg_i0_a)
    i0_a = self.params["i0_a"](
        float(self.params["cs_a0"]) * _ones_like(deg_i0_a),
        float(self.params["ce0"]) * _ones_like(deg_i0_a),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    return self.params["phie0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
    )


def get_phis_c0(self, deg_i0_a):
    deg_i0_a = _to_tensor(deg_i0_a)
    i0_a = self.params["i0_a"](
        float(self.params["cs_a0"]) * _ones_like(deg_i0_a),
        float(self.params["ce0"]) * _ones_like(deg_i0_a),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    return self.params["phis_c0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
        self.params["j_c"],
        self.params["i0_c0"],
        self.params["Uocp_c0"],
    )


# HNN helpers

def get_phie_hnn(self, t, deg_i0_a, deg_ds_c):
    t = _to_tensor(t)
    deg_i0_a = _to_tensor(deg_i0_a, t)
    deg_ds_c = _to_tensor(deg_ds_c, t)
    if self.hnn_params is not None:
        deg_i0_a_eff = self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a])
        deg_ds_c_eff = self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c])
    else:
        deg_i0_a_eff = deg_i0_a
        deg_ds_c_eff = deg_ds_c
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            torch.zeros_like(t),
            self.hnn.rescale_param(deg_i0_a_eff, self.hnn.ind_deg_i0_a),
            self.hnn.rescale_param(deg_ds_c_eff, self.hnn.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnn.ind_phie]
    return self.hnn.rescalePhie(out, t, deg_i0_a_eff, deg_ds_c_eff)


def get_phie_hnntime(self, deg_i0_a, deg_ds_c):
    deg_i0_a = _to_tensor(deg_i0_a)
    deg_ds_c = _to_tensor(deg_ds_c, deg_i0_a)
    t = torch.full_like(deg_i0_a, float(self.hnntime_val))
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            torch.zeros_like(t),
            self.hnntime.rescale_param(deg_i0_a, self.hnntime.ind_deg_i0_a),
            self.hnntime.rescale_param(deg_ds_c, self.hnntime.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnntime.ind_phie]
    return self.hnntime.rescalePhie(out, t, deg_i0_a, deg_ds_c)


def get_phis_c_hnn(self, t, deg_i0_a, deg_ds_c):
    t = _to_tensor(t)
    deg_i0_a = _to_tensor(deg_i0_a, t)
    deg_ds_c = _to_tensor(deg_ds_c, t)
    if self.hnn_params is not None:
        deg_i0_a_eff = self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a])
        deg_ds_c_eff = self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c])
    else:
        deg_i0_a_eff = deg_i0_a
        deg_ds_c_eff = deg_ds_c
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            torch.zeros_like(t),
            self.hnn.rescale_param(deg_i0_a_eff, self.hnn.ind_deg_i0_a),
            self.hnn.rescale_param(deg_ds_c_eff, self.hnn.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnn.ind_phis_c]
    return self.hnn.rescalePhis_c(out, t, deg_i0_a_eff, deg_ds_c_eff)


def get_phis_c_hnntime(self, deg_i0_a, deg_ds_c):
    deg_i0_a = _to_tensor(deg_i0_a)
    deg_ds_c = _to_tensor(deg_ds_c, deg_i0_a)
    t = torch.full_like(deg_i0_a, float(self.hnntime_val))
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            torch.zeros_like(t),
            self.hnntime.rescale_param(deg_i0_a, self.hnntime.ind_deg_i0_a),
            self.hnntime.rescale_param(deg_ds_c, self.hnntime.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnntime.ind_phis_c]
    return self.hnntime.rescalePhis_c(out, t, deg_i0_a, deg_ds_c)


def get_cs_a_hnn(self, t, r, deg_i0_a, deg_ds_c):
    t = _to_tensor(t)
    r = _to_tensor(r, t)
    deg_i0_a = _to_tensor(deg_i0_a, t)
    deg_ds_c = _to_tensor(deg_ds_c, t)
    if self.hnn_params is not None:
        deg_i0_a_eff = self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a])
        deg_ds_c_eff = self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c])
    else:
        deg_i0_a_eff = deg_i0_a
        deg_ds_c_eff = deg_ds_c
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            r / float(self.hnn.params["rescale_R"]),
            self.hnn.rescale_param(deg_i0_a_eff, self.hnn.ind_deg_i0_a),
            self.hnn.rescale_param(deg_ds_c_eff, self.hnn.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnn.ind_cs_a]
    return self.hnn.rescaleCs_a(out, t, r, deg_i0_a_eff, deg_ds_c_eff)


def get_cs_a_hnntime(self, r, deg_i0_a, deg_ds_c):
    r = _to_tensor(r)
    deg_i0_a = _to_tensor(deg_i0_a, r)
    deg_ds_c = _to_tensor(deg_ds_c, r)
    t = torch.full_like(deg_i0_a, float(self.hnntime_val))
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            r / float(self.hnntime.params["rescale_R"]),
            self.hnntime.rescale_param(deg_i0_a, self.hnntime.ind_deg_i0_a),
            self.hnntime.rescale_param(deg_ds_c, self.hnntime.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnntime.ind_cs_a]
    return self.hnntime.rescaleCs_a(out, t, r, deg_i0_a, deg_ds_c)


def get_cs_c_hnn(self, t, r, deg_i0_a, deg_ds_c):
    t = _to_tensor(t)
    r = _to_tensor(r, t)
    deg_i0_a = _to_tensor(deg_i0_a, t)
    deg_ds_c = _to_tensor(deg_ds_c, t)
    if self.hnn_params is not None:
        deg_i0_a_eff = self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a])
        deg_ds_c_eff = self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c])
    else:
        deg_i0_a_eff = deg_i0_a
        deg_ds_c_eff = deg_ds_c
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            r / float(self.hnn.params["rescale_R"]),
            self.hnn.rescale_param(deg_i0_a_eff, self.hnn.ind_deg_i0_a),
            self.hnn.rescale_param(deg_ds_c_eff, self.hnn.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnn.ind_cs_c]
    return self.hnn.rescaleCs_c(out, t, r, deg_i0_a_eff, deg_ds_c_eff)


def get_cs_c_hnntime(self, r, deg_i0_a, deg_ds_c):
    r = _to_tensor(r)
    deg_i0_a = _to_tensor(deg_i0_a, r)
    deg_ds_c = _to_tensor(deg_ds_c, r)
    t = torch.full_like(deg_i0_a, float(self.hnntime_val))
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            r / float(self.hnntime.params["rescale_R"]),
            self.hnntime.rescale_param(deg_i0_a, self.hnntime.ind_deg_i0_a),
            self.hnntime.rescale_param(deg_ds_c, self.hnntime.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnntime.ind_cs_c]
    return self.hnntime.rescaleCs_c(out, t, r, deg_i0_a, deg_ds_c)


def rescale_param(self, param, ind):
    param = _to_tensor(param)
    return (param - float(self.params_min[ind])) / float(self.resc_params[ind])


def fix_param(self, param, param_val):
    param = _to_tensor(param)
    return float(param_val) * torch.ones_like(param)


def unrescale_param(self, param_rescaled, ind):
    param_rescaled = _to_tensor(param_rescaled)
    return param_rescaled * float(self.resc_params[ind]) + float(self.params_min[ind])
