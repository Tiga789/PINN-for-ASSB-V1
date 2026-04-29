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


# -----------------------------------------------------------------------------
# Generic tensor / parameter helpers
# -----------------------------------------------------------------------------


def _to_tensor(x, like: torch.Tensor | None = None) -> torch.Tensor:
    """Convert scalar/array/tensor inputs to a 2D torch tensor.

    The translated PINNSTRIPES code calls these rescale functions from many
    locations.  Keeping this helper permissive avoids shape/device/dtype
    mismatches when evaluating data, residuals, HNN levels, or checkpoints.
    """
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


def _param_float(params: dict, names: tuple[str, ...], default: float) -> float:
    """Read the first available scalar parameter name from params."""
    for name in names:
        if name in params:
            value = params[name]
            try:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().reshape(-1)[0].cpu())
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size:
                    return float(arr[0])
            except Exception:
                try:
                    return float(value)
                except Exception:
                    pass
    return float(default)


def _param_bool(params: dict, names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        if name in params:
            value = params[name]
            if isinstance(value, str):
                return value.strip().lower() not in {"false", "0", "no", "off", "none"}
            return bool(value)
    return bool(default)


# -----------------------------------------------------------------------------
# Radial and concentration rescale policy
# -----------------------------------------------------------------------------


def _use_per_electrode_radial_rescale_from_params(params: dict) -> bool:
    """Whether r should be normalized by Rs_a/Rs_c instead of one shared scale.

    For ASSB, Rs_a=50 um and Rs_c=1.8 um.  A shared rescale_R=max(Rs_a,Rs_c)
    compresses cathode inputs into only ~0--0.036, which makes learning the
    cathode surface-gradient dynamics unnecessarily hard.  The new thermo file
    sets use_per_electrode_rescale_R=True and provides rescale_R_a/rescale_R_c.

    For old HNN checkpoints without the new keys, this function falls back to
    legacy shared rescale_R to avoid silently changing old hierarchy levels.
    """
    if any(k in params for k in ("use_per_electrode_rescale_R", "use_per_electrode_radial_rescale")):
        return _param_bool(params, ("use_per_electrode_rescale_R", "use_per_electrode_radial_rescale"), True)

    mode = str(params.get("radial_rescale_mode", "")).strip().lower()
    if mode in {"per_electrode", "electrode", "separate", "separated", "assb"}:
        return True

    return any(k in params for k in ("rescale_R_a", "rescale_R_c", "rescale_R_negative", "rescale_R_positive"))


def _radial_rescale_from_params(params: dict, electrode: str) -> float:
    """Return the physical radius scale used to normalize r for a branch."""
    electrode = electrode.lower()
    if not _use_per_electrode_radial_rescale_from_params(params):
        return _param_float(params, ("rescale_R",), 1.0)

    if electrode.startswith("a") or electrode.startswith("n"):
        return _param_float(params, ("rescale_R_a", "rescale_R_negative", "Rs_a", "rescale_R"), 1.0)
    return _param_float(params, ("rescale_R_c", "rescale_R_positive", "Rs_c", "rescale_R"), 1.0)


def _radial_rescale_from_nn(nn_obj, electrode: str) -> float:
    """Return r-normalization scale for an HNN/HNN-time object."""
    params = getattr(nn_obj, "params", {}) or {}
    return _radial_rescale_from_params(params, electrode)


def _concentration_rescale_mode(self) -> str:
    """Return concentration rescaling mode.

    cycle/bidirectional:
        Allows solid concentration to increase or decrease from the initial
        value.  This is the correct mode for the user's cycle5 and cycles5plus
        charge-discharge data.

    discharge:
        Keeps the original PINNSTRIPES discharge-only monotonic constraint.
    """
    mode = self.params.get("concentration_rescale_mode", self.params.get("cs_rescale_mode", "cycle"))
    mode = str(mode).strip().lower()
    if mode in {"charge_discharge", "cycle", "cyclic", "bidirectional", "both"}:
        return "cycle"
    if mode in {"discharge", "discharge_only", "original"}:
        return "discharge"
    return "cycle"


def _concentration_bounds(self, electrode: str, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lower/upper concentration bounds for an electrode.

    The new ASSB thermo file writes narrower theta/cs bounds, e.g. roughly
    theta_a=0.40--0.90 and theta_c=0.40--0.92.  Those bounds are used here when
    available.  The initial concentration is always included so that strict IC
    enforcement cannot be clipped out by an accidentally tight user bound.
    """
    electrode = electrode.lower()
    if electrode in {"a", "an", "anode", "negative", "n"}:
        csmax = _param_float(self.params, ("csanmax", "cs_a_max", "cs_n_max"), 1.0)
        theta_min = _param_float(self.params, ("theta_a_min", "theta_n_min"), 0.0)
        theta_max = _param_float(self.params, ("theta_a_max", "theta_n_max"), 1.0)
        lower = _param_float(self.params, ("cs_a_min", "cs_n_min", "csanmin"), theta_min * csmax)
        upper = _param_float(self.params, ("cs_a_upper", "cs_a_max_bound", "cs_n_upper"), theta_max * csmax)
        start = _param_float(self.params, ("cs_a0", "csan0", "cs_n0"), 0.5 * (lower + upper))
    elif electrode in {"c", "ca", "cathode", "positive", "p"}:
        csmax = _param_float(self.params, ("cscamax", "cs_c_max", "cs_p_max"), 1.0)
        theta_min = _param_float(self.params, ("theta_c_min", "theta_p_min"), 0.0)
        theta_max = _param_float(self.params, ("theta_c_max", "theta_p_max"), 1.0)
        lower = _param_float(self.params, ("cs_c_min", "cs_p_min", "cscamin"), theta_min * csmax)
        upper = _param_float(self.params, ("cs_c_upper", "cs_c_max_bound", "cs_p_upper"), theta_max * csmax)
        start = _param_float(self.params, ("cs_c0", "csca0", "cs_p0"), 0.5 * (lower + upper))
    else:
        raise ValueError(f"Unknown electrode label: {electrode}")

    lower_raw = float(lower)
    upper_raw = float(upper)
    start_raw = float(start)
    lower = min(lower_raw, upper_raw, start_raw)
    upper = max(lower_raw, upper_raw, start_raw)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = 0.0, max(float(csmax), 1.0)

    lower_t = torch.full_like(like, lower)
    upper_t = torch.full_like(like, upper)
    return lower_t, upper_t


def _asymmetric_bounded_delta(raw: torch.Tensor, center: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Smoothly map raw NN output to [lower, upper] around center.

    Compared with a symmetric span followed by clamp, this avoids a cathode
    branch near the upper SOC endpoint immediately saturating on the small side
    while still allowing a large decrease over the rest of the cycle.
    """
    z = torch.tanh(raw)
    pos_span = torch.clamp(upper - center, min=np.float64(1e-12))
    neg_span = torch.clamp(center - lower, min=np.float64(1e-12))
    delta = torch.where(z >= 0.0, pos_span * z, neg_span * z)
    return center + delta


def _cycle_concentration_target(
    self,
    raw: torch.Tensor,
    start: torch.Tensor,
    electrode: str,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bidirectional concentration map for charge-discharge cycles.

    raw=0 gives target=start when no HNN base is supplied.  This keeps strict
    initial-condition behavior while allowing either electrode concentration to
    move upward or downward during the same cycle.
    """
    raw = _to_tensor(raw, start)
    lower, upper = _concentration_bounds(self, electrode, start)

    if base is None:
        center = torch.clamp(start, lower, upper)
        target = _asymmetric_bounded_delta(raw, center, lower, upper)
    else:
        base = torch.clamp(_to_tensor(base, start), lower, upper)
        frac = _param_float(
            self.params,
            ("hnn_cycle_correction_fraction", "cycle_hnn_correction_fraction"),
            0.25,
        )
        # For HNN levels, apply a bounded local correction around the base.
        z = torch.tanh(raw)
        pos_span = torch.clamp(upper - base, min=np.float64(1e-12)) * float(frac)
        neg_span = torch.clamp(base - lower, min=np.float64(1e-12)) * float(frac)
        target = base + torch.where(z >= 0.0, pos_span * z, neg_span * z)

    return torch.clamp(target, lower, upper)


def _discharge_concentration_target(
    self,
    raw: torch.Tensor,
    start: torch.Tensor,
    electrode: str,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original discharge-only concentration map kept for optional fallback."""
    raw = _to_tensor(raw, start)
    lower, upper = _concentration_bounds(self, electrode, start)

    if base is not None:
        base = torch.clamp(_to_tensor(base, start), lower, upper)
        if electrode.lower().startswith(("a", "n")):
            resc = torch.clamp(base - lower, min=np.float64(1e-12))
            target = base - 0.01 * resc * torch.sigmoid(raw)
        else:
            resc = torch.clamp(upper - base, min=np.float64(1e-12))
            target = base + 0.01 * resc * torch.sigmoid(raw)
        return torch.clamp(target, lower, upper)

    if electrode.lower().startswith(("a", "n")):
        # Discharge: negative electrode Li concentration decreases.
        target = start - torch.clamp(start - lower, min=np.float64(1e-12)) * torch.sigmoid(raw)
    else:
        # Discharge: positive electrode Li concentration increases.
        target = start + torch.clamp(upper - start, min=np.float64(1e-12)) * torch.sigmoid(raw)
    return torch.clamp(target, lower, upper)


# -----------------------------------------------------------------------------
# Public rescale functions attached to myNN
# -----------------------------------------------------------------------------


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

    base = None
    if self.use_hnn:
        base = self.get_cs_a_hnn(t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)

    if _concentration_rescale_mode(self) == "discharge":
        target = _discharge_concentration_target(self, cs_a, cs_a_start, "a", base=base)
    else:
        target = _cycle_concentration_target(self, cs_a, cs_a_start, "a", base=base)

    out = (target - cs_a_start) * timeDistance + cs_a_start
    if clip:
        lower, upper = _concentration_bounds(self, "a", out)
        out = torch.clamp(out, lower, upper)
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

    base = None
    if self.use_hnn:
        base = self.get_cs_c_hnn(t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape)

    if _concentration_rescale_mode(self) == "discharge":
        target = _discharge_concentration_target(self, cs_c, cs_c_start, "c", base=base)
    else:
        target = _cycle_concentration_target(self, cs_c, cs_c_start, "c", base=base)

    out = (target - cs_c_start) * timeDistance + cs_c_start
    if clip:
        lower, upper = _concentration_bounds(self, "c", out)
        out = torch.clamp(out, lower, upper)
    return out


# -----------------------------------------------------------------------------
# Initial-condition helpers
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# HNN helpers
# -----------------------------------------------------------------------------


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

    r_scale = _radial_rescale_from_nn(self.hnn, "a")
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            r / float(r_scale),
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
    r_scale = _radial_rescale_from_nn(self.hnntime, "a")
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            r / float(r_scale),
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

    r_scale = _radial_rescale_from_nn(self.hnn, "c")
    out = self.hnn.model(
        [
            t / float(self.hnn.params["rescale_T"]),
            r / float(r_scale),
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
    r_scale = _radial_rescale_from_nn(self.hnntime, "c")
    out = self.hnntime.model(
        [
            t / float(self.hnntime.params["rescale_T"]),
            r / float(r_scale),
            self.hnntime.rescale_param(deg_i0_a, self.hnntime.ind_deg_i0_a),
            self.hnntime.rescale_param(deg_ds_c, self.hnntime.ind_deg_ds_c),
        ],
        training=False,
    )[self.hnntime.ind_cs_c]
    return self.hnntime.rescaleCs_c(out, t, r, deg_i0_a, deg_ds_c)


# -----------------------------------------------------------------------------
# Parameter rescaling helpers
# -----------------------------------------------------------------------------


def rescale_param(self, param, ind):
    param = _to_tensor(param)
    return (param - float(self.params_min[ind])) / float(self.resc_params[ind])


def fix_param(self, param, param_val):
    param = _to_tensor(param)
    return float(param_val) * torch.ones_like(param)


def unrescale_param(self, param_rescaled, ind):
    param_rescaled = _to_tensor(param_rescaled)
    return param_rescaled * float(self.resc_params[ind]) + float(self.params_min[ind])
