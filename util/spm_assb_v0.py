from __future__ import annotations

import numpy as np

try:
    from .thermo_assb import (
        infer_cs_c0_from_fullcell_voltage,
        infer_theta_a0_from_theta_c,
        infer_theta_c_from_fullcell_voltage,
        setParams,
        uocp_a_fun,
        uocp_c_fun,
        i0_a_degradation_param_fun,
        i0_c_fun,
        ds_a_fun,
        ds_c_degradation_param_fun,
    )
except ImportError:  # pragma: no cover
    from thermo_assb import (
        infer_cs_c0_from_fullcell_voltage,
        infer_theta_a0_from_theta_c,
        infer_theta_c_from_fullcell_voltage,
        setParams,
        uocp_a_fun,
        uocp_c_fun,
        i0_a_degradation_param_fun,
        i0_c_fun,
        ds_a_fun,
        ds_c_degradation_param_fun,
    )

print("INFO: USING ASSB voltage-anchored, gradient-preserving SPM MODEL")

EFFECTIVE_DISC_AREA_M2 = np.float64(7.853981633974483e-05)


def makeParams(initial_voltage_V=None, cs_c0_override=None, cs_a0_override=None, cycle_id=None):
    class Degradation:
        def __init__(self):
            self.n_params = 2
            self.ind_i0_a = 0
            self.ind_ds_c = 1
            self.bounds = [[] for _ in range(self.n_params)]
            self.ref_vals = [0 for _ in range(self.n_params)]
            self.eff = np.float64(0.0)
            self.bounds[self.ind_i0_a] = [np.float64(0.5), np.float64(4.0)]
            self.bounds[self.ind_ds_c] = [np.float64(0.5), np.float64(40.0)]
            self.ref_vals[self.ind_i0_a] = np.float64(1.0)
            self.ref_vals[self.ind_ds_c] = np.float64(1.0)

    class Macroscopic:
        def __init__(self):
            self.F = np.float64(96485.3321e3)
            self.R = np.float64(8.3145e3)
            self.T = np.float64(298.15)
            self.T_const = np.float64(298.15)
            self.T_ref = np.float64(298.15)
            self.C = np.float64(0.0)
            self.tmin = np.float64(0.0)
            self.tmax = np.float64(1000.0)
            self.rmin = np.float64(0.0)
            self.I = np.float64(0.0)

    class Anode:
        def __init__(self):
            self.thickness = np.float64(100e-6)
            self.solids = self.Anode_solids()
            self.A = EFFECTIVE_DISC_AREA_M2
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(8e-6)
            self.csmax = np.float64(30.53)
            self.uocp = uocp_a_fun
            self.i0 = i0_a_degradation_param_fun
            self.ds = ds_a_fun

        class Anode_solids:
            def __init__(self):
                self.eps = np.float64(1.0)

    class Cathode:
        def __init__(self):
            self.thickness = np.float64(16e-6)
            self.A = EFFECTIVE_DISC_AREA_M2
            self.solids = self.Cathode_solids()
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(3.6e-6)
            self.csmax = np.float64(51.554)
            self.uocp = uocp_c_fun
            self.i0 = i0_c_fun
            self.ds = ds_c_degradation_param_fun

        class Cathode_solids:
            def __init__(self):
                self.eps = np.float64(0.55)

    deg = Degradation()
    bat = Macroscopic()
    an = Anode()
    ca = Cathode()

    if cs_c0_override is not None:
        cs_c0 = np.float64(cs_c0_override)
        theta_c0 = np.float64(np.clip(cs_c0 / ca.csmax, 0.0, 1.0))
    elif initial_voltage_V is not None:
        theta_c0 = infer_theta_c_from_fullcell_voltage(initial_voltage_V)
        cs_c0 = np.float64(infer_cs_c0_from_fullcell_voltage(initial_voltage_V, ca.csmax))
    else:
        theta_c0 = np.float64(0.39)
        cs_c0 = np.float64(theta_c0 * ca.csmax)

    if cs_a0_override is not None:
        cs_a0 = np.float64(cs_a0_override)
        theta_a0 = np.float64(np.clip(cs_a0 / an.csmax, 0.0, 1.0))
    else:
        theta_a0 = infer_theta_a0_from_theta_c(theta_c0)
        cs_a0 = np.float64(theta_a0 * an.csmax)

    class IC:
        def __init__(self):
            self.an = self.Anode_IC()
            self.ca = self.Cathode_IC(self.an.cs)
            self.ce = np.float64(1.2)
            self.phie = -an.uocp(self.an.cs, an.csmax)
            self.phis_c = ca.uocp(self.ca.cs, ca.csmax) - an.uocp(self.an.cs, an.csmax)

        class Anode_IC:
            def __init__(self):
                self.ce = np.float64(1.2)
                self.cs = np.float64(cs_a0)
                self.phis = np.float64(0.0)

        class Cathode_IC:
            def __init__(self, cs_a0_for_phi):
                self.ce = np.float64(1.2)
                self.cs = np.float64(cs_c0)
                self.phis = ca.uocp(self.cs, ca.csmax) - an.uocp(cs_a0_for_phi, an.csmax)

    ic = IC()
    params = {}
    params = setParams(params, deg, bat, an, ca, ic)
    params["cycle_id"] = None if cycle_id is None else int(cycle_id)
    params["theta_c0"] = np.float64(theta_c0)
    params["theta_a0"] = np.float64(theta_a0)
    params["initial_voltage_anchor_V"] = None if initial_voltage_V is None else float(initial_voltage_V)
    return params
