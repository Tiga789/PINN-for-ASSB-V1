import json
from pathlib import Path

import numpy as np
from keras.backend import set_floatx
from thermo import *

set_floatx("float64")

print("INFO: USING ASSB DISCHARGE TRAINING SPM PRIOR")

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_JSON = ROOT / "Data" / "assb_csv_3targets_spm_v0_discharge_train70_val30" / "meta" / "train_pack_summary.json"

DEFAULT_AREA_M2 = np.float64(7.853981633974483e-05)
DEFAULT_TMAX_S = np.float64(3600.0)
DEFAULT_I_A = np.float64(-3.3e-4)


def _load_summary():
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def makeParams():
    summary = _load_summary()
    tmax_s = np.float64(summary.get("tmax_train_s", DEFAULT_TMAX_S))
    current_ref_A = np.float64(summary.get("current_ref_A", DEFAULT_I_A))

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
            self.tmax = tmax_s
            self.rmin = np.float64(0.0)
            self.I = current_ref_A

    class Anode:
        def __init__(self):
            self.thickness = np.float64(100e-6)
            self.solids = self.Anode_solids()
            self.A = DEFAULT_AREA_M2
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
            self.A = DEFAULT_AREA_M2
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

    class IC:
        def __init__(self):
            self.an = self.Anode_IC()
            self.ca = self.Cathode_IC(self.an.cs)
            self.ce = np.float64(1.2)
            self.phie = -an.uocp(self.an.cs, an.csmax)
            self.phis_c = ca.uocp(self.ca.cs, ca.csmax) - an.uocp(
                self.an.cs, an.csmax
            )

        class Anode_IC:
            def __init__(self):
                self.ce = np.float64(1.2)
                self.cs = np.float64(0.91 * an.csmax)
                self.phis = np.float64(0.0)

        class Cathode_IC:
            def __init__(self, cs_a0):
                self.ce = np.float64(1.2)
                self.cs = np.float64(0.39 * ca.csmax)
                self.phis = ca.uocp(self.cs, ca.csmax) - an.uocp(
                    cs_a0, an.csmax
                )

    ic = IC()
    params = {}
    params = setParams(params, deg, bat, an, ca, ic)
    params["train_summary_json"] = str(SUMMARY_JSON)
    return params
