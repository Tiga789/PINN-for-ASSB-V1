import numpy as np
import tensorflow as tf
from keras.backend import set_floatx
from uocp_cs import uocp_c_fun_x

set_floatx("float64")

# -----------------------------------------------------------------------------
# ASSB gradient-preserving v1-in-place revision.
#
# This file keeps the same filename so it can overwrite the previous v0 script,
# but the logic is updated to address three issues discussed in the chat:
# 1) avoid using an overly large constant D_s,c that washes out radial gradients;
# 2) expose robust NumPy helpers so the batch script can infer cs_c0 from the
#    cycle starting / pre-cycle-rest voltage instead of fixing cs_c0=0.39*cmax;
# 3) keep the Li-In negative electrode as an effective flat-potential boundary,
#    but preserve compatibility with the existing project API.
# -----------------------------------------------------------------------------

UOCP_A_LIIN_V = np.float64(0.62)
I0_A_REF = np.float64(0.1)  # A / m^2

# Lower than the old constant 1e-14 m^2/s to reduce radial over-smoothing.
DS_C_REF = np.float64(5.0e-15)
DS_C_MIN_FACTOR = np.float64(0.35)
DS_C_PEAK_CENTER = np.float64(0.45)
DS_C_PEAK_WIDTH = np.float64(0.22)

# Legacy cathode OCP remapped to an ASSB-compatible full-cell window.
UOCP_C_SCALE = np.float64(0.7624151100416336)
UOCP_C_SHIFT = np.float64(0.44293391056209597)

# Pure NumPy polynomial coefficients copied from uocp_cs.py so that the batch
# generator can invert the cathode OCP without relying on TensorFlow ops.
UOCP_C_COEFFS_NUMPY = np.array(
    [
        -43309.69063512314,
        122888.63938515769,
        -69735.99554716503,
        -59749.183217994185,
        25744.002733171154,
        15730.398058573825,
        54021.915506318735,
        -44566.03206954511,
        64.32177924593454,
        -7780.173422833786,
        1117.4042221859695,
        7387.492376558274,
        -7237.289515884936,
        -705.4465901574707,
        17170.20236584321,
        -42.60228181558803,
        -23266.56994359366,
        10810.92851132453,
        2545.4065429021307,
        1.6554268823619098,
        751.3515882152476,
        -4447.12851190078,
        3727.268889820381,
        -1331.1791971457515,
        227.4712483170547,
        -17.646894926746256,
        0.8568207255402533,
        -2.34505930698951,
        5.059010555584711,
    ],
    dtype=np.float64,
)


def cathode_ocp_theta_numpy(theta):
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    u_old = np.polyval(UOCP_C_COEFFS_NUMPY, theta)
    return UOCP_C_SCALE * u_old + UOCP_C_SHIFT


def cathode_ocp_cs_numpy(cs_c, cscamax):
    theta = np.asarray(cs_c, dtype=np.float64) / np.float64(cscamax)
    return cathode_ocp_theta_numpy(theta)


def infer_theta_c_from_fullcell_voltage(fullcell_voltage_V, uocp_a=UOCP_A_LIIN_V):
    """
    Infer the cathode stoichiometric fraction from a rested / quasi-rested
    full-cell voltage using the flat Li-In negative electrode approximation.

    Because the remapped cathode OCP is strictly monotonic decreasing over
    theta in [0, 1], inversion is done by interpolation on a dense lookup grid.
    """
    theta_grid = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    u_grid = cathode_ocp_theta_numpy(theta_grid)
    target_uc = np.float64(fullcell_voltage_V) + np.float64(uocp_a)
    target_uc = np.clip(target_uc, np.min(u_grid), np.max(u_grid))
    # u_grid is decreasing, so reverse both sides for np.interp
    theta = np.interp(target_uc, u_grid[::-1], theta_grid[::-1])
    return np.float64(theta)


def infer_cs_c0_from_fullcell_voltage(fullcell_voltage_V, cscamax, uocp_a=UOCP_A_LIIN_V):
    theta_c0 = infer_theta_c_from_fullcell_voltage(fullcell_voltage_V, uocp_a=uocp_a)
    return np.float64(theta_c0 * np.float64(cscamax))


# Legacy reference pair was roughly theta_c0=0.39 and theta_a0=0.91, so their
# sum is ~1.30. Reusing that affine complement keeps the new initial state close
# to the old project inventory while allowing voltage-anchored cycle-specific
# cathode initialization.
def infer_theta_a0_from_theta_c(theta_c):
    theta_c = np.float64(theta_c)
    theta_a = np.float64(1.30) - theta_c
    return np.float64(np.clip(theta_a, 0.75, 0.99))


def uocp_a_simp(cs_a, csanmax):
    x = cs_a / csanmax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return np.float64(0.2) - np.float64(0.2) * x


# Li-In negative electrode approximated as a flat plateau.
def uocp_a_fun(cs_a, csanmax):
    x = cs_a / csanmax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return UOCP_A_LIIN_V + np.float64(0.0) * x


# Keep a copy of the original smooth cathode shape.
def uocp_c_fun_legacy(cs_c, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return uocp_c_fun_x(x)


# Remapped cathode OCP.
def uocp_c_fun(cs_c, cscamax):
    u_old = uocp_c_fun_legacy(cs_c, cscamax)
    return UOCP_C_SCALE * u_old + UOCP_C_SHIFT


def uocp_c_simp(cs_c, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return np.float64(4.30) - np.float64(1.50) * x


def i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R):
    return (
        np.float64(2.5)
        * np.float64(0.27)
        * tf.exp(
            np.float64(
                (-30.0e6 / R)
                * (np.float64(1.0) / T - np.float64(1.0) / np.float64(303.15))
            )
        )
        * tf.math.maximum(ce, np.float64(0.0)) ** alpha
        * tf.math.maximum(csanmax - cs_a_max, np.float64(0.0)) ** alpha
        * tf.math.maximum(cs_a_max, np.float64(0.0))
        ** (np.float64(1.0) - alpha)
    )


# Effective Li-In kinetic baseline times degradation factor.
def i0_a_degradation_param_fun(
    cs_a_max, ce, T, alpha, csanmax, R, degradation_param
):
    deg = tf.maximum(tf.cast(degradation_param, tf.float64), np.float64(1e-12))
    ce_like = tf.ones_like(tf.cast(ce, tf.float64))
    return I0_A_REF * deg * ce_like


def i0_a_simp(cs_a_max, ce, T, alpha, csanmax, R):
    return np.float64(2.0) * np.ones(np.shape(ce), dtype="float64")


def i0_a_simp_degradation_param(
    cs_a_max, ce, T, alpha, csanmax, R, degradation_param
):
    ce_like = tf.ones_like(tf.cast(ce, tf.float64))
    deg = tf.maximum(tf.cast(degradation_param, tf.float64), np.float64(1e-12))
    return np.float64(2.0) * deg * ce_like


def i0_c_fun(cs_c_max, ce, T, alpha, cscamax, R):
    x = cs_c_max / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return (
        np.float64(9.0)
        * (
            np.float64(1.650452829641290e01) * x**5
            - np.float64(7.523567141488800e01) * x**4
            + np.float64(1.240524690073040e02) * x**3
            - np.float64(9.416571081287610e01) * x**2
            + np.float64(3.249768821737960e01) * x
            - np.float64(3.585290065824760e00)
        )
        * tf.math.maximum(ce / np.float64(1.2), np.float64(0.0)) ** alpha
        * tf.exp(
            (np.float64(-30.0e6) / R)
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )


def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    return np.float64(3.0) * np.ones(np.shape(ce), dtype="float64")


def ds_a_fun(T, R):
    return np.float64(3.0e-14) * tf.exp(
        (np.float64(-30.0e6) / R)
        * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
    )


def grad_ds_a_cs_a(T, R):
    return np.float64(0.0)


def ds_a_fun_simp(T, R):
    return np.float64(3.0e-14)


# NMC811 effective diffusivity with SOC dependence.
def ds_c_fun(cs_c, T, R, cscamax):
    theta = tf.clip_by_value(cs_c / cscamax, np.float64(0.0), np.float64(1.0))
    envelope = DS_C_MIN_FACTOR + (np.float64(1.0) - DS_C_MIN_FACTOR) * tf.exp(
        -((theta - DS_C_PEAK_CENTER) / DS_C_PEAK_WIDTH) ** np.float64(2.0)
    )
    return DS_C_REF * envelope


# Analytical derivative of the Gaussian-envelope diffusivity.
def grad_ds_c_cs_c(cs_c, T, R, cscamax):
    theta = tf.clip_by_value(cs_c / cscamax, np.float64(0.0), np.float64(1.0))
    exp_term = tf.exp(-((theta - DS_C_PEAK_CENTER) / DS_C_PEAK_WIDTH) ** np.float64(2.0))
    d_env_d_theta = (
        (np.float64(1.0) - DS_C_MIN_FACTOR)
        * exp_term
        * (
            -np.float64(2.0)
            * (theta - DS_C_PEAK_CENTER)
            / (DS_C_PEAK_WIDTH**np.float64(2.0))
        )
    )
    return DS_C_REF * d_env_d_theta / cscamax


# degradation factor > 1 means slower effective diffusion.
def ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param):
    deg = tf.maximum(tf.cast(degradation_param, tf.float64), np.float64(1e-12))
    return ds_c_fun(cs_c, T, R, cscamax) / deg


def ds_c_fun_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_plot(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_plot_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_simp_degradation_param(cs_c, T, R, cscamax, degradation_param):
    return ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param)


def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    return -j_a * (F / i0_a) * (R * T / F) - Uocp_a0


def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return j_c * (F / i0_c) * (R * T / F) + Uocp_c0 + phie0


def setParams(params, deg, bat, an, ca, ic):
    # Parametric domain
    params["deg_i0_a_min"] = deg.bounds[deg.ind_i0_a][0]
    params["deg_i0_a_max"] = deg.bounds[deg.ind_i0_a][1]
    params["deg_ds_c_min"] = deg.bounds[deg.ind_ds_c][0]
    params["deg_ds_c_max"] = deg.bounds[deg.ind_ds_c][1]

    params["param_eff"] = deg.eff
    params["deg_i0_a_ref"] = deg.ref_vals[deg.ind_i0_a]
    params["deg_ds_c_ref"] = deg.ref_vals[deg.ind_ds_c]
    params["deg_i0_a_min_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_min"] - params["deg_i0_a_ref"])
        * params["param_eff"]
    )
    params["deg_i0_a_max_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_max"] - params["deg_i0_a_ref"])
        * params["param_eff"]
    )
    params["deg_ds_c_min_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_min"] - params["deg_ds_c_ref"])
        * params["param_eff"]
    )
    params["deg_ds_c_max_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_max"] - params["deg_ds_c_ref"])
        * params["param_eff"]
    )
    # Domain
    params["tmin"] = bat.tmin
    params["tmax"] = bat.tmax
    params["rmin"] = bat.rmin

    # Params fixed
    params["A_a"] = an.A
    params["A_c"] = ca.A
    params["F"] = bat.F
    params["R"] = bat.R
    params["T"] = bat.T
    params["C"] = bat.C
    params["I_discharge"] = bat.I

    params["alpha_a"] = an.alpha
    params["alpha_c"] = ca.alpha

    # Params to fit
    params["Rs_a"] = an.D50 / np.float64(2.0)
    params["Rs_c"] = ca.D50 / np.float64(2.0)
    params["rescale_R"] = np.float64(max(params["Rs_a"], params["Rs_c"]))
    params["csanmax"] = an.csmax
    params["cscamax"] = ca.csmax
    params["rescale_T"] = np.float64(max(bat.tmax, 1e-16))

    # Typical variables magnitudes
    params["mag_cs_a"] = np.float64(25)
    params["mag_cs_c"] = np.float64(32.5)
    params["mag_phis_c"] = np.float64(4.25)
    params["mag_phie"] = np.float64(0.15)
    params["mag_ce"] = np.float64(1.2)

    # FUNCTIONS
    params["Uocp_a"] = an.uocp
    params["Uocp_c"] = ca.uocp
    params["i0_a"] = an.i0
    params["i0_c"] = ca.i0
    params["D_s_a"] = an.ds
    params["D_s_c"] = ca.ds

    # INIT
    params["ce0"] = ic.ce
    params["ce_a0"] = ic.ce
    params["ce_c0"] = ic.ce
    params["cs_a0"] = ic.an.cs
    params["cs_c0"] = ic.ca.cs
    params["eps_s_a"] = an.solids.eps
    params["eps_s_c"] = ca.solids.eps
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness
    j_a = (
        -(params["I_discharge"] / params["A_a"])
        * params["Rs_a"]
        / (np.float64(3.0) * params["eps_s_a"] * params["F"] * params["L_a"])
    )
    j_c = (
        (params["I_discharge"] / params["A_c"])
        * params["Rs_c"]
        / (np.float64(3.0) * params["eps_s_c"] * params["F"] * params["L_c"])
    )
    params["j_a"] = j_a
    params["j_c"] = j_c

    cse_a = ic.an.cs
    i0_a = params["i0_a"](
        cse_a,
        params["ce0"],
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        np.float64(1.0),
    )
    Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
    params["Uocp_a0"] = Uocp_a

    params["phie0"] = phie0_fun

    cse_c = ic.ca.cs
    i0_c = params["i0_c"](
        cse_c,
        params["ce0"],
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    params["i0_c0"] = i0_c
    Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
    params["Uocp_c0"] = Uocp_c

    params["phis_c0"] = phis_c0_fun

    params["rescale_cs_a"] = -ic.an.cs
    params["rescale_cs_c"] = params["cscamax"] - ic.ca.cs
    params["rescale_phis_c"] = abs(np.float64(3.8) - np.float64(4.110916387038547))
    params["rescale_phie"] = abs(np.float64(-0.15) - np.float64(-0.07645356566609385))

    return params
