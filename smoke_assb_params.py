import numpy as np
from util.spm_assb_train_discharge import makeParams

p = makeParams()

print("T =", p["T"])
print("csanmax =", p["csanmax"])
print("cscamax =", p["cscamax"])
print("theta_a0 =", p["cs_a0"] / p["csanmax"])
print("theta_c0 =", p["cs_c0"] / p["cscamax"])
print("ce0 =", p["ce0"])
print("R_ohm_eff =", p.get("R_ohm_eff", None))
print("Ds_a =", p["D_s_a"](p["T"], p["R"]))

cp = p.get("current_profile", None)
print("has current_profile =", cp is not None)

if cp is not None:
    t, I = cp
    t = np.asarray(t)
    I = np.asarray(I)
    print("profile points =", len(t))
    print("t range =", t[0], "to", t[-1])
    print("I min/max =", I.min(), I.max())
    print("I first =", I[0])