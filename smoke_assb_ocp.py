import torch
from util.spm_assb_train_discharge import makeParams

p = makeParams()

cs_a0 = torch.tensor(p["cs_a0"], dtype=torch.float64)
cs_c0 = torch.tensor(p["cs_c0"], dtype=torch.float64)

Ua = p["Uocp_a"](cs_a0, p["csanmax"])
Uc = p["Uocp_c"](cs_c0, p["cscamax"])

print("U_a0 =", float(Ua))
print("U_c0 =", float(Uc))
print("U_full_ocp =", float(Uc - Ua))
print("Uocp_c0 stored =", p.get("Uocp_c0", None))
print("Uocp_a0 stored =", p.get("Uocp_a0", None))