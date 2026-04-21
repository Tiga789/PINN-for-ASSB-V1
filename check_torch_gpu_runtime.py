from __future__ import annotations

import time
import torch

print(f"torch = {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.version.cuda = {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
print(f"visible GPU count = {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU[{i}] = {torch.cuda.get_device_name(i)}")

x = torch.randn((4096, 4096), device='cuda', dtype=torch.float64)
y = torch.randn((4096, 4096), device='cuda', dtype=torch.float64)
torch.cuda.synchronize()
t0 = time.time()
z = x @ y
torch.cuda.synchronize()
print(f"cuda matmul ok, elapsed = {time.time() - t0:.3f}s")
print(f"result mean = {float(z.mean().detach().cpu())}")
print("GPU runtime sanity check passed.")
