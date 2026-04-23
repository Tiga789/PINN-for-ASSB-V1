from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

print(f"torch = {torch.__version__}")
print(f"cuda_build = {torch.version.cuda}")
print(f"cuda_available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_count = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU[{i}] = {torch.cuda.get_device_name(i)}")
    dev = torch.device("cuda:0")
    x = torch.randn((1024, 1024), device=dev, dtype=torch.float64)
    y = torch.randn((1024, 1024), device=dev, dtype=torch.float64)
    torch.cuda.synchronize()
    z = x @ y
    torch.cuda.synchronize()
    print(f"matmul device = {z.device}")
