from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

print("=== Runtime backend check ===")

try:
    import torch
    print(f"[Torch] version = {torch.__version__}")
    print(f"[Torch] cuda build = {torch.version.cuda}")
    print(f"[Torch] cuda available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Torch] device count = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[Torch] GPU[{i}] = {torch.cuda.get_device_name(i)}")
        dev = torch.device("cuda:0")
        x = torch.randn((1024, 1024), device=dev, dtype=torch.float64)
        y = torch.randn((1024, 1024), device=dev, dtype=torch.float64)
        torch.cuda.synchronize()
        z = x @ y
        torch.cuda.synchronize()
        print(f"[Torch] matmul OK on {z.device}")
except Exception as exc:
    print(f"[Torch] ERROR: {exc}")

try:
    import tensorflow as tf
    print(f"[TensorFlow] version = {tf.__version__}")
    print(f"[TensorFlow] GPUs = {tf.config.list_physical_devices('GPU')}")
except Exception as exc:
    print(f"[TensorFlow] INFO/ERROR: {exc}")
