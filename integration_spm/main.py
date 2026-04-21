from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Keep original path semantics.
_THIS_DIR = Path(__file__).resolve().parent
_UTIL_DIR = _THIS_DIR / 'util'
if str(_UTIL_DIR) not in sys.path:
    sys.path.append(str(_UTIL_DIR))

import argument
from init_pinn import initialize_nn, initialize_params


def _gpu_guard_and_warmup() -> None:
    print(f"[Torch] version = {torch.__version__}")
    print(f"[Torch] cuda available = {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA GPU is not available in the current PyTorch runtime. "
            "This training run is configured to require GPU."
        )
    n_gpu = torch.cuda.device_count()
    print(f"[Torch] visible GPU count = {n_gpu}")
    for i in range(n_gpu):
        print(f"[Torch] GPU[{i}] = {torch.cuda.get_device_name(i)}")
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda:0")
    x = torch.randn((2048, 2048), device=dev, dtype=torch.float64)
    y = torch.randn((2048, 2048), device=dev, dtype=torch.float64)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = x @ y
    torch.cuda.synchronize()
    print(f"[Torch] CUDA warm-up matmul OK, elapsed = {time.time() - t0:.3f}s")
    print("[Torch] Training will now start on GPU.")


def do_training_only(input_params, nn):
    LEARNING_RATE_LBFGS = input_params["LEARNING_RATE_LBFGS"]
    LEARNING_RATE_MODEL = input_params["LEARNING_RATE_MODEL"]
    LEARNING_RATE_MODEL_FINAL = input_params["LEARNING_RATE_MODEL_FINAL"]
    LEARNING_RATE_WEIGHTS = input_params["LEARNING_RATE_WEIGHTS"]
    LEARNING_RATE_WEIGHTS_FINAL = input_params["LEARNING_RATE_WEIGHTS_FINAL"]
    GRADIENT_THRESHOLD = input_params["GRADIENT_THRESHOLD"]
    INNER_EPOCHS = input_params["INNER_EPOCHS"]
    EPOCHS = input_params["EPOCHS"]
    START_WEIGHT_TRAINING_EPOCH = input_params["START_WEIGHT_TRAINING_EPOCH"]

    factorSchedulerModel = np.log(LEARNING_RATE_MODEL_FINAL / (LEARNING_RATE_MODEL + 1e-16)) / ((EPOCHS + 1e-16) / 2)

    def schedulerModel(epoch, lr):
        if epoch < EPOCHS // 2:
            return lr
        return max(lr * np.exp(factorSchedulerModel), LEARNING_RATE_MODEL_FINAL)

    factorSchedulerWeights = np.log(LEARNING_RATE_WEIGHTS_FINAL / (LEARNING_RATE_WEIGHTS + 1e-16)) / ((EPOCHS + 1e-16) / 2)

    def schedulerWeights(epoch, lr):
        if epoch < EPOCHS // 2:
            return lr
        return max(lr * np.exp(factorSchedulerWeights), LEARNING_RATE_WEIGHTS_FINAL)

    time_start = time.time()
    unweighted_loss = nn.train(
        learningRateModel=LEARNING_RATE_MODEL,
        learningRateModelFinal=LEARNING_RATE_MODEL_FINAL,
        lrSchedulerModel=schedulerModel,
        learningRateWeights=LEARNING_RATE_WEIGHTS,
        learningRateWeightsFinal=LEARNING_RATE_WEIGHTS_FINAL,
        lrSchedulerWeights=schedulerWeights,
        learningRateLBFGS=LEARNING_RATE_LBFGS,
        inner_epochs=INNER_EPOCHS,
        start_weight_training_epoch=START_WEIGHT_TRAINING_EPOCH,
        gradient_threshold=GRADIENT_THRESHOLD,
    )
    time_end = time.time()
    return time_end - time_start, unweighted_loss


def _copytree_replace(src: str, dst: str):
    src_p = Path(src)
    dst_p = Path(dst)
    if dst_p.exists():
        shutil.rmtree(dst_p)
    shutil.copytree(src_p, dst_p)


def do_training(input_params, nn):
    ID = input_params["ID"]
    elapsedTime, unweighted_loss = do_training_only(input_params, nn)
    _copytree_replace(nn.modelFolder, f"ModelFin_{ID}")
    _copytree_replace(nn.logLossFolder, f"LogFin_{ID}")
    return elapsedTime, unweighted_loss


def main():
    _gpu_guard_and_warmup()
    args = argument.initArg()
    input_params = initialize_params(args)
    nn = initialize_nn(args=args, input_params=input_params)
    print(f"[Torch] model device = {nn.device}")
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    print(f"Total time {elapsed:.2f}s")
    print(f"Unweighted loss {unweighted_loss}")


if __name__ == "__main__":
    main()
