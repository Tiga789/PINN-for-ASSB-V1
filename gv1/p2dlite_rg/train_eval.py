from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .data import build_features, build_targets, load_profile_arrays
from .metrics import compute_rg_metrics


def standardize_x(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean[None, :]) / std[None, :]).astype(np.float32)


def standardize_y(Y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((Y - mean[None, :]) / std[None, :]).astype(np.float32)


def unstandardize_y(Yn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Yn * std[None, :] + mean[None, :]).astype(np.float32)


def predict_numpy(model, X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device, batch_size: int = 65536) -> np.ndarray:
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = standardize_x(X[i:i + batch_size], x_mean, x_std)
            xt = torch.from_numpy(xb).to(device)
            yp = model(xt).detach().cpu().numpy()
            outs.append(unstandardize_y(yp, y_mean, y_std))
    return np.concatenate(outs, axis=0)


def evaluate_one_profile(npz_path: Path, root: Path, profile_index: int, profile_count: int, model, state: Dict[str, Any], device, batch_size: int = 65536, eval_stride: int = 1, save_prediction_path: Path | None = None) -> Dict[str, Any]:
    prof = load_profile_arrays(npz_path, root)
    X, feature_names = build_features(prof, profile_index, profile_count, include_profile_onehot=bool(state.get('include_profile_onehot', True)))
    Y, target_names, slices = build_targets(prof)
    stride = max(1, int(eval_stride))
    if stride > 1:
        X_eval = X[::stride]
        Y_eval = Y[::stride]
        t_eval = prof['t'][::stride]
    else:
        X_eval = X
        Y_eval = Y
        t_eval = prof['t']
    Y_pred = predict_numpy(
        model,
        X_eval,
        np.asarray(state['x_mean'], dtype=np.float32),
        np.asarray(state['x_std'], dtype=np.float32),
        np.asarray(state['y_mean'], dtype=np.float32),
        np.asarray(state['y_std'], dtype=np.float32),
        device,
        batch_size=batch_size,
    )
    metrics = compute_rg_metrics(Y_eval, Y_pred, state['target_slices'])
    metrics.update({
        'profile_id': prof['profile_id'],
        'npz_path': str(npz_path),
        'n_eval': int(Y_eval.shape[0]),
        'eval_stride': int(stride),
    })
    if save_prediction_path is not None:
        save_prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_prediction_path,
            t_global_s=t_eval.astype(np.float32),
            y_true=Y_eval.astype(np.float32),
            y_pred=Y_pred.astype(np.float32),
            target_names=np.array(state['target_names']),
            feature_names=np.array(state['feature_names']),
            profile_id=np.array(str(prof['profile_id'])),
        )
    return metrics
