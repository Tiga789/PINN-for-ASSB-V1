"""D12-S1G high-local train-inside P2D-like correction trainer.

This trainer subclasses the existing GV1 D9.5.1 trainer where possible, but
instantiates D12-S1G model/transform/loss classes.  It is an additive module;
D9.6/D9.5.1 files remain untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from .d12_s1_p2d_losses import D12S1LossComputer, D12S1LossWeights, make_optimizer
from .d12_s1_p2d_model import D12S1ModelConfig, D12S1P2DLocalPINN, count_trainable_parameters
from .d12_s1_p2d_transform import D12S1OutputTransformConfig, D12S1P2DOutputTransform
from .trainer import GV1ReplayDataset, GV1Trainer, TrainerConfig, _load_npz, resolve_device, set_seed


@dataclass
class D12S1TrainerConfig(TrainerConfig):
    """TrainerConfig with D12-S1G identity marker.

    It inherits all D9.5.1 fields, so existing command-generation logic can be
    reused.  The custom classes are selected by :class:`D12S1Trainer`.
    """

    experiment_tag: str = "D12-S1G_s1e_soft_highlocal_limiter"
    model: dict[str, Any] = field(default_factory=dict)
    transform: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class D12S1Trainer(GV1Trainer):
    """GV1 trainer variant using the D12-S1G P2D-local classes."""

    def __init__(self, config: D12S1TrainerConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = resolve_device(config.device)
        self.output_dir = __import__("pathlib").Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        arrays = _load_npz(config.solution_npz)
        self.dataset = GV1ReplayDataset(arrays, config)

        model_cfg = D12S1ModelConfig.from_mapping(config.model)
        if model_cfg.condition_dim != len(self.dataset.condition):
            model_cfg.condition_dim = int(len(self.dataset.condition))
        self.model = D12S1P2DLocalPINN(model_cfg).to(self.device)

        tcfg_data = dict(self.dataset.transform_config.to_dict())
        tcfg_data.update(dict(config.transform))
        self.transform = D12S1P2DOutputTransform(D12S1OutputTransformConfig.from_mapping(tcfg_data))
        self.loss_computer = D12S1LossComputer(D12S1LossWeights.from_mapping(config.losses))
        self.optimizer = make_optimizer(self.model, lr=config.lr, weight_decay=config.weight_decay)
        self.history: list[dict[str, Any]] = []

    def _checkpoint_dict(self, epoch: int, loss: float) -> dict[str, Any]:
        data = super()._checkpoint_dict(epoch, loss)
        data["d12_s1"] = {
            "experiment_tag": self.config.experiment_tag,
            "train_inside_p2d_localized_correction": True,
            "mainline_overwritten": False,
            "notes": (
                "D12-S1G uses S1E low anchor plus local high-only regret/overshoot preservation; "
                "D9.6/D9.5.1 mainline files are not modified."
            ),
        }
        return data

    @torch.no_grad()
    def save_prediction_npz(self):  # type: ignore[override]
        """Save prediction.npz and include D12-S1G P2D diagnostic arrays."""
        # Reuse the robust parent implementation for all common arrays.
        pred_path = super().save_prediction_npz()

        # Append D12-S1G correction arrays.  This is intentionally a second pass
        # to keep the parent implementation stable and avoid copying its long
        # grid-inference routine.
        self.model.eval()
        idx = self.dataset.prediction_time_indices(self.config.prediction_time_points)
        t_norm = torch.as_tensor(self.dataset.t_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        current_A = torch.as_tensor(self.dataset.current_A[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        current_norm = torch.as_tensor(self.dataset.current_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        temp_norm = torch.as_tensor(self.dataset.temperature_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        cbar_a = torch.as_tensor(self.dataset.cbar_a[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        cbar_c = torch.as_tensor(self.dataset.cbar_c[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        condition = torch.as_tensor(
            np.repeat(self.dataset.condition.reshape(1, -1), len(idx), axis=0),
            device=self.device,
            dtype=torch.float32,
        )
        r_surface = torch.ones_like(t_norm)
        raw = self.model(t_norm, r_surface, current_norm, temp_norm, condition)
        out = self.transform(
            raw,
            r_norm=r_surface,
            current_A=current_A,
            current_norm=current_norm,
            cbar_a=cbar_a,
            cbar_c=cbar_c,
            temperature_norm=temp_norm,
            condition=condition,
        )
        base = self.transform.without_p2d(
            raw,
            r_norm=r_surface,
            current_A=current_A,
            current_norm=current_norm,
            cbar_a=cbar_a,
            cbar_c=cbar_c,
            temperature_norm=temp_norm,
            condition=condition,
        )
        with np.load(pred_path, allow_pickle=True) as old:
            arrays = {k: old[k] for k in old.files}
        arrays.update(
            {
                "voltage_p2d_transport_deficit": out["voltage_p2d_transport_deficit"].detach().cpu().numpy().reshape(-1).astype(np.float32),
                "voltage_p2d_deficit_raw": out["voltage_p2d_deficit_raw"].detach().cpu().numpy().reshape(-1).astype(np.float32),
                "voltage_without_p2d": base["phis_c"].detach().cpu().numpy().reshape(-1).astype(np.float32),
                "voltage_with_p2d": out["phis_c"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            }
        )
        np.savez_compressed(pred_path, **arrays)
        return pred_path


def run_d12_s1_training(config: D12S1TrainerConfig | Mapping[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, D12S1TrainerConfig) else D12S1TrainerConfig(**dict(config))
    trainer = D12S1Trainer(cfg)
    summary = trainer.train()
    summary["stage"] = "D12-S1G high-local train-inside P2D-like correction"
    summary["d12_s1_mainline_overwritten"] = False
    return summary
