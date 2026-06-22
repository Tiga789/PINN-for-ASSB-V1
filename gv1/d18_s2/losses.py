from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class S2LossConfig:
    w_cs_a: float = 1.0
    w_cs_c: float = 1.0
    w_theta_a: float = 0.25
    w_theta_c: float = 0.25
    w_phie: float = 1.5
    w_phis_c: float = 1.0
    w_cbar: float = 0.5
    w_inventory_residual: float = 0.02
    w_inventory_cycle_smooth: float = 0.02
    w_boundary_jump_match: float = 0.10
    w_gauge_smooth: float = 0.01
    w_cbar_clip: float = 2.0
    huber_beta: float = 0.5

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "S2LossConfig":
        if not value:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: value[k] for k in value if k in allowed})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scaled_huber(pred: torch.Tensor, true: torch.Tensor, scale: float, beta: float) -> torch.Tensor:
    s = max(1.0e-8, float(scale))
    return F.smooth_l1_loss(pred / s, true / s, beta=beta)


def _boundary_jump_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    cycle_index: torch.Tensor,
    selected_cycle_ids: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for b in range(pred.shape[0]):
        cids = selected_cycle_ids[b]
        for pos in range(cids.numel() - 1):
            if int(cids[pos + 1] - cids[pos]) != 1:
                continue
            left = torch.nonzero(cycle_index[b] == pos, as_tuple=False).reshape(-1)
            right = torch.nonzero(cycle_index[b] == pos + 1, as_tuple=False).reshape(-1)
            if left.numel() == 0 or right.numel() == 0:
                continue
            pred_jump = pred[b, right[0]] - pred[b, left[-1]]
            true_jump = true[b, right[0]] - true[b, left[-1]]
            losses.append(torch.mean(((pred_jump - true_jump) / max(scale, 1e-8)) ** 2))
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def compute_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    target_scales: Mapping[str, float],
    config: S2LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    for key in ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c"):
        components[key] = _scaled_huber(
            outputs[key], batch[f"{key}_true"], target_scales[key], config.huber_beta
        )
    pred_cbar = torch.cat([outputs["cbar_a"], outputs["cbar_c"]], dim=-1)
    cbar_scale = max(float(target_scales["cs_a"]), float(target_scales["cs_c"]))
    components["cbar"] = _scaled_huber(
        pred_cbar, batch["cbar_true_report_only"], cbar_scale, config.huber_beta
    )
    inventory = torch.cat([outputs["inventory_delta_a"], outputs["inventory_delta_c"]], dim=-1)
    components["inventory_residual"] = torch.mean((inventory / cbar_scale) ** 2)
    cycle_inventory = outputs["inventory_cycle_unit"]
    if cycle_inventory.shape[1] > 1:
        components["inventory_cycle_smooth"] = torch.mean(
            (cycle_inventory[:, 1:] - cycle_inventory[:, :-1]) ** 2
        )
    else:
        components["inventory_cycle_smooth"] = cycle_inventory.new_zeros(())
    components["gauge_smooth"] = torch.mean(
        (outputs["potential_gauge"][:, 1:] - outputs["potential_gauge"][:, :-1]) ** 2
    ) if outputs["potential_gauge"].shape[1] > 1 else outputs["potential_gauge"].new_zeros(())
    components["cbar_clip"] = torch.mean(outputs["cbar_clip_a"] ** 2 + outputs["cbar_clip_c"] ** 2) / (cbar_scale**2)

    jump_terms = []
    for key in ("cs_a", "cs_c", "phie", "phis_c"):
        jump_terms.append(
            _boundary_jump_loss(
                outputs[key], batch[f"{key}_true"], batch["cycle_index"],
                batch["selected_cycle_ids"], float(target_scales[key])
            )
        )
    components["boundary_jump_match"] = torch.stack(jump_terms).mean()

    total = (
        config.w_cs_a * components["cs_a"]
        + config.w_cs_c * components["cs_c"]
        + config.w_theta_a * components["theta_a"]
        + config.w_theta_c * components["theta_c"]
        + config.w_phie * components["phie"]
        + config.w_phis_c * components["phis_c"]
        + config.w_cbar * components["cbar"]
        + config.w_inventory_residual * components["inventory_residual"]
        + config.w_inventory_cycle_smooth * components["inventory_cycle_smooth"]
        + config.w_boundary_jump_match * components["boundary_jump_match"]
        + config.w_gauge_smooth * components["gauge_smooth"]
        + config.w_cbar_clip * components["cbar_clip"]
    )
    components["total"] = total
    return total, components
