#!/usr/bin/env python
r"""Train the GV1 D9.5.1 trend-first warmup rare-regime conditioned PINN.

D9.5.1 repairs D9.5 by returning to a D9.3 trend-first objective and warming
up rare-regime terms.  Ordinary voltage/range/correlation losses are active from
epoch 1; low/high-tail, ultra-quantile, coverage and event losses are scheduled
in later.  Presets are selected from protocol/current/temperature only; they do
not inspect validation or test metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.profile_adaptive import resolve_profile_adaptive_preset  # noqa: E402
from gv1.trainer import TrainerConfig, run_training  # noqa: E402


def _resolve_solution_npz(solution_npz: str | None, prepared_dir: str | None) -> str:
    if solution_npz:
        p = Path(solution_npz)
    elif prepared_dir:
        p = Path(prepared_dir) / "solution_replay_profile.npz"
    else:
        raise ValueError("--solution_npz or --prepared_dir is required")
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def _bool_arg(x: str) -> bool:
    key = str(x).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {x!r}")


def _override_from_args(base: dict[str, Any], args: argparse.Namespace, keys: list[str]) -> dict[str, Any]:
    out = dict(base)
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    return out


def _pick(args: argparse.Namespace, name: str, preset: dict[str, Any], default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    if name in preset:
        return preset[name]
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="GV1 D9.5.1 trend-first warmup conditioned effective-SPM PINN smoke trainer.")
    ap.add_argument("--solution_npz", default=None, help="D8 solution_replay_profile.npz")
    ap.add_argument("--prepared_dir", default=None, help="Directory containing solution_replay_profile.npz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--profile_adaptive_mode", default="auto", choices=["auto", "trend_warmup", "warmup", "trend_tail_hybrid", "d951", "lowrate_trend_warmup", "lowrate_trend_tail", "event_highrate_trend_warmup", "event_highrate_trend_tail", "event_highrate", "d93", "d92", "smooth_2c", "event", "highrate", "lowrate"], help="D9.5.1 preset selector")

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument("--max_time_points", type=int, default=4096)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--start_time_s", type=float, default=None)
    ap.add_argument("--prediction_time_points", type=int, default=2048)
    ap.add_argument("--prediction_radial_points", type=int, default=64)
    ap.add_argument("--nominal_capacity_Ah", type=float, default=2.0)
    ap.add_argument("--current_scale_A", type=float, default=None)

    # Adaptive sampling options. Defaults are filled by gv1.profile_adaptive.
    ap.add_argument("--event_sampling_mix", type=float, default=None)
    ap.add_argument("--sample_weight_exponent", type=float, default=None)
    ap.add_argument("--low_voltage_threshold_V", type=float, default=None)
    ap.add_argument("--high_voltage_threshold_V", type=float, default=None)
    ap.add_argument("--low_voltage_quantile", type=float, default=None)
    ap.add_argument("--high_voltage_quantile", type=float, default=None)
    ap.add_argument("--high_current_quantile", type=float, default=None)
    ap.add_argument("--transition_current_delta_quantile", type=float, default=None)
    ap.add_argument("--temperature_extreme_quantile", type=float, default=None)

    # D9.5.1 warmup options. These normally come from the selected preset.
    ap.add_argument("--rare_loss_warmup_start_frac", type=float, default=None)
    ap.add_argument("--rare_loss_warmup_full_frac", type=float, default=None)
    ap.add_argument("--rare_loss_warmup_power", type=float, default=None)
    ap.add_argument("--rare_loss_start_scale", type=float, default=None)
    ap.add_argument("--rare_loss_final_scale", type=float, default=None)
    ap.add_argument("--rare_sample_start_scale", type=float, default=None)
    ap.add_argument("--rare_sample_final_scale", type=float, default=None)

    # Model options.
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--activation", default="tanh", choices=["tanh", "silu", "gelu", "relu"])
    ap.add_argument("--dropout", type=float, default=0.0)

    # Loss options. Defaults are filled by the selected D9.5.1 preset.
    ap.add_argument("--voltage_weight", type=float, default=None)
    ap.add_argument("--voltage_tail_weight", type=float, default=None)
    ap.add_argument("--voltage_bias_weight", type=float, default=None)
    ap.add_argument("--voltage_range_weight", type=float, default=None)
    ap.add_argument("--voltage_guardrail_weight", type=float, default=None)
    ap.add_argument("--voltage_quantile_weight", type=float, default=None)
    ap.add_argument("--voltage_asymmetry_weight", type=float, default=None)
    ap.add_argument("--voltage_event_weight", type=float, default=None)
    ap.add_argument("--voltage_corr_weight", type=float, default=None)
    ap.add_argument("--voltage_ultra_quantile_weight", type=float, default=None)
    ap.add_argument("--voltage_low_coverage_weight", type=float, default=None)
    ap.add_argument("--voltage_tail_balance_weight", type=float, default=None)
    ap.add_argument("--tail_fraction", type=float, default=None)
    ap.add_argument("--tail_weight_gain", type=float, default=None)
    ap.add_argument("--low_tail_extra_gain", type=float, default=None)
    ap.add_argument("--high_tail_extra_gain", type=float, default=None)
    ap.add_argument("--event_weight_gain", type=float, default=None)
    ap.add_argument("--huber_delta_V", type=float, default=None)
    ap.add_argument("--low_coverage_threshold_V", type=float, default=None)
    ap.add_argument("--high_coverage_threshold_V", type=float, default=None)
    ap.add_argument("--coverage_temperature_V", type=float, default=None)
    ap.add_argument("--center_weight", type=float, default=None)
    ap.add_argument("--surface_flux_weight", type=float, default=None)
    ap.add_argument("--radial_smooth_weight", type=float, default=None)
    ap.add_argument("--potential_smooth_weight", type=float, default=None)

    # Transform options. Defaults are filled by the selected D9.5.1 preset.
    ap.add_argument("--profile_event_gate", type=float, default=None)
    ap.add_argument("--profile_dynamic_event_gate", type=float, default=None)
    ap.add_argument("--radial_scale_a", type=float, default=None)
    ap.add_argument("--radial_scale_c", type=float, default=None)
    ap.add_argument("--resistance_ohm", type=float, default=None)
    ap.add_argument("--phis_c_head_mode", default=None, choices=["linear", "tanh", "softsign"])
    ap.add_argument("--phis_c_direct_scale", type=float, default=None)
    ap.add_argument("--phis_c_correction_scale_V", type=float, default=None)
    ap.add_argument("--low_voltage_gate_center_V", type=float, default=None)
    ap.add_argument("--low_voltage_gate_width_V", type=float, default=None)
    ap.add_argument("--phis_c_low_tail_scale_V", type=float, default=None)
    ap.add_argument("--phis_c_event_scale_V", type=float, default=None)
    ap.add_argument("--event_current_gain", type=float, default=None)
    ap.add_argument("--temperature_polarization_scale_V", type=float, default=None)
    ap.add_argument("--ocv_baseline_mix", type=float, default=None)
    ap.add_argument("--direct_voltage_mix", type=float, default=None)
    ap.add_argument("--ohmic_mix", type=float, default=None)
    ap.add_argument("--phie_current_scale_V", type=float, default=None)
    ap.add_argument("--phie_correction_scale_V", type=float, default=None)
    ap.add_argument("--surface_flux_gain", type=float, default=None)
    ap.add_argument("--voltage_range_strategy", default=None, choices=["profile_minmax", "percentile", "fixed"])
    ap.add_argument("--voltage_low_percentile", type=float, default=None)
    ap.add_argument("--voltage_high_percentile", type=float, default=None)
    ap.add_argument("--voltage_margin_V", type=float, default=None)
    ap.add_argument("--voltage_floor_V", type=float, default=None)
    ap.add_argument("--voltage_ceil_V", type=float, default=None)
    ap.add_argument("--voltage_guard_low_V", type=float, default=None)
    ap.add_argument("--voltage_guard_high_V", type=float, default=None)
    ap.add_argument("--enable_voltage_hard_clamp", type=_bool_arg, default=None)
    ap.add_argument("--voltage_min_V", type=float, default=None, help="Only used when --voltage_range_strategy fixed")
    ap.add_argument("--voltage_max_V", type=float, default=None, help="Only used when --voltage_range_strategy fixed")

    ap.add_argument("--no_prediction", action="store_true", help="Do not save prediction.npz")
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    solution_npz = _resolve_solution_npz(args.solution_npz, args.prepared_dir)
    diagnostics, preset = resolve_profile_adaptive_preset(solution_npz, args.profile_adaptive_mode)
    print(json.dumps({"d9_5_1_profile_adaptive": diagnostics.to_dict()}, ensure_ascii=False))

    transform_keys = [
        "profile_event_gate",
        "profile_dynamic_event_gate",
        "radial_scale_a",
        "radial_scale_c",
        "resistance_ohm",
        "phis_c_head_mode",
        "phis_c_direct_scale",
        "phis_c_correction_scale_V",
        "low_voltage_gate_center_V",
        "low_voltage_gate_width_V",
        "phis_c_low_tail_scale_V",
        "phis_c_event_scale_V",
        "event_current_gain",
        "temperature_polarization_scale_V",
        "ocv_baseline_mix",
        "direct_voltage_mix",
        "ohmic_mix",
        "phie_current_scale_V",
        "phie_correction_scale_V",
        "surface_flux_gain",
        "voltage_range_strategy",
        "voltage_low_percentile",
        "voltage_high_percentile",
        "voltage_margin_V",
        "voltage_floor_V",
        "voltage_ceil_V",
        "voltage_guard_low_V",
        "voltage_guard_high_V",
        "enable_voltage_hard_clamp",
        "voltage_min_V",
        "voltage_max_V",
    ]
    transform_overrides = _override_from_args(preset.get("transform", {}), args, transform_keys)

    loss_map = {
        "voltage_weight": "voltage_data",
        "voltage_tail_weight": "voltage_tail",
        "voltage_bias_weight": "voltage_bias",
        "voltage_range_weight": "voltage_range",
        "voltage_guardrail_weight": "voltage_guardrail",
        "voltage_quantile_weight": "voltage_quantile",
        "voltage_asymmetry_weight": "voltage_asymmetry",
        "voltage_event_weight": "voltage_event",
        "voltage_corr_weight": "voltage_corr",
        "voltage_ultra_quantile_weight": "voltage_ultra_quantile",
        "voltage_low_coverage_weight": "voltage_low_coverage",
        "voltage_tail_balance_weight": "voltage_tail_balance",
        "tail_fraction": "tail_fraction",
        "tail_weight_gain": "tail_weight_gain",
        "low_tail_extra_gain": "low_tail_extra_gain",
        "high_tail_extra_gain": "high_tail_extra_gain",
        "event_weight_gain": "event_weight_gain",
        "huber_delta_V": "huber_delta_V",
        "low_coverage_threshold_V": "low_coverage_threshold_V",
        "high_coverage_threshold_V": "high_coverage_threshold_V",
        "coverage_temperature_V": "coverage_temperature_V",
        "center_weight": "center_symmetry",
        "surface_flux_weight": "surface_flux",
        "radial_smooth_weight": "radial_smooth",
        "potential_smooth_weight": "potential_smooth",
    }
    losses = dict(preset.get("losses", {}))
    for arg_key, loss_key in loss_map.items():
        value = getattr(args, arg_key)
        if value is not None:
            losses[loss_key] = value

    trainer_defaults = preset.get("trainer", {})
    cfg = TrainerConfig(
        solution_npz=solution_npz,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=float(args.lr if args.lr is not None else 7e-4),
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_time_points=args.max_time_points,
        time_window_s=args.time_window_s,
        start_time_s=args.start_time_s,
        prediction_time_points=args.prediction_time_points,
        prediction_radial_points=args.prediction_radial_points,
        log_every=args.log_every,
        save_prediction=not args.no_prediction,
        nominal_capacity_Ah=args.nominal_capacity_Ah,
        current_scale_A=args.current_scale_A,
        event_sampling_mix=float(_pick(args, "event_sampling_mix", trainer_defaults, 0.55)),
        sample_weight_exponent=float(_pick(args, "sample_weight_exponent", trainer_defaults, 1.0)),
        low_voltage_threshold_V=float(_pick(args, "low_voltage_threshold_V", trainer_defaults, 2.75)),
        high_voltage_threshold_V=float(_pick(args, "high_voltage_threshold_V", trainer_defaults, 4.10)),
        low_voltage_quantile=float(_pick(args, "low_voltage_quantile", trainer_defaults, 0.08)),
        high_voltage_quantile=float(_pick(args, "high_voltage_quantile", trainer_defaults, 0.92)),
        high_current_quantile=float(_pick(args, "high_current_quantile", trainer_defaults, 0.90)),
        transition_current_delta_quantile=float(_pick(args, "transition_current_delta_quantile", trainer_defaults, 0.90)),
        temperature_extreme_quantile=float(_pick(args, "temperature_extreme_quantile", trainer_defaults, 0.90)),
        rare_loss_warmup_start_frac=float(_pick(args, "rare_loss_warmup_start_frac", trainer_defaults, 0.30)),
        rare_loss_warmup_full_frac=float(_pick(args, "rare_loss_warmup_full_frac", trainer_defaults, 0.85)),
        rare_loss_warmup_power=float(_pick(args, "rare_loss_warmup_power", trainer_defaults, 1.25)),
        rare_loss_start_scale=float(_pick(args, "rare_loss_start_scale", trainer_defaults, 0.05)),
        rare_loss_final_scale=float(_pick(args, "rare_loss_final_scale", trainer_defaults, 1.0)),
        rare_sample_start_scale=float(_pick(args, "rare_sample_start_scale", trainer_defaults, 0.30)),
        rare_sample_final_scale=float(_pick(args, "rare_sample_final_scale", trainer_defaults, 0.80)),
        model={
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "activation": args.activation,
            "dropout": args.dropout,
        },
        transform=transform_overrides,
        losses=losses,
        profile_adaptive_diagnostics=diagnostics.to_dict(),
    )
    summary = run_training(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
