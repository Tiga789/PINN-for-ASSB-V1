from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from gv1.d18_cycleaware.common import (  # noqa: E402
    dump_json,
    load_json,
    resolve_config_path,
    resolve_project_root,
    utc_now_iso,
)
from gv1.d18_cycleaware.model_scaffold import (  # noqa: E402
    D18ModelConfig,
    architecture_contract,
    synthetic_architecture_check,
)


def run_s0(config_path: str | Path, output_root_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    project_root = resolve_project_root(config_path, config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else resolve_config_path(str(paths.get("output_root", "D18_S0_S1_FIX_Output")), config, project_root)
    )
    out = output_root / "d18_s0_architecture_fix"
    out.mkdir(parents=True, exist_ok=True)
    s0 = config.get("s0", {}) if isinstance(config.get("s0"), Mapping) else {}
    model_cfg = D18ModelConfig.from_mapping(s0.get("model"))
    print("[D18-S0] validating cycle-aware operator tensor contract (no training)", flush=True)
    check = synthetic_architecture_check(
        model_cfg,
        seed=int(s0.get("synthetic_seed", 1801)),
        batch_size=int(s0.get("synthetic_batch_size", 3)),
        cycle_count=int(s0.get("synthetic_cycle_count", 7)),
        time_count=int(s0.get("synthetic_time_count", 257)),
    )
    contract = architecture_contract(model_cfg)
    result = {
        "stage": "D18-S0-FIX",
        "created_at_utc": utc_now_iso(),
        "status": check["status"],
        "training_launched": False,
        "architecture_contract": contract,
        "synthetic_check": check,
        "required_data_features": s0.get("required_data_features", []),
        "promotion_rule": "S0-FIX validates tensor interfaces and pointwise physical bounds; S2 remains blocked until fixed S1 manual review.",
    }
    dump_json(contract, out / "d18_s0_architecture_contract.json")
    dump_json(result, out / "d18_s0_validation.json")
    lines = [
        "# D18-S0-FIX Architecture and Physical-Bounds Validation",
        "",
        f"- Status: **{result['status']}**",
        "- Training launched: **False**",
        f"- Parameter count: **{check['parameter_count']:,}**",
        f"- Zero-volume-mean max error A/C: `{check['zero_volume_mean_max_abs_a']:.3e}` / `{check['zero_volume_mean_max_abs_c']:.3e}`",
        f"- Radial shape peak A/C: `{check['radial_shape_peak_max_abs_a']:.6f}` / `{check['radial_shape_peak_max_abs_c']:.6f}`",
        f"- Theta outside fraction A/C: `{check['theta_outside_fraction_a']:.3e}` / `{check['theta_outside_fraction_c']:.3e}`",
        f"- Concentration outside fraction A/C: `{check['concentration_outside_fraction_a']:.3e}` / `{check['concentration_outside_fraction_c']:.3e}`",
        "",
        "## Architecture frozen for S1 review",
        "",
        "1. Cycle-history GRU over full-profile cycle summaries.",
        "2. Causal within-cycle GRU over local I/V/T/step/time features.",
        "3. Shared encoder followed by separate RG/P4D residual adapters.",
        "4. Deterministic cbar baseline plus combined-shape normalized zero-volume-mean radial basis.",
        "5. Pointwise delta-c bounded by admissible inventory margin; cs/theta out-of-range fraction must be zero.",
        "6. Theta derived from cs, not independently predicted.",
        "7. Shared bounded potential gauge plus differential phie/phis_c residuals.",
        "",
        "This file is a design/shape validation, not evidence that the model has been trained or achieves all-cycle accuracy.",
    ]
    (out / "S0_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[D18-S0] status={result['status']} parameters={check['parameter_count']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate D18-S0 architecture without training")
    parser.add_argument("--config", default="configs/d18_s0_s1_fix.json")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = run_s0(args.config, args.output_root)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
