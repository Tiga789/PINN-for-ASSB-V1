from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from gv1.d18_cycleaware.common import dump_json, load_json, resolve_config_path, resolve_project_root, utc_now_iso  # noqa: E402
from d18_p0_freeze import run_p0  # noqa: E402
from d18_s0_validate_architecture import run_s0  # noqa: E402
from d18_s1_array_latent_diagnostic import run_s1  # noqa: E402


def run_all(
    config_path: str | Path,
    output_root_override: str | Path | None = None,
    *,
    no_plots: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    project_root = resolve_project_root(config_path, config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else resolve_config_path(str(paths.get("output_root", "D18_Output")), config, project_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print("[D18] P0 -> S0 -> S1; this command never launches training", flush=True)
    p0 = run_p0(config_path, output_root)
    s0 = run_s0(config_path, output_root)
    s1 = run_s1(config_path, output_root, no_plots=no_plots)
    overall_status = (
        "PASS_READY_FOR_MANUAL_S1_REVIEW"
        if p0["status"] == "PASS" and s0["status"] == "PASS" and s1["status"] == "PASS"
        else "REVIEW"
    )
    summary = {
        "stage": "D18-P0_S0_S1",
        "created_at_utc": utc_now_iso(),
        "status": overall_status,
        "training_launched": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "p0_status": p0["status"],
        "s0_status": s0["status"],
        "s1_status": s1["status"],
        "s1_labels": s1.get("recommendation", {}).get("labels", []),
        "go_to_s2": False,
        "next_action": "Review D18-S1 arrays and explicitly design D18-S2; do not start long training from this package.",
        "final_goal": "55 cells x all cycles high-accuracy generator surrogate with streaming audit",
    }
    dump_json(summary, output_root / "D18_P0_S0_S1_OVERALL_SUMMARY.json")
    lines = [
        "# D18-P0/S0/S1 Overall Status",
        "",
        f"- Overall: **{overall_status}**",
        f"- P0: **{p0['status']}**",
        f"- S0: **{s0['status']}**",
        f"- S1: **{s1['status']}**",
        "- Training launched: **False**",
        "- Go to S2: **False (manual review required)**",
        "",
        "Final target remains 55 cells × all cycles, but promotion requires dense selected-cycle and cycle-wise streaming evidence, not sampled-grid accuracy alone.",
    ]
    (output_root / "D18_P0_S0_S1_OVERALL_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[D18] overall_status={overall_status}; go_to_s2=False", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D18 P0/S0/S1 only; no model training")
    parser.add_argument("--config", default="configs/d18_p0_s0_s1.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_all(args.config, args.output_root, no_plots=args.no_plots)
    return 0 if result["status"] in {"PASS_READY_FOR_MANUAL_S1_REVIEW", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
