from __future__ import annotations

import argparse
import copy
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
from d18_s0_validate_architecture import run_s0  # noqa: E402
from d18_s1_array_latent_diagnostic import run_s1  # noqa: E402
from d18_s1_build_dense_casepack import resolve_dense_config  # noqa: E402
from gv1.d18_cycleaware.dense_casepack import build_dense_casepack  # noqa: E402


def _prior_p0_status(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    raw = str(paths.get("prior_p0_manifest", "")).strip()
    if not raw:
        return {"status": "MISSING", "path": "", "reason": "config.paths.prior_p0_manifest is empty"}
    path = resolve_config_path(raw, config, project_root)
    if not path.exists():
        return {"status": "MISSING", "path": str(path), "reason": "prior P0 manifest does not exist"}
    data = load_json(path)
    status = str(data.get("status", "UNKNOWN")) if isinstance(data, Mapping) else "INVALID"
    return {
        "status": status,
        "path": str(path),
        "stage": data.get("stage") if isinstance(data, Mapping) else None,
        "required_missing_count": data.get("required_missing_count") if isinstance(data, Mapping) else None,
        "artifact_error_count": data.get("artifact_error_count") if isinstance(data, Mapping) else None,
    }


def run_fix(
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
        else resolve_config_path(str(paths.get("output_root", "D18_S0_S1_FIX_Output")), config, project_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("[D18-S0/S1-FIX] diagnostic-only run; no optimizer, epoch, checkpoint update, or S2 training", flush=True)
    p0 = _prior_p0_status(config, project_root)
    print(f"[D18-S0/S1-FIX] prior_p0_status={p0['status']} path={p0.get('path', '')}", flush=True)

    s0 = run_s0(config_path, output_root)

    runtime_config = resolve_dense_config(config, project_root)
    casepack = build_dense_casepack(
        project_root=project_root,
        output_dir=output_root / "d18_s1_dense_casepack",
        config=runtime_config,
    )
    print(
        f"[D18-S0/S1-FIX] dense_casepack={casepack.status} cases={len(casepack.case_files)} failures={len(casepack.failures)}",
        flush=True,
    )

    s1_runtime = copy.deepcopy(runtime_config)
    s1 = run_s1(
        config_path,
        output_root,
        no_plots=no_plots,
        prediction_root_override=output_root / "d18_s1_dense_casepack" / "cases",
        runtime_config=s1_runtime,
    )

    all_fixed_stages_pass = (
        p0.get("status") == "PASS"
        and s0.get("status") == "PASS"
        and casepack.status == "PASS"
        and s1.get("status") == "PASS_VALID_DIAGNOSTIC_COVERAGE"
    )
    overall_status = "PASS_READY_FOR_HUMAN_ARCHITECTURE_REVIEW" if all_fixed_stages_pass else "REVIEW_FIX_OUTPUTS"
    summary = {
        "stage": "D18-S0_S1-FIX",
        "created_at_utc": utc_now_iso(),
        "status": overall_status,
        "training_launched": False,
        "go_to_s2": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "prior_p0": p0,
        "s0_status": s0.get("status"),
        "dense_casepack_status": casepack.status,
        "dense_case_count": len(casepack.case_files),
        "dense_case_failure_count": len(casepack.failures),
        "s1_status": s1.get("status"),
        "s1_coverage_status": s1.get("coverage_audit", {}).get("status"),
        "s1_failed_state_count": s1.get("recommendation", {}).get("failed_state_count"),
        "s1_labels": s1.get("recommendation", {}).get("labels", []),
        "frozen_test_used": s1.get("frozen_test_used"),
        "next_action": "Upload this full output directory for review. D18-S2 remains blocked until the fixed S1 evidence is inspected.",
        "final_goal": "55 cells x all cycles high-accuracy full-cycle generator surrogate",
    }
    dump_json(summary, output_root / "D18_S0_S1_FIX_OVERALL_SUMMARY.json")
    lines = [
        "# D18-S0/S1-FIX Overall Status",
        "",
        f"- Overall: **{overall_status}**",
        f"- Prior P0: **{p0.get('status')}**",
        f"- S0 physical bounds: **{s0.get('status')}**",
        f"- Dense casepack: **{casepack.status}**",
        f"- S1: **{s1.get('status')}**",
        f"- S1 coverage: **{s1.get('coverage_audit', {}).get('status')}**",
        f"- Frozen/test/flagged data used: **{s1.get('frozen_test_used')}**",
        "- Training launched: **False**",
        "- Go to S2: **False**",
        "",
        "The target remains 55 cells × all cycles. This package repairs diagnostic validity; it does not train the D18 model.",
    ]
    (output_root / "D18_S0_S1_FIX_OVERALL_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[D18-S0/S1-FIX] overall={overall_status}; go_to_s2=False", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D18-S0/S1 fix only; no model training")
    parser.add_argument("--config", default="configs/d18_s0_s1_fix.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_fix(args.config, args.output_root, no_plots=args.no_plots)
    return 0 if result["status"] in {"PASS_READY_FOR_HUMAN_ARCHITECTURE_REVIEW", "REVIEW_FIX_OUTPUTS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
