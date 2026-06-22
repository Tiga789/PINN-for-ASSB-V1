from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from gv1.d18_cycleaware.array_io import discover_array_cases  # noqa: E402
from gv1.d18_cycleaware.common import load_json, resolve_config_path, resolve_project_root  # noqa: E402
from gv1.d18_cycleaware.diagnostics import diagnose_case  # noqa: E402
from gv1.d18_cycleaware.reporting import write_s1_reports  # noqa: E402


def run_s1(
    config_path: str | Path,
    output_root_override: str | Path | None = None,
    *,
    no_plots: bool = False,
    prediction_root_override: str | Path | None = None,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = copy.deepcopy(dict(runtime_config)) if runtime_config is not None else load_json(config_path)
    project_root = resolve_project_root(config_path, config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else resolve_config_path(str(paths.get("output_root", "D18_S0_S1_FIX_Output")), config, project_root)
    )
    out = output_root / "d18_s1_array_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    s1 = config.get("s1", {}) if isinstance(config.get("s1"), Mapping) else {}
    if prediction_root_override is not None:
        s1 = dict(s1)
        s1["prediction_roots"] = [str(Path(prediction_root_override).resolve())]
        s1["prediction_globs"] = ["*.npz"]
        config["s1"] = s1
    print("[D18-S1-FIX] reading only explicit dense casepack arrays", flush=True)
    discovery = discover_array_cases(config, project_root)
    print(
        f"[D18-S1-FIX] selected_cases={len(discovery.cases)} candidates={len(discovery.inventory_rows)} warnings={len(discovery.warnings)}",
        flush=True,
    )
    diagnostics = []
    for index, case in enumerate(discovery.cases, start=1):
        print(
            f"[D18-S1-FIX] {index}/{len(discovery.cases)} diagnose {case.canonical_cell_uid} "
            f"split={case.split} protocol={case.protocol} cycles={len(set(case.cycle_id.tolist())) if case.cycle_id is not None else 0} points={case.n_time}",
            flush=True,
        )
        diagnostics.append(diagnose_case(case, s1))
    summary = write_s1_reports(
        output_dir=out,
        cases=discovery.cases,
        diagnostics=diagnostics,
        inventory_rows=discovery.inventory_rows,
        warnings=discovery.warnings,
        s1_config=s1,
        make_plots=bool(s1.get("make_plots", True)) and not no_plots,
    )
    print(
        f"[D18-S1-FIX] status={summary['status']} coverage={summary['coverage_audit']['status']} "
        f"labels={summary['recommendation'].get('labels', [])}",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="D18-S1-FIX explicit dense array diagnostics; no training")
    parser.add_argument("--config", default="configs/d18_s0_s1_fix.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_s1(args.config, args.output_root, no_plots=args.no_plots, prediction_root_override=args.prediction_root)
    return 0 if result["status"] in {
        "PASS_VALID_DIAGNOSTIC_COVERAGE",
        "REVIEW_INVALID_DIAGNOSTIC_COVERAGE",
        "REVIEW_NO_STRUCTURAL_FAILURE_DETECTED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
