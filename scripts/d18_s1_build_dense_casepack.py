from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from gv1.d18_cycleaware.common import load_json, resolve_config_path, resolve_project_root  # noqa: E402
from gv1.d18_cycleaware.dense_casepack import build_dense_casepack  # noqa: E402


DENSE_PATH_KEYS = (
    "split_manifest",
    "semantics_csv",
    "internal_heldout_manifest",
    "candidate_dir",
    "candidate_summary",
    "checkpoint",
)


def resolve_dense_config(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(config))
    dense = runtime.get("dense_casepack", {})
    if not isinstance(dense, dict):
        raise ValueError("config.dense_casepack must be an object")
    for key in DENSE_PATH_KEYS:
        raw = str(dense.get(key, "")).strip()
        if raw:
            dense[key] = str(resolve_config_path(raw, runtime, project_root))
    return runtime


def run_casepack(config_path: str | Path, output_root_override: str | Path | None = None):
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    project_root = resolve_project_root(config_path, config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else resolve_config_path(str(paths.get("output_root", "D18_S0_S1_FIX_Output")), config, project_root)
    )
    runtime = resolve_dense_config(config, project_root)
    print("[D18-S1-FIX] exporting explicit dense pred/true casepack from frozen D17 checkpoint", flush=True)
    result = build_dense_casepack(
        project_root=project_root,
        output_dir=output_root / "d18_s1_dense_casepack",
        config=runtime,
    )
    print(
        f"[D18-S1-FIX] casepack_status={result.status} cases={len(result.case_files)} failures={len(result.failures)}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build explicit D18-S1 dense-cycle casepack; no training")
    parser.add_argument("--config", default="configs/d18_s0_s1_fix.json")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = run_casepack(args.config, args.output_root)
    return 0 if result.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
