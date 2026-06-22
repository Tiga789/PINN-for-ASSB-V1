from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PACKAGE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from d18_run_s0_s1_fix import run_fix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic end-to-end self-test for D18-S0/S1-FIX")
    parser.add_argument("--keep-output", default=None)
    args = parser.parse_args()

    temp_ctx = tempfile.TemporaryDirectory(prefix="d18_s0_s1_fix_selftest_") if args.keep_output is None else None
    root = Path(args.keep_output).resolve() if args.keep_output else Path(temp_ctx.name)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PACKAGE_ROOT / "configs" / "d18_s0_s1_fix.json").read_text(encoding="utf-8"))
    p0 = root / "prior_p0.json"
    p0.write_text(json.dumps({
        "stage": "D18-P0",
        "status": "PASS",
        "required_missing_count": 0,
        "artifact_error_count": 0,
    }, indent=2), encoding="utf-8")
    cfg["project"]["project_root"] = str(PACKAGE_ROOT)
    cfg["paths"]["output_root"] = str(root / "output")
    cfg["paths"]["prior_p0_manifest"] = str(p0)
    cfg["dense_casepack"]["synthetic_fixture_mode"] = True
    cfg["dense_casepack"]["synthetic_seed"] = 18018
    cfg["dense_casepack"]["dense_min_time_points"] = 512
    cfg["s1"]["dense_min_time_points"] = 512
    config_path = root / "selftest_config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = run_fix(config_path, root / "output", no_plots=True)
    failures: list[str] = []
    if summary.get("status") != "PASS_READY_FOR_HUMAN_ARCHITECTURE_REVIEW":
        failures.append(f"overall status={summary.get('status')}")
    if summary.get("s0_status") != "PASS":
        failures.append(f"S0 status={summary.get('s0_status')}")
    if summary.get("dense_casepack_status") != "PASS" or int(summary.get("dense_case_count", 0)) != 8:
        failures.append("dense casepack did not produce exactly 8 cases")
    if summary.get("s1_status") != "PASS_VALID_DIAGNOSTIC_COVERAGE":
        failures.append(f"S1 status={summary.get('s1_status')}")
    if summary.get("s1_coverage_status") != "PASS":
        failures.append("S1 coverage audit did not pass")
    if summary.get("go_to_s2") is not False or summary.get("training_launched") is not False:
        failures.append("training/S2 safety flags are not false")
    if summary.get("frozen_test_used") is not False:
        failures.append("synthetic self-test unexpectedly used blocked split")

    out = root / "output"
    for path in out.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
            json.loads(text)
        except Exception as exc:
            failures.append(f"invalid JSON {path}: {exc}")
            continue
        if any(token in text for token in (": NaN", ": Infinity", ": -Infinity")):
            failures.append(f"non-strict JSON token in {path}")
    for rel in (
        "d18_s1_array_diagnostic/d18_s1_warnings.csv",
        "d18_s1_array_diagnostic/d18_s1_cycle_boundary_audit.csv",
        "d18_s1_dense_casepack/D18_S1_DENSE_CASEPACK_FAILURES.csv",
    ):
        path = out / rel
        if not path.exists() or not path.read_text(encoding="utf-8-sig").splitlines():
            failures.append(f"CSV missing header: {rel}")

    if failures:
        print("FAIL: D18-S0/S1-FIX self-test")
        for failure in failures:
            print(f"  - {failure}")
        if temp_ctx is not None:
            print(f"  output retained temporarily at: {root}")
        return 2
    print("PASS: D18-S0/S1-FIX synthetic end-to-end self-test")
    print(f"  overall={summary['status']}")
    print(f"  s0={summary['s0_status']} casepack={summary['dense_casepack_status']} s1={summary['s1_status']}")
    print("  training_launched=False go_to_s2=False")
    if temp_ctx is not None:
        temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
