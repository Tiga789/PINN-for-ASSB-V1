from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d18_s2.common import resolve_config
from gv1.d18_s2.preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only D18-S2 preflight; no training")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "d18_s2_preflight_micro_smoke.json"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = resolve_config(args.config, project_root=args.project_root)
    output = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["output_root"]) / "d18_s2_preflight_only"
    result = run_preflight(
        cfg,
        project_root=Path(args.project_root).resolve(),
        output_dir=output,
        progress=lambda m: print(f"[D18-S2 preflight] {m}", flush=True),
    )
    status = result["summary"]["status"]
    print(f"D18-S2 preflight status: {status}", flush=True)
    return 0 if status == "PASS_PREFLIGHT_FOR_MICRO_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
