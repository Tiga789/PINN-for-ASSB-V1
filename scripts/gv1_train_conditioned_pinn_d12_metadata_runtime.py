#!/usr/bin/env python
"""Opt-in D12 metadata runtime wrapper around D9.6/D9.5.1 GV1 trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d12_metadata_runtime import D12MetadataRuntimeConfig, configure_runtime, register_patch  # noqa: E402


def _bool(x: str) -> bool:
    k = str(x).strip().lower()
    if k in {"1", "true", "yes", "y", "on"}: return True
    if k in {"0", "false", "no", "n", "off"}: return False
    raise argparse.ArgumentTypeError(f"Expected bool-like value, got {x!r}")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--metadata_mode", default="off", choices=["off", "zero", "on"])
    ap.add_argument("--metadata_manifest", default=None)
    ap.add_argument("--metadata_profile_id", default=None)
    ap.add_argument("--metadata_feature_columns", default="auto")
    ap.add_argument("--metadata_strict_profile_match", type=_bool, default=True)
    ap.add_argument("--metadata_allow_target_probe", type=_bool, default=False)
    ap.add_argument("--metadata_target_profile_id", default="Batch-1_2C_battery-8")
    ap.add_argument("--metadata_runtime_tag", default="d12_metadata_runtime")
    ap.add_argument("--metadata_print_config", type=_bool, default=True)
    meta, forwarded = ap.parse_known_args(sys.argv[1:])
    cfg = D12MetadataRuntimeConfig(
        mode=meta.metadata_mode,
        metadata_manifest=meta.metadata_manifest,
        profile_id=meta.metadata_profile_id,
        feature_columns=meta.metadata_feature_columns,
        strict_profile_match=meta.metadata_strict_profile_match,
        allow_target_probe=meta.metadata_allow_target_probe,
        target_profile_id=meta.metadata_target_profile_id,
        runtime_tag=meta.metadata_runtime_tag,
    )
    configure_runtime(cfg)
    register_patch()
    if meta.metadata_print_config:
        print(json.dumps({"d12_metadata_runtime": {"ok": True, "config": cfg.to_dict(), "delegates_to": "scripts/gv1_train_conditioned_pinn.py", "source_overwrite": False}}, ensure_ascii=False))
    target = ROOT / "scripts" / "gv1_train_conditioned_pinn.py"
    if not target.exists():
        raise FileNotFoundError(target)
    sys.argv = [str(target)] + forwarded
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
