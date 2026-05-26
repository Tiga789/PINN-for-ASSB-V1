#!/usr/bin/env python
r"""Build a GV1 dataset index by recursively scanning a dataset directory.

Example for your local XJTU path:

python scripts/gv1_build_dataset_index.py ^
  --dataset_root "E:\\XJTU battery dataset" ^
  --dataset_id XJTU ^
  --include_batches Batch-1 Batch-3 Batch-4 ^
  --patterns "*.mat" ^
  --output_dir manifests\xjtu_batch134_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Allow running from a project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.dataset_index import build_dataset_index, write_index_csv, write_index_jsonl, write_summary_json  # noqa: E402


def _load_mapping(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"protocol mapping file does not exist: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML protocol mapping files. Use JSON or install pyyaml.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("protocol mapping must be a JSON/YAML object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a GV1 dataset index.")
    ap.add_argument("--dataset_root", required=True, help="Root directory to scan.")
    ap.add_argument("--dataset_id", default="dataset", help="Dataset identifier, e.g. XJTU.")
    ap.add_argument("--include_batches", nargs="*", default=None, help="Optional batch folders to include, e.g. Batch-1 Batch-3.")
    ap.add_argument("--patterns", nargs="*", default=["*.mat", "*.csv", "*.parquet"], help="Filename patterns to include.")
    ap.add_argument("--output_dir", default="manifests/gv1_dataset_index", help="Output directory.")
    ap.add_argument("--protocol_mapping", default=None, help="Optional JSON/YAML mapping for protocol labels.")
    ap.add_argument("--max_files", type=int, default=None, help="Optional maximum number of files for smoke tests.")
    ap.add_argument("--non_recursive", action="store_true", help="Disable recursive scan.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = _load_mapping(args.protocol_mapping)
    rows, summary = build_dataset_index(
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        include_batches=args.include_batches,
        file_patterns=args.patterns,
        recursive=not args.non_recursive,
        protocol_mapping=mapping,
        max_files=args.max_files,
    )

    csv_path = out_dir / "dataset_index.csv"
    jsonl_path = out_dir / "dataset_index.jsonl"
    summary_path = out_dir / "dataset_index_summary.json"
    write_index_csv(rows, csv_path)
    write_index_jsonl(rows, jsonl_path)
    write_summary_json(summary, summary_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
