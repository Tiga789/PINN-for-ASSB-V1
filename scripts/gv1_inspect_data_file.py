#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running from project root without installing as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.io import ReadOptions, read_battery_file, write_standard_cache
from gv1.io.field_mapper import validate_standard_table


def parse_field_map(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"field-map item must be raw=standard, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and standardize one battery data file (.mat/.csv/.parquet).")
    parser.add_argument("--input", required=True, help="Input data file path")
    parser.add_argument("--output", default=None, help="Optional standardized cache output path (.parquet or .csv)")
    parser.add_argument("--preview_csv", default=None, help="Optional preview CSV path")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--batch_id", default=None)
    parser.add_argument("--battery_id", default=None)
    parser.add_argument("--cycle_id", type=int, default=None)
    parser.add_argument("--default_temperature_C", type=float, default=25.0)
    parser.add_argument("--current_sign_convention", default="positive_is_charge", choices=["positive_is_charge", "positive_is_discharge", "as_recorded"])
    parser.add_argument("--infer_time_from_row_index", action="store_true")
    parser.add_argument("--sample_frequency_hz", type=float, default=None)
    parser.add_argument("--mat_table_path", default=None, help="Optional preferred nested .mat table path")
    parser.add_argument("--field_map", action="append", help="Raw-to-standard mapping, e.g. Voltage=voltage_V")
    args = parser.parse_args()

    options = ReadOptions(
        dataset_id=args.dataset_id,
        batch_id=args.batch_id,
        battery_id=args.battery_id,
        cycle_id=args.cycle_id,
        default_temperature_C=args.default_temperature_C,
        current_sign_convention=args.current_sign_convention,
        infer_time_from_row_index=args.infer_time_from_row_index,
        sample_frequency_hz=args.sample_frequency_hz,
        mat_table_path=args.mat_table_path,
        field_map=parse_field_map(args.field_map),
    )
    result = read_battery_file(args.input, options)
    valid, problems = validate_standard_table(result.dataframe)
    summary = {
        "input": str(args.input),
        "source_format": result.source_format,
        "n_rows": int(len(result.dataframe)),
        "columns": list(result.dataframe.columns),
        "standard_columns_present": [c for c in ["time_s", "current_A", "voltage_V", "temperature_C", "capacity_Ah"] if c in result.dataframe.columns],
        "valid_standard_table": valid,
        "validation_problems": problems,
        "warnings": result.warnings,
        "reader_metadata": result.metadata,
        "head": result.dataframe.head(5).to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        write_standard_cache(result, args.output)
    if args.preview_csv:
        p = Path(args.preview_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        result.dataframe.head(200).to_csv(p, index=False, encoding="utf-8-sig")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
