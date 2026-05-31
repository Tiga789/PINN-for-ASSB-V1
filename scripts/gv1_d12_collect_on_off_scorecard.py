#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect D12 metadata_off vs metadata_on metrics after a separate ablation has been run.

This script does not train. It scans supplied output folders for JSON metrics and
builds a conservative comparison table. It is safe to run before training; it will
report missing metrics instead of failing noisily.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_OUT_DIR = r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_scorecard"


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _profile_id_from_json(path: Path, data: Dict[str, Any]) -> str:
    for key in ["profile_id", "label", "run_id", "cell_uid", "profile_uid"]:
        val = data.get(key)
        if val:
            return str(val)
    for nested_key in ["profile", "config", "metadata"]:
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for key in ["profile_id", "label", "cell_uid", "run_id"]:
                val = nested.get(key)
                if val:
                    return str(val)
    return path.parent.name


def _extract_metrics(path: Path, mode: str) -> Optional[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    # Skip scorecard summary jsons that are not per-profile metrics unless they contain enough fields.
    metric_keys = ["mae_V", "MAE", "mae", "rmse_V", "corr", "bias_V", "status"]
    if not any(k in data for k in metric_keys) and "metrics" not in data:
        return None
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        metrics = data
    row: Dict[str, Any] = {
        "mode": mode,
        "profile_id": _profile_id_from_json(path, data),
        "json_path": str(path),
        "status": data.get("status", metrics.get("status", "")),
    }
    aliases = {
        "mae_V": ["mae_V", "MAE", "mae", "voltage_mae_V"],
        "rmse_V": ["rmse_V", "RMSE", "rmse", "voltage_rmse_V"],
        "corr": ["corr", "pearson_corr", "voltage_corr"],
        "bias_V": ["bias_V", "bias", "voltage_bias_V"],
        "pred_max_V": ["pred_max_V", "pred_max", "v_pred_max"],
        "pred_upper_frac_ge_4p269": ["pred_upper_frac_ge_4p269"],
        "pred_overshoot_frac_gt_4p35": ["pred_overshoot_frac_gt_4p35"],
    }
    for out_key, candidates in aliases.items():
        for c in candidates:
            if c in metrics:
                row[out_key] = metrics[c]
                break
            if c in data:
                row[out_key] = data[c]
                break
    return row


def _scan(root: Path, mode: str) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    patterns = ["*metrics*.json", "metrics*.json", "*summary*.json"]
    seen = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            row = _extract_metrics(path, mode)
            if row:
                rows.append(row)
    # De-duplicate by profile_id using the shortest path / most metrics-rich row.
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = str(row.get("profile_id", ""))
        richness = sum(1 for k, v in row.items() if v not in [None, ""])
        old = best.get(pid)
        old_richness = sum(1 for k, v in old.items() if v not in [None, ""]) if old else -1
        if old is None or richness > old_richness:
            best[pid] = row
    return list(best.values())


def _compare(off_rows: List[Dict[str, Any]], on_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    off = {str(r.get("profile_id")): r for r in off_rows}
    on = {str(r.get("profile_id")): r for r in on_rows}
    keys = sorted(set(off) | set(on))
    out: List[Dict[str, Any]] = []
    for k in keys:
        r0 = off.get(k, {})
        r1 = on.get(k, {})
        row: Dict[str, Any] = {"profile_id": k, "has_off": bool(r0), "has_on": bool(r1)}
        for metric in ["mae_V", "rmse_V", "corr", "bias_V", "pred_max_V", "pred_upper_frac_ge_4p269", "pred_overshoot_frac_gt_4p35"]:
            v0 = _as_float(r0.get(metric))
            v1 = _as_float(r1.get(metric))
            row[f"off_{metric}"] = r0.get(metric, "")
            row[f"on_{metric}"] = r1.get(metric, "")
            if v0 is not None and v1 is not None:
                row[f"delta_on_minus_off_{metric}"] = v1 - v0
        row["off_status"] = r0.get("status", "")
        row["on_status"] = r1.get("status", "")
        out.append(row)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Collect D12 metadata_on/off scorecard after separate ablation runs.")
    ap.add_argument("--metadata_off_dir", required=True, help="Directory containing metadata_off run outputs.")
    ap.add_argument("--metadata_on_dir", required=True, help="Directory containing metadata_on run outputs.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    off_dir = Path(args.metadata_off_dir)
    on_dir = Path(args.metadata_on_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    off_rows = _scan(off_dir, "metadata_off")
    on_rows = _scan(on_dir, "metadata_on")
    compare_rows = _compare(off_rows, on_rows)

    summary = {
        "ok": True,
        "stage": "D12 metadata on/off scorecard collection",
        "created_at": _now(),
        "metadata_off_dir": str(off_dir),
        "metadata_on_dir": str(on_dir),
        "out_dir": str(out_dir),
        "off_metric_rows": len(off_rows),
        "on_metric_rows": len(on_rows),
        "common_profile_rows": sum(1 for r in compare_rows if r.get("has_off") and r.get("has_on")),
        "missing_off_rows": sum(1 for r in compare_rows if not r.get("has_off")),
        "missing_on_rows": sum(1 for r in compare_rows if not r.get("has_on")),
        "interpretation": "This collector only compares available result JSON files; it does not train or validate a runtime metadata model patch.",
    }
    _write_csv(out_dir / "d12_metadata_off_metrics.csv", off_rows)
    _write_csv(out_dir / "d12_metadata_on_metrics.csv", on_rows)
    _write_csv(out_dir / "d12_metadata_on_off_comparison.csv", compare_rows)
    _write_json(out_dir / "d12_metadata_on_off_scorecard_summary.json", summary)
    md = f"""# D12 Metadata On/Off Scorecard

```text
off_metric_rows={len(off_rows)}
on_metric_rows={len(on_rows)}
common_profile_rows={summary['common_profile_rows']}
missing_off_rows={summary['missing_off_rows']}
missing_on_rows={summary['missing_on_rows']}
```

This is a collector only. Absence of metrics means the separate D12 runtime ablation has not been run or used a non-standard output layout.
"""
    (out_dir / "D12_SCORECARD_SUMMARY.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
