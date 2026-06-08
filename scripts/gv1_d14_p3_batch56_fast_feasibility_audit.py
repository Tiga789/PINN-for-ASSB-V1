#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D14-P3 FAST Batch-5/6 feasibility audit.

This script intentionally avoids deep .mat loading. It is a fast feasibility
audit, not a profile generator and not a training step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def status_rank(s: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(s).upper(), 1)


def combine_status(rows: Sequence[dict]) -> str:
    worst = "PASS"
    for r in rows:
        s = str(r.get("status", "WARN")).upper()
        if status_rank(s) > status_rank(worst):
            worst = s
    return worst


def partial_sha256(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        st = path.stat()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        with path.open("rb") as f:
            h.update(f.read(block))
            if st.st_size > block:
                f.seek(max(0, st.st_size - block))
                h.update(f.read(block))
        return h.hexdigest()
    except Exception:
        return ""


def should_skip(path: Path, skip_fragments: Sequence[str]) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(frag.lower() in s for frag in skip_fragments)


def parse_batch(path: Path) -> str:
    s = str(path).replace("\\", "/")
    m = re.search(r"batch[\s_\-]*([56])", s, flags=re.I)
    if m:
        return f"Batch-{m.group(1)}"
    return "unknown"


def parse_protocol(batch: str) -> str:
    if batch == "Batch-5":
        return "random_walk"
    if batch == "Batch-6":
        return "GEO"
    return "unknown"


def parse_cell_uid(path: Path, batch: str) -> str:
    s = path.stem
    pats = [r"battery[\s_\-]*([0-9]+)", r"cell[\s_\-]*([0-9]+)", r"bat[\s_\-]*([0-9]+)", r"([0-9]+)$"]
    for p in pats:
        m = re.search(p, s, flags=re.I)
        if m:
            return f"{batch}_battery-{int(m.group(1))}"
    return f"{batch}_{s}"


def discover(data_root: Path, batches: Sequence[str], exts: Sequence[str], skip_fragments: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    if not data_root.exists():
        return rows
    extset = {e.lower() for e in exts}
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p, skip_fragments):
            continue
        if p.suffix.lower() not in extset:
            continue
        batch = parse_batch(p)
        if batch not in batches:
            continue
        try:
            st = p.stat()
            size_mb = round(st.st_size / 1024 / 1024, 3)
            mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
        except Exception:
            size_mb = ""
            mtime = ""
        rows.append({
            "batch": batch,
            "protocol": parse_protocol(batch),
            "cell_uid": parse_cell_uid(p, batch),
            "file_name": p.name,
            "file_path": str(p),
            "extension": p.suffix.lower(),
            "size_mb": size_mb,
            "mtime_utc": mtime,
            "partial_sha256": partial_sha256(p),
        })
    rows.sort(key=lambda r: (r["batch"], r["cell_uid"], r["file_name"]))
    return rows


def inspect_mat_shallow(path: Path) -> dict:
    out = {
        "loader": "",
        "loader_ok": False,
        "loader_error": "",
        "root_keys": "",
        "mat_variables": "",
        "schema_confidence": "unknown",
        "has_data_root": False,
    }
    try:
        import scipy.io  # type: ignore
        vars_info = scipy.io.whosmat(str(path))
        names = [x[0] for x in vars_info]
        out.update({
            "loader": "scipy.io.whosmat",
            "loader_ok": True,
            "root_keys": "|".join(names[:30]),
            "mat_variables": "|".join([f"{n}:{shape}:{dtype}" for n, shape, dtype in vars_info[:30]]),
            "has_data_root": any(str(n).lower() == "data" for n in names),
            "schema_confidence": "shallow_mat_metadata_only",
        })
        return out
    except Exception as exc:
        out["loader_error"] = f"whosmat failed: {type(exc).__name__}: {exc}"
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
        out.update({
            "loader": "h5py_shallow",
            "loader_ok": True,
            "root_keys": "|".join(map(str, keys[:30])),
            "has_data_root": any(str(k).lower() == "data" for k in keys),
            "schema_confidence": "shallow_hdf5_metadata_only",
        })
    except Exception as exc:
        out["loader"] = "none"
        out["loader_error"] += f" | h5py failed: {type(exc).__name__}: {exc}"
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def has_alias(cols: Sequence[str], aliases: Sequence[str]) -> bool:
    ncols = [norm(c) for c in cols]
    for a in aliases:
        na = norm(a)
        if na in ncols:
            return True
        if any(na in c for c in ncols):
            return True
    return False


def inspect_table_shallow(path: Path) -> dict:
    out = {
        "loader": "",
        "loader_ok": False,
        "loader_error": "",
        "root_keys": "",
        "schema_confidence": "unknown",
        "has_time": False,
        "has_current": False,
        "has_voltage": False,
        "has_temperature": False,
        "has_capacity_like": False,
        "row_count_metadata": "",
    }
    try:
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
                pf = pq.ParquetFile(path)
                cols = pf.schema.names
                out["row_count_metadata"] = sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))
                out["loader"] = "pyarrow.parquet.metadata"
            except Exception:
                import pandas as pd  # type: ignore
                df = pd.read_parquet(path)
                cols = list(df.columns)
                out["row_count_metadata"] = len(df)
                out["loader"] = "pandas.read_parquet"
        else:
            import pandas as pd  # type: ignore
            df = pd.read_csv(path, nrows=1000)
            cols = list(df.columns)
            out["row_count_metadata"] = "sample_1000"
            out["loader"] = "pandas.read_csv_nrows1000"
        out["loader_ok"] = True
        out["root_keys"] = "|".join(map(str, cols[:60]))
        out["has_time"] = has_alias(cols, ["time", "time_s", "system_time", "date_time", "datetime", "test_time"])
        out["has_current"] = has_alias(cols, ["current", "current_a", "current_A", "I", "Current"])
        out["has_voltage"] = has_alias(cols, ["voltage", "voltage_v", "voltage_V", "V", "Voltage"])
        out["has_temperature"] = has_alias(cols, ["temperature", "temperature_C", "temp", "temp_C", "Temperature"])
        out["has_capacity_like"] = has_alias(cols, ["capacity", "Capacity", "q_discharge", "charge_capacity", "discharge_capacity"])
        out["schema_confidence"] = "column_metadata"
    except Exception as exc:
        out["loader"] = "none"
        out["loader_error"] = f"{type(exc).__name__}: {exc}"
    return out


def inspect_file(row: dict) -> dict:
    p = Path(row["file_path"])
    if p.suffix.lower() == ".mat":
        meta = inspect_mat_shallow(p)
        # For raw XJTU .mat, shallow metadata usually exposes only root 'data'.
        has_core_schema = False
        replay_feasibility = "requires_existing_gv1_mat_reader_or_standardization"
        schema_status = "WARN" if meta["loader_ok"] else "FAIL"
    else:
        meta = inspect_table_shallow(p)
        has_core_schema = bool(meta.get("has_time") and meta.get("has_current") and meta.get("has_voltage"))
        replay_feasibility = "ready_for_replay_profile_build" if has_core_schema else "needs_column_mapping_or_standardization"
        schema_status = "PASS" if has_core_schema else ("WARN" if meta["loader_ok"] else "FAIL")
    return {
        **row,
        **meta,
        "has_core_time_current_voltage_schema": has_core_schema,
        "replay_feasibility": replay_feasibility,
        "schema_status": schema_status,
        "full_discharge_candidates": "not_deep_scanned_in_fast_mode",
        "partial_discharge_candidates": "not_deep_scanned_in_fast_mode",
        "soh_policy": "SOH source is original XJTU capacity/cycle data; voltage generator remains SOH-free",
    }


def load_status(dir_s: str, filenames: Sequence[str]) -> Tuple[str, str]:
    if not dir_s:
        return "WARN", "directory not provided"
    d = Path(dir_s)
    if not d.exists():
        return "WARN", f"directory missing: {d}"
    for fn in filenames:
        obj = read_json(d / fn)
        if isinstance(obj, dict):
            st = obj.get("overall_status") or obj.get("status") or obj.get("summary", {}).get("overall_status")
            if st:
                return str(st).upper(), str(d / fn)
    return "WARN", f"no known status json found in {d}"


def md_table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return ""
    s = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        s.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--p0_dir", default="")
    ap.add_argument("--p1_dir", default="")
    ap.add_argument("--p2_dir", default="")
    ap.add_argument("--batches", nargs="*", default=["Batch-5", "Batch-6"])
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else project_root / "configs" / "d14_p3_fast_feasibility_config.json"
    cfg = read_json(cfg_path) or {}
    batches = args.batches or cfg.get("expected_batches", ["Batch-5", "Batch-6"])
    exts = cfg.get("raw_extensions", [".mat", ".csv", ".parquet"])
    skip = cfg.get("skip_path_fragments", ["_gv1_cache", "CacheGV1", "xjtu_d14_", ".git", "__pycache__"])

    print(f"[D14-P3 FAST] start {utc_now()}", flush=True)
    print(f"[D14-P3 FAST] data_root={args.data_root}", flush=True)
    raw_rows = discover(Path(args.data_root), batches, exts, skip)
    print(f"[D14-P3 FAST] discovered raw files={len(raw_rows)}", flush=True)

    schema_rows = []
    for idx, row in enumerate(raw_rows, 1):
        print(f"[D14-P3 FAST] shallow inspect {idx}/{len(raw_rows)}: {row['batch']} {row['file_name']}", flush=True)
        schema_rows.append(inspect_file(row))

    write_csv(output_dir / "D14_P3_RAW_FILE_INDEX.csv", raw_rows)
    write_csv(output_dir / "D14_P3_FILE_SCHEMA_AUDIT.csv", schema_rows)

    # In fast mode, cycle eligibility is policy-level rather than deep subrecord scanning.
    cycle_rows = []
    for row in schema_rows:
        cycle_rows.append({
            "batch": row.get("batch"),
            "protocol": row.get("protocol"),
            "cell_uid": row.get("cell_uid"),
            "file_name": row.get("file_name"),
            "eligibility_mode": "fast_policy_only",
            "soh_label_eligibility": "requires capacity-check / complete-discharge extraction in later profile build",
            "partial_discharge_policy": "replay-only unless complete discharge capacity-check is detected",
        })
    write_csv(output_dir / "D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv", cycle_rows)

    replay_rows = []
    for r in schema_rows:
        replay_rows.append({
            "batch": r.get("batch"),
            "protocol": r.get("protocol"),
            "cell_uid": r.get("cell_uid"),
            "file_name": r.get("file_name"),
            "file_path": r.get("file_path"),
            "schema_status": r.get("schema_status"),
            "loader": r.get("loader"),
            "loader_ok": r.get("loader_ok"),
            "schema_confidence": r.get("schema_confidence"),
            "has_core_time_current_voltage_schema": r.get("has_core_time_current_voltage_schema"),
            "replay_feasibility": r.get("replay_feasibility"),
            "loader_error": r.get("loader_error"),
        })
    write_csv(output_dir / "D14_P3_REPLAY_READINESS.csv", replay_rows)

    batch_rows = []
    expected_per = int(cfg.get("expected_files_per_batch", 8))
    for b in batches:
        subset = [r for r in schema_rows if r.get("batch") == b]
        batch_rows.append({
            "batch": b,
            "protocol": "random_walk" if b == "Batch-5" else ("GEO" if b == "Batch-6" else "unknown"),
            "raw_file_count": sum(1 for r in raw_rows if r.get("batch") == b),
            "expected_file_count": expected_per,
            "loader_ok_count": sum(1 for r in subset if r.get("loader_ok")),
            "schema_pass_count": sum(1 for r in subset if r.get("schema_status") == "PASS"),
            "schema_warn_count": sum(1 for r in subset if r.get("schema_status") == "WARN"),
            "schema_fail_count": sum(1 for r in subset if r.get("schema_status") == "FAIL"),
            "mat_file_count": sum(1 for r in subset if r.get("extension") == ".mat"),
            "table_file_count": sum(1 for r in subset if r.get("extension") in [".csv", ".parquet"]),
        })
    write_csv(output_dir / "D14_P3_BATCH_SUMMARY.csv", batch_rows)

    soh_rows = [
        {"policy_item": "SOH source", "status": "PASS", "detail": "Use original XJTU cycle/capacity data. Do not generate SOH in voltage soft-label generator."},
        {"policy_item": "Batch-5/6 partial cycles", "status": "PASS", "detail": "Partial-discharge cycles are replay-only unless a capacity-check full discharge is detected."},
        {"policy_item": "Voltage replay separation", "status": "PASS", "detail": "Usable I(t), V(t), T(t) records can enter measured-current replay independent of SOH eligibility."},
    ]
    write_csv(output_dir / "D14_P3_SOH_POLICY.csv", soh_rows)

    p0 = load_status(args.p0_dir, ["D14_P0_FREEZE_AUDIT.json"])
    p1 = load_status(args.p1_dir, ["D14_P1_EVIDENCE_BOUNDARY_REPORT.json"])
    p2 = load_status(args.p2_dir, ["D14_P2_GENERALIZATION_SCORECARD_REPORT.json"])
    prereq_status = combine_status([
        {"status": p0[0]}, {"status": p1[0]}, {"status": p2[0]}
    ])

    checks = []
    checks.append({
        "check_id": "P3F-C00",
        "name": "P0/P1/P2 prerequisite status",
        "status": "FAIL" if prereq_status == "FAIL" else ("WARN" if prereq_status == "WARN" else "PASS"),
        "detail": f"P0={p0[0]} ({p0[1]}); P1={p1[0]} ({p1[1]}); P2={p2[0]} ({p2[1]})"
    })
    for b in batches:
        n = sum(1 for r in raw_rows if r.get("batch") == b)
        st = "FAIL" if n == 0 else ("WARN" if n != expected_per else "PASS")
        checks.append({"check_id": f"P3F-C01-{b}", "name": f"{b} raw file discovery", "status": st, "detail": f"found={n}; expected≈{expected_per}"})
    ok = sum(1 for r in schema_rows if r.get("loader_ok"))
    checks.append({"check_id": "P3F-C02", "name": "shallow raw metadata inspection", "status": "PASS" if ok == len(schema_rows) and ok > 0 else ("WARN" if ok > 0 else "FAIL"), "detail": f"loader_ok={ok}/{len(schema_rows)}"})
    mat_warn = sum(1 for r in schema_rows if r.get("extension") == ".mat" and r.get("schema_status") == "WARN")
    checks.append({"check_id": "P3F-C03", "name": "MAT shallow mode boundary", "status": "WARN" if mat_warn else "PASS", "detail": f"{mat_warn} .mat files were inspected shallowly; full time/current/voltage extraction is deferred to existing GV1 standardization/profile build."})
    checks.append({"check_id": "P3F-C04", "name": "SOH policy remains external", "status": "PASS", "detail": "XJTU voltage soft-label generator remains SOH-free."})
    checks.append({"check_id": "P3F-C05", "name": "Batch-1_2C_battery-8 policy unchanged", "status": "PASS", "detail": "P3 fast audit scans Batch-5/6 only and does not unflag Batch-1_2C_battery-8."})

    overall = combine_status(checks)
    if overall == "FAIL":
        recommendation = "Do not proceed until FAIL checks are resolved."
    elif overall == "WARN":
        recommendation = "Proceed to a controlled Batch-5/6 profile-build smoke only; WARN is expected for raw .mat shallow inspection."
    else:
        recommendation = "Batch-5/6 are ready for the next controlled profile-build smoke."

    report = {
        "package": "D14-P3 FAST Batch-5/6 feasibility audit",
        "created_utc": utc_now(),
        "overall_status": overall,
        "recommendation": recommendation,
        "paths": {
            "project_root": args.project_root,
            "data_root": args.data_root,
            "cache_root": args.cache_root,
            "output_dir": args.output_dir,
            "config": str(cfg_path),
            "p0_dir": args.p0_dir,
            "p1_dir": args.p1_dir,
            "p2_dir": args.p2_dir,
        },
        "summary": {
            "raw_file_count": len(raw_rows),
            "schema_file_count": len(schema_rows),
            "batch_summary": batch_rows,
        },
        "checks": checks,
        "boundaries": {
            "does_train": False,
            "deep_mat_loading": False,
            "modifies_gv1_mainline": False,
            "generates_soh_in_voltage_soft_label_generator": False,
            "generates_p2d_internal_state_labels": False,
        }
    }
    write_json(output_dir / "D14_P3_BATCH56_FEASIBILITY_REPORT.json", report)

    md = []
    md.append("# D14-P3 FAST Batch-5/6 Feasibility Audit Report\n")
    md.append(f"Created UTC: `{report['created_utc']}`\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Batch summary\n")
    md.append(md_table(batch_rows, ["batch", "protocol", "raw_file_count", "expected_file_count", "loader_ok_count", "schema_pass_count", "schema_warn_count", "schema_fail_count", "mat_file_count"]))
    md.append("\n## Boundary\n")
    md.append("- This is a fast shallow feasibility audit.\n")
    md.append("- It does not load full `.mat` arrays.\n")
    md.append("- It does not train or modify GV1 mainline.\n")
    md.append("- It does not generate SOH in the voltage soft-label generator.\n")
    (output_dir / "D14_P3_BATCH56_FEASIBILITY_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    readme_patch = f"""# README D14-P3 FAST Patch

D14-P3 FAST audits Batch-5 random-walk and Batch-6 GEO raw data using shallow metadata inspection.

Status: **{overall}**

Recommendation: {recommendation}

Boundary:

- No training.
- No GV1 mainline changes.
- No SOH generation in the XJTU voltage soft-label generator.
- `.mat` files are inspected shallowly only; full extraction is deferred to a later controlled profile-build smoke.
"""
    (output_dir / "README_D14_P3_PATCH.md").write_text(readme_patch, encoding="utf-8")

    outputs = [
        "D14_P3_BATCH56_FEASIBILITY_REPORT.json",
        "D14_P3_BATCH56_FEASIBILITY_REPORT.md",
        "D14_P3_RAW_FILE_INDEX.csv",
        "D14_P3_FILE_SCHEMA_AUDIT.csv",
        "D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv",
        "D14_P3_REPLAY_READINESS.csv",
        "D14_P3_BATCH_SUMMARY.csv",
        "D14_P3_SOH_POLICY.csv",
        "D14_P3_OUTPUT_INDEX.json",
        "D14_P3_RUN_SUMMARY.txt",
        "README_D14_P3_PATCH.md",
    ]
    index = {
        "overall_status": overall,
        "output_dir": args.output_dir,
        "files": [{"name": f, "exists": (output_dir / f).exists()} for f in outputs],
    }
    write_json(output_dir / "D14_P3_OUTPUT_INDEX.json", index)
    (output_dir / "D14_P3_RUN_SUMMARY.txt").write_text(
        "\n".join([
            "D14-P3 FAST Batch-5/6 feasibility audit",
            f"created_utc={report['created_utc']}",
            f"overall_status={overall}",
            f"raw_file_count={len(raw_rows)}",
            f"schema_file_count={len(schema_rows)}",
            f"recommendation={recommendation}",
        ]) + "\n",
        encoding="utf-8"
    )

    print(f"[D14-P3 FAST] overall_status={overall}", flush=True)
    print(f"[D14-P3 FAST] recommendation={recommendation}", flush=True)
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
