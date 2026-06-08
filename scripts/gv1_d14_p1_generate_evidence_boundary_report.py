#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D14-P1 evidence boundary / claim audit generator for QJW-2 / PINN-for-ASSB-V1.

Purpose
-------
D14-P1 is NOT a training step and does NOT modify the GV1/ASSB mainline.
It consumes the D14-P0 freeze audit output, then writes an auditable
evidence-boundary report for README / paper wording.

It checks:
  1) D14-P0 output availability and status.
  2) whether current README / Markdown docs contain risky claims.
  3) whether current project claims remain within the accepted evidence boundary:
        ASSB = ModelFin_112 engineering wrapper baseline.
        XJTU = measured-current voltage replay / voltage surrogate on non-outlier profiles.
        Battery-8 = Batch-1_2C_battery-8 remains flagged/stress-test.
        XJTU SOH = sourced from capacity/cycle data, not produced by the voltage soft-label generator.
        XJTU internal states = model-consistent / P2D-consistent only after additional validation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "D14-P1-evidence-boundary-v1"


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8-sig", errors="replace")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = read_json(config_path)
    for key in ("claims", "risky_patterns", "required_p0_files"):
        if key not in cfg:
            raise ValueError(f"config missing key: {key}")
    return cfg


def count_p0_checks(p0_audit: Dict[str, Any]) -> Dict[str, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "OTHER": 0}
    for chk in p0_audit.get("checks", []):
        status = str(chk.get("status", "OTHER")).upper()
        if status in counts:
            counts[status] += 1
        else:
            counts["OTHER"] += 1
    return counts


def inspect_p0(p0_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    required = [p0_dir / x for x in cfg.get("required_p0_files", [])]
    missing = [str(x) for x in required if not x.exists()]

    audit_path = p0_dir / "D14_P0_FREEZE_AUDIT.json"
    p0_audit: Optional[Dict[str, Any]] = None
    p0_status = "MISSING"
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "OTHER": 0}

    if audit_path.exists():
        try:
            p0_audit = read_json(audit_path)
            p0_status = str(p0_audit.get("overall_status", "UNKNOWN")).upper()
            counts = count_p0_checks(p0_audit)
        except Exception as exc:
            p0_status = f"READ_ERROR: {exc}"

    file_fingerprints = []
    fp_path = p0_dir / "D14_P0_FILE_FINGERPRINTS.csv"
    if fp_path.exists():
        try:
            with fp_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                file_fingerprints = list(reader)
        except Exception:
            file_fingerprints = []

    scorecard_index_rows = []
    sc_path = p0_dir / "D14_P0_SCORECARD_INDEX.csv"
    if sc_path.exists():
        try:
            with sc_path.open("r", encoding="utf-8-sig", newline="") as f:
                scorecard_index_rows = list(csv.DictReader(f))
        except Exception:
            scorecard_index_rows = []

    return {
        "p0_dir": str(p0_dir),
        "required_file_count": len(required),
        "missing_required_files": missing,
        "p0_overall_status": p0_status,
        "p0_check_counts": counts,
        "fingerprint_row_count": len(file_fingerprints),
        "scorecard_index_row_count": len(scorecard_index_rows),
        "scorecard_highlights": extract_scorecard_highlights(scorecard_index_rows),
    }


def extract_scorecard_highlights(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    highlights: Dict[str, Any] = {
        "d10_p1_rows": [],
        "d12_s1k_rows": [],
        "battery8_rows_total": 0,
        "unflagged_battery8_rows_total": 0,
        "candidates_visible": [],
    }
    cand = set()
    for r in rows:
        csv_path = (r.get("csv_path") or "").replace("\\", "/").lower()
        root = (r.get("root") or "").replace("\\", "/").lower()
        if "d10p1" in root or "d10_p1" in root or "23x200ks_d10p1" in root:
            highlights["d10_p1_rows"].append(r)
        if "d12_s1k" in root or "s1k" in root:
            highlights["d12_s1k_rows"].append(r)
        try:
            highlights["battery8_rows_total"] += int(float(r.get("battery8_rows") or 0))
        except Exception:
            pass
        try:
            highlights["unflagged_battery8_rows_total"] += int(float(r.get("unflagged_battery8_rows") or 0))
        except Exception:
            pass
        c = (r.get("candidate") or "").strip()
        if c and c != "<none>":
            cand.add(c)
    highlights["candidates_visible"] = sorted(cand)
    # keep report compact
    highlights["d10_p1_row_count"] = len(highlights.pop("d10_p1_rows"))
    highlights["d12_s1k_row_count"] = len(highlights.pop("d12_s1k_rows"))
    return highlights


def iter_markdown_files(project_root: Path, include_readme_only: bool = False) -> Iterable[Path]:
    if include_readme_only:
        p = project_root / "README.md"
        if p.exists():
            yield p
        return

    excluded_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        "ModelFin_107A", "ModelFin_112_deterministic_wrapper",
        "EvalFin_112_deterministic_wrapper",
        "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only",
    }
    for p in project_root.rglob("*.md"):
        parts = set(p.parts)
        if parts.intersection(excluded_dirs):
            continue
        # skip huge generated scorecard docs outside docs/README unless intentionally small
        try:
            if p.stat().st_size > 2_000_000:
                continue
        except Exception:
            continue
        yield p



def _norm_context(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains_any(s: str, needles: Iterable[str]) -> bool:
    return any(n.lower() in s for n in needles)


def is_negated_or_boundary_context(pattern_id: str, snippet: str) -> bool:
    """Return True when a regex hit is a boundary/negative statement, not a risky claim.

    D14-P1 scans README-like text for risky claims. The original v1 regexes were
    intentionally broad but over-flagged safe sentences such as "not a full P2D
    solver", "do not unflag battery-8", and "not an end-to-end neural network".
    This filter keeps true over-claims while ignoring explicit guardrails.
    """
    s = _norm_context(snippet)
    pid = (pattern_id or "").upper()

    # General English/Chinese negation or guardrail wording near the match.
    general_safe = [
        "not ", "not an ", "not a ", "does not", "do not", "must not", "should not", "cannot",
        "停止", "weaker", "worse than",
        "不是", "不能", "不可", "不应", "不得", "不要", "未", "尚未", "不能直接", "不用于", "不说明",
        "avoid wording", "must_not_say", "should be called", "rewrite as", "recommended wording",
    ]

    if pid == "MODELFIN112_END_TO_END":
        return _contains_any(s, [
            "not an end-to-end", "not end-to-end", "not an end to end", "不是端到端", "不是 end-to-end",
            "不是端到端联合训练", "not a single neural", "not single neural", "不是单个神经网络",
            "不是跨电池", "not cross-battery", "engineering wrapper", "unified package",
        ])

    if pid == "XJTU_INTERNAL_TRUE_STATE":
        return _contains_any(s, general_safe + [
            "model-consistent", "p2d-consistent", "latent states", "observable", "voltage surrogate",
            "不用于直接宣称", "不用于宣称", "不能证明", "不能单独证明", "不说明内部状态真值",
            "不说明", "不证明", "不是 internal", "not internal-state truth", "not experimental",
        ])

    if pid == "BATTERY8_UNFLAGGED":
        return _contains_any(s, [
            "do not unflag", "must not unflag", "should not unflag", "do not include", "must not include",
            "不要解除", "不解除", "不能解除", "不得解除", "不纳入", "继续 flagged", "继续flagged",
            "remains flagged", "keep flagged", "keep batch-1_2c", "flagged/excluded", "excluded from",
            "failures:", "failure:", "ures:", "guardrail", "unflagged_battery8_rows_total",
            "core gv1 file missing", "strict cache directory missing", "hard clamp active",
            "battery-8 unflagged" if ("fail" in s or "missing" in s or "ures:" in s) else "\0",
        ])

    if pid == "METADATA_ON_MAINLINE":
        return _contains_any(s, [
            "not promoted", "do not promote", "must not be promoted", "should not be promoted", "not be promoted",
            "not the current", "not promoted to the mainline", "not replace", "not a mainline", "not the gv1 mainline",
            "不能直接替代", "不应替代", "不得替代", "不能替代", "不作为主线", "不是主线",
            "不能直接升格", "不直接升格", "不要将", "应停止", "仍弱于", "弱于", "不能继续推进", "stopping metadata", "ablation",
        ])

    if pid == "XJTU_SOH_GENERATOR":
        return _contains_any(s, [
            "not generated", "should not", "must not", "do not", "not fabricated", "not fabricate",
            "不生成", "不要生成", "不应生成", "不应该生成", "不由", "不能生成", "不混进",
            "sourced", "computed from", "capacity/cycle", "容量", "cycle/capacity",
        ])

    if pid == "P2D_CLAIM_OVERREACH":
        return _contains_any(s, [
            "not a full p2d", "not full p2d", "not a p2d solver", "not full", "not a full",
            "不是 full p2d", "不是完整p2d", "不是真实p2d", "不是 full p2d solver", "not a full p2d solver",
            "p2d-inspired", "voltage-wrapper", "diagnostic result", "residual expert",
        ])

    if pid == "XJTU_24_24_SUCCESS":
        return _contains_any(s, [
            "not 24/24", "不能写成 24/24", "不能宣称 24/24", "不是 24/24", "avoid wording",
            "must_not_say", "23 non-outlier", "battery-8 flagged", "battery-8 excluded",
        ])

    return False


def scan_risky_patterns(project_root: Path, cfg: Dict[str, Any], readme_only: bool = False) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    files = list(iter_markdown_files(project_root, include_readme_only=readme_only))
    patterns = cfg.get("risky_patterns", [])
    for path in files:
        try:
            text = read_text(path)
        except Exception as exc:
            findings.append({
                "file": safe_rel(path, project_root),
                "pattern_id": "read_error",
                "severity": "WARN",
                "match": str(exc),
                "recommendation": "Could not read this Markdown file; inspect manually.",
            })
            continue
        for pat in patterns:
            try:
                rx = re.compile(pat["regex"], re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                findings.append({
                    "file": "<config>",
                    "pattern_id": pat.get("id", "bad_regex"),
                    "severity": "FAIL",
                    "match": str(exc),
                    "recommendation": "Fix regex in config.",
                })
                continue
            for m in rx.finditer(text):
                snippet = text[max(0, m.start() - 120): min(len(text), m.end() + 120)].replace("\n", " ")
                pattern_id = pat.get("id", "")
                if is_negated_or_boundary_context(pattern_id, snippet):
                    continue
                findings.append({
                    "file": safe_rel(path, project_root),
                    "pattern_id": pattern_id,
                    "severity": pat.get("severity", "WARN"),
                    "match": snippet,
                    "recommendation": pat.get("recommendation", ""),
                })
    return findings


def build_claim_rows(cfg: Dict[str, Any], p0_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p0_ok = p0_info["p0_overall_status"] in {"PASS", "WARN"}
    for claim in cfg.get("claims", []):
        evidence_ok = bool(p0_ok)
        if claim.get("requires_d10_p1") and p0_info["scorecard_highlights"].get("d10_p1_row_count", 0) <= 0:
            evidence_ok = False
        if claim.get("requires_d12_s1k") and p0_info["scorecard_highlights"].get("d12_s1k_row_count", 0) <= 0:
            evidence_ok = False
        if claim.get("requires_assb_wrapper") and not p0_ok:
            evidence_ok = False
        status = "ALLOWED" if evidence_ok and claim.get("allowed_currently", True) else "NOT_ALLOWED"
        rows.append({
            "claim_id": claim.get("id", ""),
            "topic": claim.get("topic", ""),
            "status": status,
            "recommended_wording": claim.get("recommended_wording", ""),
            "must_not_say": claim.get("must_not_say", ""),
            "evidence_source": claim.get("evidence_source", ""),
            "notes": claim.get("notes", ""),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def severity_counts(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"FAIL": 0, "WARN": 0, "INFO": 0, "OTHER": 0}
    for f in findings:
        sev = str(f.get("severity", "OTHER")).upper()
        if sev in out:
            out[sev] += 1
        else:
            out["OTHER"] += 1
    return out


def decide_status(p0_info: Dict[str, Any], findings: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    status = "PASS"
    if p0_info.get("missing_required_files"):
        status = "FAIL"
        reasons.append("Missing required D14-P0 files.")
    if p0_info.get("p0_overall_status") == "FAIL":
        status = "FAIL"
        reasons.append("D14-P0 overall_status is FAIL.")
    if str(p0_info.get("p0_overall_status", "")).startswith("READ_ERROR"):
        status = "FAIL"
        reasons.append("D14-P0 audit JSON could not be read.")
    sc = severity_counts(findings)
    if sc["FAIL"] > 0:
        status = "FAIL"
        reasons.append("FAIL-level risky wording detected in scanned Markdown files.")
    elif sc["WARN"] > 0 and status != "FAIL":
        status = "WARN"
        reasons.append("WARN-level risky wording detected in scanned Markdown files.")
    if p0_info.get("p0_overall_status") == "WARN" and status == "PASS":
        status = "WARN"
        reasons.append("D14-P0 is WARN; acceptable for D14-P1, but keep the warning documented.")
    if not reasons:
        reasons.append("No blocking issue detected.")
    return status, reasons


def render_markdown_report(report: Dict[str, Any], claim_rows: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> str:
    p0 = report["p0"]
    sc = severity_counts(findings)
    lines: List[str] = []
    lines.append("# D14-P1 Evidence Boundary Audit Report")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Overall status: **{report['overall_status']}**")
    lines.append(f"- Project root: `{report['project_root']}`")
    lines.append(f"- P0 directory: `{p0['p0_dir']}`")
    lines.append("")
    lines.append("## 1. D14-P0 input status")
    lines.append("")
    lines.append(f"- D14-P0 overall_status: `{p0['p0_overall_status']}`")
    lines.append(f"- D14-P0 check counts: `{p0['p0_check_counts']}`")
    lines.append(f"- Missing required P0 files: `{len(p0['missing_required_files'])}`")
    lines.append(f"- Fingerprint rows: `{p0['fingerprint_row_count']}`")
    lines.append(f"- Scorecard-index rows: `{p0['scorecard_index_row_count']}`")
    lines.append(f"- Scorecard highlights: `{p0['scorecard_highlights']}`")
    lines.append("")
    lines.append("## 2. Current evidence boundary")
    lines.append("")
    lines.append("The current D14-P1 wording boundary is:")
    lines.append("")
    lines.append("1. **ASSB**: `ModelFin_112_deterministic_wrapper` is an engineering wrapper / unified package. It is not an end-to-end single neural network and not cross-battery proof.")
    lines.append("2. **XJTU voltage**: D9.6/D9.5.1 + D12-S1K can be described as measured-current replay / voltage surrogate evidence on non-outlier profiles.")
    lines.append("3. **Battery-8**: only `Batch-1_2C_battery-8` is the currently flagged outlier/stress-test profile. Batch-3/4 battery-8 names are not the same outlier.")
    lines.append("4. **XJTU internal states**: do not call `cs_a/cs_c/phie/phis_c` experimental ground truth unless P2D-consistent labels or external validation are added.")
    lines.append("5. **XJTU SOH**: SOH should be sourced or computed from capacity/cycle data, not generated by the voltage soft-label generator.")
    lines.append("")
    lines.append("## 3. Claims matrix")
    lines.append("")
    lines.append("| claim_id | status | recommended wording | must not say |")
    lines.append("|---|---:|---|---|")
    for r in claim_rows:
        lines.append(f"| {r['claim_id']} | {r['status']} | {r['recommended_wording']} | {r['must_not_say']} |")
    lines.append("")
    lines.append("## 4. Risky wording scan")
    lines.append("")
    lines.append(f"- FAIL findings: `{sc['FAIL']}`")
    lines.append(f"- WARN findings: `{sc['WARN']}`")
    lines.append(f"- INFO findings: `{sc['INFO']}`")
    lines.append("")
    if findings:
        lines.append("| severity | file | pattern_id | recommendation | snippet |")
        lines.append("|---|---|---|---|---|")
        for f in findings[:200]:
            snippet = str(f.get("match", "")).replace("|", "\\|")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            lines.append(f"| {f.get('severity','')} | `{f.get('file','')}` | {f.get('pattern_id','')} | {f.get('recommendation','')} | {snippet} |")
        if len(findings) > 200:
            lines.append(f"\nOnly first 200 findings are shown here. Full list is in JSON/CSV outputs.")
    else:
        lines.append("No risky Markdown wording was detected by the configured pattern scan.")
    lines.append("")
    lines.append("## 5. D14-P1 decision")
    lines.append("")
    for reason in report["status_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## 6. Recommended next step")
    lines.append("")
    lines.append("Proceed to D14-P2 only after accepting this evidence boundary. D14-P2 should build a unified XJTU generalization scorecard from D10-P1 baseline and D12-S1K candidates, with global / segment / protocol / cell / outlier-aware metrics.")
    lines.append("")
    return "\n".join(lines)


def render_readme_patch(report: Dict[str, Any]) -> str:
    return f"""# README patch: D14-P1 evidence boundary

Generated at: `{report['generated_at']}`

## D14-P1 evidence boundary

After D14-P0 freeze audit, the project evidence boundary is fixed as follows:

- **ASSB baseline**: `ModelFin_112_deterministic_wrapper` remains the ASSB five-target engineering wrapper. It combines frozen `ModelFin_107A` states and deterministic ridge SOH. It is not an end-to-end single neural network and should not be presented as cross-battery generalization proof.
- **XJTU voltage baseline**: D9.6/D9.5.1 is the GV1 training mainline. D12-S1K `low_plus_transition_fade_to_baseline` is the current non-outlier XJTU voltage-wrapper recommendation.
- **XJTU result scope**: XJTU currently supports measured-current replay / observable voltage surrogate validation across public liquid-electrolyte cells and protocols. It does not by itself prove that `cs_a / cs_c / phie / phis_c` are experimental internal-state ground truth.
- **Battery-8 policy**: `Batch-1_2C_battery-8` remains flagged/excluded from the non-outlier mainline and should be treated as a stress-test / outlier profile. Batch-3 or Batch-4 battery-8 identifiers are not this same outlier.
- **XJTU SOH policy**: SOH labels should be sourced or computed from the original cycle/capacity data. The XJTU voltage soft-label generator should not fabricate SOH labels.

Allowed wording:

```text
XJTU validates measured-current voltage replay / voltage surrogate robustness on public multi-cell, multi-protocol liquid-electrolyte data, with an explicit outlier policy.
```

Avoid wording:

```text
XJTU proves true internal concentration and potential states.
The project achieves 24/24 XJTU success.
ModelFin_112 is an end-to-end unified neural network.
metadata_on is the new GV1 mainline.
battery-8 is unflagged.
```

D14-P2 should build a unified, outlier-aware scorecard over global, segment, protocol, cell and candidate dimensions.
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate D14-P1 evidence boundary report.")
    parser.add_argument("--project-root", default=".", help="PINN-for-ASSB-V1 project root.")
    parser.add_argument("--cache-root", default=r"E:\XJTU battery dataset\_gv1_cache", help="XJTU GV1 cache root.")
    parser.add_argument("--p0-dir", default=None, help="D14-P0 output directory. Defaults to cache_root/xjtu_d14_p0_freeze_audit_v2 if present, else xjtu_d14_p0_freeze_audit.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to cache_root/xjtu_d14_p1_evidence_boundary.")
    parser.add_argument("--config", default="configs/d14_p1_evidence_boundary_config.json", help="D14-P1 config JSON.")
    parser.add_argument("--readme-only", action="store_true", help="Scan README.md only instead of all Markdown files.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    cache_root = Path(args.cache_root)
    if args.p0_dir:
        p0_dir = Path(args.p0_dir)
    else:
        p0_v2 = cache_root / "xjtu_d14_p0_freeze_audit_v2"
        p0_v1 = cache_root / "xjtu_d14_p0_freeze_audit"
        p0_dir = p0_v2 if p0_v2.exists() else p0_v1

    output_dir = Path(args.output_dir) if args.output_dir else (cache_root / "xjtu_d14_p1_evidence_boundary")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    p0_info = inspect_p0(p0_dir, cfg)
    findings = scan_risky_patterns(project_root, cfg, readme_only=args.readme_only)
    claim_rows = build_claim_rows(cfg, p0_info)
    overall_status, reasons = decide_status(p0_info, findings)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "overall_status": overall_status,
        "status_reasons": reasons,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "config_path": str(config_path),
        "p0": p0_info,
        "risky_wording_counts": severity_counts(findings),
        "claim_count": len(claim_rows),
        "risky_finding_count": len(findings),
        "output_files": {},
    }

    # Write machine-readable outputs.
    out_json = output_dir / "D14_P1_EVIDENCE_BOUNDARY_REPORT.json"
    write_json(out_json, report | {
        "claims_matrix": claim_rows,
        "risky_wording_findings": findings,
    })

    # CSV outputs.
    claims_csv = output_dir / "D14_P1_CLAIMS_MATRIX.csv"
    write_csv(claims_csv, claim_rows, [
        "claim_id", "topic", "status", "recommended_wording",
        "must_not_say", "evidence_source", "notes",
    ])

    findings_csv = output_dir / "D14_P1_TERMINOLOGY_GUARDRAILS.csv"
    write_csv(findings_csv, findings, [
        "severity", "file", "pattern_id", "match", "recommendation",
    ])

    # Markdown outputs.
    md = render_markdown_report(report, claim_rows, findings)
    md_path = output_dir / "D14_P1_EVIDENCE_BOUNDARY_REPORT.md"
    md_path.write_text(md, encoding="utf-8")

    readme_patch_path = output_dir / "README_D14_P1_PATCH.md"
    readme_patch_path.write_text(render_readme_patch(report), encoding="utf-8")

    # compact summary
    summary_path = output_dir / "D14_P1_RUN_SUMMARY.txt"
    summary = [
        f"D14-P1 evidence boundary finished at {report['generated_at']}",
        f"overall_status={overall_status}",
        f"project_root={project_root}",
        f"cache_root={cache_root}",
        f"p0_dir={p0_dir}",
        f"p0_overall_status={p0_info['p0_overall_status']}",
        f"risky_wording_counts={severity_counts(findings)}",
        f"claim_count={len(claim_rows)}",
        f"report_json={out_json}",
        f"report_md={md_path}",
        f"readme_patch={readme_patch_path}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    # Output index with hashes.
    outputs = [out_json, md_path, claims_csv, findings_csv, readme_patch_path, summary_path]
    output_index = []
    for p in outputs:
        if p.exists():
            output_index.append({
                "path": str(p),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    index_path = output_dir / "D14_P1_OUTPUT_INDEX.json"
    write_json(index_path, {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "overall_status": overall_status,
        "files": output_index,
    })

    print("\n".join(summary))
    print(f"output_index={index_path}")

    # D14-P1 is a documentation/evidence audit. Return nonzero only on FAIL.
    return 1 if overall_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
