# -*- coding: utf-8 -*-
"""
D17-P1: audit no-state-label input protocol.

This script checks the D17 config and split manifest for obvious leakage risks.
It does not prove the future training code is perfect, but it creates a hard
first gate before D17-P2 smoke training.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List


FORBIDDEN_STATE_FIELDS = {
    "cs_a", "cs_c", "theta_a", "theta_c",
    "phie", "phis_c", "phie_soft", "phis_c_soft",
    "cs_a_soft", "cs_c_soft", "theta_a_soft", "theta_c_soft",
    "theta0_oracle", "oracle_shift",
}
FORBIDDEN_LOSS_NAMES = {
    "state_supervised", "softlabel_supervised", "cs_soft", "theta_soft",
    "phie_soft", "phis_c_soft", "L_cs_soft", "L_theta_soft",
    "L_phie_soft", "L_phis_c_soft",
}
ALLOWED_OBS_FIELDS = {
    "t_global_s", "time_s", "I_profile", "current_A", "voltage_exp",
    "temperature_C", "cycle_id", "step_id", "step_type", "protocol",
    "batch", "cell_uid", "source_file", "q_cum_Ah", "q_norm",
}


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # Prefer JSON if possible.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Prefer PyYAML if installed.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        pass
    # Lightweight fallback for the simple config shipped in this package.
    data: Dict[str, Any] = {"_raw_text": text}
    fields_match = re.search(r"allowed_profile_fields:\s*\[(.*?)\]", text, flags=re.S)
    if fields_match:
        data.setdefault("train", {})["allowed_profile_fields"] = [
            x.strip().strip("'\"") for x in fields_match.group(1).split(",") if x.strip()
        ]
    return data


def flatten(obj: Any, prefix: str = "") -> List[tuple]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def token_set_from_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    tokens: Dict[str, List[str]] = {}
    for path, value in flatten(config):
        if isinstance(value, str):
            low = value.strip()
            tokens.setdefault(low, []).append(path)
        elif isinstance(value, bool):
            tokens.setdefault(str(value).lower(), []).append(path)
        elif value is not None:
            tokens.setdefault(str(value), []).append(path)
    return tokens


def scan_source_for_high_risk(project_root: Path, scan_dirs: List[str]) -> List[Dict[str, str]]:
    findings = []
    high_risk_patterns = [
        # direct NPZ/dict indexing of state arrays in non-audit code
        re.compile(r"\[\s*['\"](cs_a|cs_c|theta_a|theta_c|phie|phis_c|phie_soft|phis_c_soft)['\"]\s*\]"),
        re.compile(r"(MSE|mse|loss|criterion).{0,80}(cs_a|cs_c|theta_a|theta_c|phie_soft|phis_c_soft)"),
        re.compile(r"(theta0_oracle|oracle_shift)"),
    ]
    allow_if_path_contains = ("d17_audit_no_state_label_inputs.py", "README", "docs", "audits.py")
    for rel in scan_dirs:
        root = project_root / rel
        if not root.exists():
            continue
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*.py")]
        for p in files:
            relp = str(p.relative_to(project_root)) if p.is_relative_to(project_root) else str(p)
            if any(x in relp for x in allow_if_path_contains):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for pat in high_risk_patterns:
                    if pat.search(line):
                        findings.append({"file": relp, "line": str(lineno), "text": line.strip()[:240]})
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--scan_dir", action="append", default=["gv1/d17_pinn", "scripts", "configs"])
    ns = ap.parse_args()

    config_path = Path(ns.config)
    manifest_path = Path(ns.split_manifest)
    project_root = Path(ns.project_root).resolve()
    out_json = Path(ns.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    findings = []
    tokens = token_set_from_config(config)

    allowed_fields = []
    try:
        allowed_fields = config.get("train", {}).get("allowed_profile_fields", [])
    except Exception:
        allowed_fields = []
    if not allowed_fields:
        findings.append({"severity": "WARN", "where": "config.train.allowed_profile_fields", "message": "No explicit allowed_profile_fields found."})
    for f in allowed_fields:
        if f in FORBIDDEN_STATE_FIELDS:
            findings.append({"severity": "FAIL", "where": "config.train.allowed_profile_fields", "message": f"Forbidden state field listed as allowed input: {f}"})
        elif f not in ALLOWED_OBS_FIELDS:
            findings.append({"severity": "WARN", "where": "config.train.allowed_profile_fields", "message": f"Unknown/nonstandard observed field: {f}"})

    losses = config.get("losses", {}) if isinstance(config, dict) else {}
    for k, v in flatten(losses):
        key_text = k.split(".")[-1]
        if key_text in FORBIDDEN_LOSS_NAMES and str(v).lower() not in ("false", "0", "none"):
            findings.append({"severity": "FAIL", "where": f"config.losses.{k}", "message": f"Forbidden supervised loss appears enabled: {key_text}={v}"})

    ckpt = config.get("checkpoint_selection", {}) if isinstance(config, dict) else {}
    for k, v in flatten(ckpt):
        if any(tok in str(k) for tok in FORBIDDEN_STATE_FIELDS) and str(v).lower() not in ("false", "0", "none"):
            findings.append({"severity": "FAIL", "where": f"config.checkpoint_selection.{k}", "message": f"Checkpoint selection may use state labels: {k}={v}"})

    counts = manifest.get("counts", {})
    if not manifest.get("manifest_hash_sha256"):
        findings.append({"severity": "FAIL", "where": "manifest", "message": "manifest_hash_sha256 missing."})
    for split_name in ("train", "validation", "frozen_test"):
        if counts.get(split_name, 0) <= 0:
            findings.append({"severity": "FAIL", "where": "manifest.counts", "message": f"{split_name} split is empty."})
    if counts.get("flagged_probe", 0) <= 0:
        findings.append({"severity": "WARN", "where": "manifest.counts", "message": "No flagged_probe split found; battery-8 may not have been flagged."})

    source_findings = scan_source_for_high_risk(project_root, ns.scan_dir)
    for sf in source_findings:
        findings.append({"severity": "REVIEW", "where": f"{sf['file']}:{sf['line']}", "message": sf["text"]})

    hard_fail = any(f["severity"] == "FAIL" for f in findings)

    report = {
        "protocol": "D17-P1_NO_STATE_LABEL_INPUT_AUDIT",
        "created_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "pass": not hard_fail,
        "no_state_label_input_audit": not hard_fail,
        "config": str(config_path),
        "split_manifest": str(manifest_path),
        "manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
        "counts": counts,
        "allowed_profile_fields": allowed_fields,
        "forbidden_state_fields": sorted(FORBIDDEN_STATE_FIELDS),
        "findings": findings,
        "interpretation": (
            "PASS means the D17-P1 config/manifest did not declare state soft labels as training inputs/losses. "
            "It does not authorize future training code to read cs/theta/phie/phis arrays."
        ),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS" if report["pass"] else "FAIL", "out_json": str(out_json), "findings": len(findings)}, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
