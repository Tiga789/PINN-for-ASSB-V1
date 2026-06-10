from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def pwin(path: str | Path) -> Path:
    return Path(str(path))


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4B collect final scorecard.')
    ap.add_argument('--config', default='configs/d15_p4b_ready18_generation_config.json')
    ap.add_argument('--preflight-json', required=True)
    ap.add_argument('--generation-dir', required=True)
    ap.add_argument('--audit-dir', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    cfg = load_json(args.config, {})
    pre = load_json(args.preflight_json, {})
    gen = load_json(Path(args.generation_dir) / 'D15_P4B_READY18_RG_GENERATION_REPORT.json', {})
    audit = load_json(Path(args.audit_dir) / 'radial_gradient_audit_summary.json', {})
    rows = read_csv_rows(Path(args.audit_dir) / 'radial_gradient_audit_by_profile.csv')

    expected = int(cfg.get('expected_ready_cell_count', 18))
    gen_ok = gen.get('overall_status') == 'PASS' and int(gen.get('generated_count', -1)) == expected and int(gen.get('error_count', 999)) == 0
    audit_ok = audit.get('overall_status') == 'PASS' and int(audit.get('profile_count', -1)) == expected and int(audit.get('read_error_count', 999)) == 0 and int(audit.get('fail_count', 999)) == 0
    pre_ok = pre.get('overall_status') == 'PASS'
    final = 'PASS' if pre_ok and gen_ok and audit_ok else ('FAIL' if not pre_ok or gen.get('overall_status') == 'FAIL' or audit.get('overall_status') == 'FAIL' else 'REVIEW')

    by_batch = {}
    for r in rows:
        pid = str(r.get('profile_id', ''))
        b = ''
        import re
        m = re.search(r'Batch-(\d+)', pid)
        if m:
            b = 'Batch-' + m.group(1)
        if not b:
            b = 'unknown'
        by_batch.setdefault(b, {'count': 0, 'pass': 0, 'warn': 0, 'fail': 0})
        by_batch[b]['count'] += 1
        flag = str(r.get('overall_flag', '')).upper()
        if flag == 'PASS':
            by_batch[b]['pass'] += 1
        elif flag == 'WARN':
            by_batch[b]['warn'] += 1
        elif flag == 'FAIL':
            by_batch[b]['fail'] += 1

    out = {
        'stage': 'D15-P4B remaining-ready 18-cell P2Dlite-RG soft-label generation + radial audit',
        'final_status': final,
        'preflight_status': pre.get('overall_status'),
        'generation_status': gen.get('overall_status'),
        'radial_audit_status': audit.get('overall_status'),
        'expected_ready_cell_count': expected,
        'generated_count': gen.get('generated_count'),
        'generation_error_count': gen.get('error_count'),
        'audit_profile_count': audit.get('profile_count'),
        'audit_pass_count': audit.get('pass_count'),
        'audit_warn_count': audit.get('warn_count'),
        'audit_fail_count': audit.get('fail_count'),
        'audit_read_error_count': audit.get('read_error_count'),
        'by_batch_audit_counts': by_batch,
        'allowed_claim_if_pass': 'P2Dlite-RG soft labels and radial-gradient audit are established for the 18 replay-ready remaining XJTU cells from Batch-1/3/4.',
        'not_allowed_claims': [
            'Do not claim held-out generalization from D15-P4B alone.',
            'Do not claim experimental internal-state truth for cs_a/cs_c.',
            'Do not claim all remaining 32 cells are completed; Batch-5/6 remaining 14 cells still require replay-profile completion.'
        ]
    }
    write_json(out, args.out_json)
    print('[D15-P4B scorecard] final_status:', final)
    return 0 if final == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
