from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


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
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4D full: collect final scorecard from generation and radial audit.')
    ap.add_argument('--config', default='configs/d15_p4d_full_remaining14_config.json')
    ap.add_argument('--generation-json', required=True)
    ap.add_argument('--audit-dir', required=True)
    ap.add_argument('--resource-csv', default=None)
    ap.add_argument('--gpu-csv', default=None)
    ap.add_argument('--process-csv', default=None)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    cfg = load_json(args.config)
    gen = load_json(args.generation_json, {})
    audit = load_json(Path(args.audit_dir) / 'radial_gradient_audit_summary.json', {})
    audit_rows = read_csv_rows(Path(args.audit_dir) / 'radial_gradient_audit_by_profile.csv')
    expected = int(cfg.get('target_cell_count', 14))

    gen_ok = gen.get('overall_status') == 'PASS' and int(gen.get('generated_count', -1)) == expected and int(gen.get('error_count', 999)) == 0
    audit_ok = audit.get('overall_status') == 'PASS' and int(audit.get('profile_count', -1)) == expected and int(audit.get('fail_count', 999)) == 0 and int(audit.get('read_error_count', 999)) == 0

    resource_summary = {}
    res_rows = read_csv_rows(args.resource_csv) if args.resource_csv else []
    if res_rows:
        def _num(k):
            vals = []
            for r in res_rows:
                try:
                    if r.get(k, '') != '':
                        vals.append(float(r[k]))
                except Exception:
                    pass
            return vals
        cpu = _num('cpu_total_percent')
        mem = _num('memory_available_mb')
        py = _num('python_cell_processes')
        resource_summary = {
            'cpu_peak_percent': max(cpu) if cpu else None,
            'cpu_mean_percent': sum(cpu)/len(cpu) if cpu else None,
            'memory_available_min_mb': min(mem) if mem else None,
            'python_cell_processes_max': max(py) if py else None,
        }
    gpu_summary = {}
    gpu_rows = read_csv_rows(args.gpu_csv) if args.gpu_csv else []
    if gpu_rows:
        vals = []
        for r in gpu_rows:
            try:
                vals.append(float(r.get('gpu_util_percent', '')))
            except Exception:
                pass
        gpu_summary = {'gpu_peak_percent': max(vals) if vals else None, 'gpu_mean_percent': (sum(vals)/len(vals) if vals else None)}

    by_batch = {}
    for r in audit_rows:
        pid = str(r.get('profile_id', ''))
        m = re.search(r'Batch-(\d+)', pid)
        b = 'Batch-' + m.group(1) if m else 'unknown'
        by_batch.setdefault(b, {'count': 0, 'pass': 0, 'warn': 0, 'fail': 0})
        by_batch[b]['count'] += 1
        flag = str(r.get('overall_flag', '')).upper()
        if flag == 'PASS':
            by_batch[b]['pass'] += 1
        elif flag == 'WARN':
            by_batch[b]['warn'] += 1
        elif flag == 'FAIL':
            by_batch[b]['fail'] += 1

    failures = []
    if not gen_ok: failures.append('generation_not_clean_pass')
    if not audit_ok: failures.append('radial_audit_not_clean_pass')
    final = 'PASS' if not failures else ('FAIL' if gen.get('overall_status') == 'FAIL' or audit.get('overall_status') == 'FAIL' else 'REVIEW')
    out = {
        'stage': 'D15-P4D full Batch-5/6 remaining14 P2Dlite-RG soft-label generation + radial audit',
        'final_status': final,
        'generation_status': gen.get('overall_status'),
        'radial_audit_status': audit.get('overall_status'),
        'expected_cell_count': expected,
        'generated_count': gen.get('generated_count'),
        'generation_error_count': gen.get('error_count'),
        'generation_missing_status_count': gen.get('missing_status_count'),
        'audit_profile_count': audit.get('profile_count'),
        'audit_pass_count': audit.get('pass_count'),
        'audit_warn_count': audit.get('warn_count'),
        'audit_fail_count': audit.get('fail_count'),
        'audit_read_error_count': audit.get('read_error_count'),
        'total_time_points': gen.get('total_time_points'),
        'total_output_size_mb': gen.get('total_output_size_mb'),
        'resource_summary': resource_summary,
        'gpu_summary': gpu_summary,
        'by_batch_audit_counts': by_batch,
        'failures': failures,
        'allowed_claim_if_pass': 'P2Dlite-RG soft labels and radial-gradient audit are established for the remaining 14 Batch-5/6 XJTU cells.',
        'not_allowed_claims': [
            'Do not claim held-out generalization from D15-P4D alone.',
            'Do not claim experimental internal-state truth for cs_a/cs_c.',
            'Do not claim GPU acceleration for NumPy P2Dlite-RG generation; GPU is not expected in the current backend.'
        ]
    }
    write_json(out, args.out_json)
    print('[D15-P4D scorecard] final_status:', final)
    return 0 if final == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
