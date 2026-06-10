from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='D15-P4D collect smoke scorecard.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--config', default='configs/d15_p4d_smoke_config.json')
    ap.add_argument('--out-json', required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_json(args.config, {})
    expected_cells = list(cfg.get('smoke_cells', []))
    summaries = []
    for cell in expected_cells:
        s = load_json(run_dir / 'logs' / f'{cell}_summary.json', None)
        if s is not None:
            summaries.append(s)
    cpu_rows = read_csv(run_dir / 'D15_P4D_SMOKE_RESOURCE_MONITOR.csv')
    gpu_rows = read_csv(run_dir / 'D15_P4D_SMOKE_GPU_MONITOR.csv')
    cuda = load_json(run_dir / 'D15_P4D_CUDA_SMOKE_REPORT.json', {})

    cpu_vals = [parse_float(r.get('cpu_total_percent')) for r in cpu_rows if r.get('cpu_total_percent') not in (None, '')]
    cpu_vals = [v for v in cpu_vals if v == v]
    gpu_vals = [parse_float(r.get('gpu_util_percent')) for r in gpu_rows if r.get('gpu_util_percent') not in (None, '')]
    gpu_vals = [v for v in gpu_vals if v == v]

    generation_status = 'PASS' if len(summaries) == len(expected_cells) and len(summaries) > 0 else 'REVIEW'
    cpu_peak = max(cpu_vals) if cpu_vals else None
    cpu_mean = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
    gpu_peak = max(gpu_vals) if gpu_vals else None
    gpu_mean = sum(gpu_vals) / len(gpu_vals) if gpu_vals else None
    monitor_status = 'PASS' if cpu_vals else 'REVIEW'
    cuda_status = cuda.get('status', 'MISSING')

    thresholds = cfg.get('resource_monitor', {})
    cpu_peak_expect = float(thresholds.get('expected_min_cpu_peak_percent', 35.0))
    cpu_mean_expect = float(thresholds.get('expected_min_cpu_mean_percent', 18.0))
    resource_findings = []
    if cpu_peak is not None and cpu_peak < cpu_peak_expect:
        resource_findings.append(f'cpu_peak_below_expectation:{cpu_peak:.1f}<{cpu_peak_expect:.1f}')
    if cpu_mean is not None and cpu_mean < cpu_mean_expect:
        resource_findings.append(f'cpu_mean_below_expectation:{cpu_mean:.1f}<{cpu_mean_expect:.1f}')
    if cuda_status != 'PASS':
        resource_findings.append(f'cuda_smoke_not_pass:{cuda_status}')

    final = 'PASS' if generation_status == 'PASS' and monitor_status == 'PASS' and cuda_status == 'PASS' else 'REVIEW'
    score = {
        'stage': 'D15-P4D-smoke Batch-5/6 remaining14 soft-label generation resource smoke',
        'final_status': final,
        'generation_status': generation_status,
        'monitor_status': monitor_status,
        'cuda_smoke_status': cuda_status,
        'expected_smoke_cell_count': len(expected_cells),
        'completed_smoke_cell_count': len(summaries),
        'completed_cells': [s.get('cell_id') for s in summaries],
        'cpu_peak_percent': cpu_peak,
        'cpu_mean_percent': cpu_mean,
        'gpu_monitor_peak_percent': gpu_peak,
        'gpu_monitor_mean_percent': gpu_mean,
        'cuda_device_name': cuda.get('device_name'),
        'cuda_estimated_tflops': cuda.get('estimated_tflops'),
        'per_cell_wall_seconds': {s.get('cell_id'): s.get('total_wall_seconds') for s in summaries},
        'per_cell_compute_wall_seconds': {s.get('cell_id'): s.get('compute_wall_seconds') for s in summaries},
        'per_cell_output_size_mb': {s.get('cell_id'): s.get('output_size_mb') for s in summaries},
        'resource_findings': resource_findings,
        'interpretation': [
            'This smoke intentionally generates partial soft labels only; it does not complete the remaining14 final labels.',
            'CUDA smoke can PASS even though current P2Dlite-RG generator remains NumPy/CPU.',
            'If CPU remains low during label generation and CUDA smoke passes, P4D requires generator backend redesign or chunked/resume CPU generation rather than more worker tuning.'
        ],
        'recommended_next_action': 'If generation and audits are reasonable, design P4D full remaining14 with chunked/resume generation. Do not launch full 14-cell generation without using this smoke evidence.',
        'not_allowed_claims': [
            'Do not claim remaining14 full soft labels are generated by this smoke.',
            'Do not claim GPU is used by NumPy label generation.',
            'Do not claim experimental internal-state truth.'
        ]
    }
    write_json(score, args.out_json)
    print('[D15-P4D smoke scorecard] final_status:', final)
    print('[D15-P4D smoke scorecard] CPU peak/mean:', cpu_peak, cpu_mean)
    print('[D15-P4D smoke scorecard] CUDA:', cuda_status)
    return 0 if final == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
