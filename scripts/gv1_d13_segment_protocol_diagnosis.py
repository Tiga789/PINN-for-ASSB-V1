#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D13 segment/protocol diagnosis for GV1 D9.6/D9.5.1 mainline and D12-S3 metadata ablation.

This script is deliberately analysis-only.  It never launches training.
It reads existing prediction.npz files and/or D12-S3 scorecard CSV files,
then writes protocol-, segment-, mode- and run-level summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == '':
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def corrcoef(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return math.nan
    yy = y[mask].astype(float)
    pp = p[mask].astype(float)
    if float(np.nanstd(yy)) <= 1e-12 or float(np.nanstd(pp)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(yy, pp)[0, 1])


def metrics_for(label: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask & np.isfinite(y) & np.isfinite(p)
    n = int(mask.sum())
    if n <= 0:
        return {'segment': label, 'n': 0}
    err = p[mask] - y[mask]
    ae = np.abs(err)
    return {
        'segment': label,
        'n': n,
        'mae_V': float(np.nanmean(ae)),
        'rmse_V': float(np.sqrt(np.nanmean(err ** 2))),
        'bias_V': float(np.nanmean(err)),
        'corr': corrcoef(y[mask], p[mask]),
        'p50_abs_err_V': float(np.nanquantile(ae, 0.50)),
        'p90_abs_err_V': float(np.nanquantile(ae, 0.90)),
        'p95_abs_err_V': float(np.nanquantile(ae, 0.95)),
        'target_min_V': float(np.nanmin(y[mask])),
        'target_max_V': float(np.nanmax(y[mask])),
        'pred_min_V': float(np.nanmin(p[mask])),
        'pred_max_V': float(np.nanmax(p[mask])),
        'pred_upper_frac_ge_4p269': float(np.nanmean(p[mask] >= 4.269)),
        'pred_overshoot_frac_gt_4p35': float(np.nanmean(p[mask] > 4.35)),
        'pred_low_frac_le_2p75': float(np.nanmean(p[mask] <= 2.75)),
        'target_low_frac_le_2p75': float(np.nanmean(y[mask] <= 2.75)),
    }


def find_key(keys: set[str], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in keys:
            return key
    return None


def protocol_from_profile(profile: str) -> str:
    text = profile.replace('R25', 'R2.5')
    if 'R2.5' in text:
        return 'R2.5'
    if 'R3' in text:
        return 'R3'
    if '2C' in text:
        return '2C'
    return 'unknown'


def mode_from_name(name: str) -> str:
    m = re.search(r'metadata_(off|zero|on)', name)
    if m:
        return m.group(1)
    return 'mainline'


def profile_from_path(path: Path) -> str:
    name = path.parent.name
    # D12-S3 names: xjtu_batch134_d12_s3_metadata_off_Batch-1_2C_battery-1_STRICT_40ks
    m = re.search(r'metadata_(?:off|zero|on)_(.*?)_STRICT_40ks', name)
    if m:
        return m.group(1)
    # D10-P1 or D9.6 names often contain Batch-...battery-N somewhere.
    m = re.search(r'(Batch-[134]_[A-Za-z0-9\.]+_battery-\d+)', name)
    if m:
        return m.group(1)
    m = re.search(r'(B[134]_[A-Za-z0-9\.]+_battery-\d+)', name)
    if m:
        return m.group(1)
    return name


def find_prediction_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.rglob('prediction.npz'))


def compute_prediction_segments(prediction_npz: Path, dataset: str, mode: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(prediction_npz, allow_pickle=True) as z:
        keys = set(z.files)
        y_key = find_key(keys, ['voltage_exp', 'target_voltage', 'voltage_true', 'y_voltage'])
        p_key = find_key(keys, ['voltage_exp_pred', 'phis_c_pred', 'voltage_pred', 'pred_voltage'])
        if y_key is None:
            raise KeyError(f'No target voltage key in {prediction_npz}; keys={sorted(keys)}')
        if p_key is None:
            raise KeyError(f'No prediction voltage key in {prediction_npz}; keys={sorted(keys)}')
        y = np.asarray(z[y_key], dtype=float).reshape(-1)
        p = np.asarray(z[p_key], dtype=float).reshape(-1)
        if 'I_profile' in keys:
            current = np.asarray(z['I_profile'], dtype=float).reshape(-1)
        elif 'current_A' in keys:
            current = np.asarray(z['current_A'], dtype=float).reshape(-1)
        else:
            current = np.zeros_like(y)
        if 't_global_s' in keys:
            time = np.asarray(z['t_global_s'], dtype=float).reshape(-1)
        elif 't_s' in keys:
            time = np.asarray(z['t_s'], dtype=float).reshape(-1)
        else:
            time = np.arange(len(y), dtype=float)

    n = min(len(y), len(p), len(current), len(time))
    y, p, current, time = y[:n], p[:n], current[:n], time[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    eps = 1e-8
    abs_i = np.abs(current[np.isfinite(current)])
    high_i = float(np.nanquantile(abs_i, 0.90)) if len(abs_i) else 0.0
    # Time tertiles are useful for detecting early/mid/late drift.
    if np.isfinite(time).sum() >= 3:
        t0, t1 = float(np.nanmin(time)), float(np.nanmax(time))
        span = max(t1 - t0, 1e-12)
        tau = (time - t0) / span
    else:
        tau = np.linspace(0.0, 1.0, n)
    masks = {
        'all': np.ones(n, dtype=bool),
        'charge_I_pos': current > eps,
        'discharge_I_neg': current < -eps,
        'rest_I_zero': np.abs(current) <= eps,
        'low_target_le_2p75': y <= 2.75,
        'mid_target_2p75_4p10': (y > 2.75) & (y < 4.10),
        'high_target_ge_4p10': y >= 4.10,
        'high_current_abs_q90': np.abs(current) >= max(high_i, eps),
        'early_time_0_33': tau <= 1.0/3.0,
        'mid_time_33_66': (tau > 1.0/3.0) & (tau <= 2.0/3.0),
        'late_time_66_100': tau > 2.0/3.0,
        'pred_high_overshoot_gt_4p35': p > 4.35,
        'pred_upper_ge_4p269': p >= 4.269,
    }
    profile = profile_from_path(prediction_npz)
    protocol = protocol_from_profile(profile)
    mode = mode if mode is not None else mode_from_name(prediction_npz.parent.name)
    run_base = {
        'dataset': dataset,
        'run_name': prediction_npz.parent.name,
        'run_dir': str(prediction_npz.parent),
        'prediction_npz': str(prediction_npz),
        'mode': mode,
        'profile': profile,
        'protocol': protocol,
        'n_points': int(n),
        'target_key': y_key,
        'prediction_key': p_key,
        'current_high_quantile_threshold_A': high_i,
    }
    seg_rows: list[dict[str, Any]] = []
    for label, mask in masks.items():
        m = metrics_for(label, y, p, mask & finite)
        row = dict(run_base)
        row.update(m)
        seg_rows.append(row)
    all_metrics = next(row for row in seg_rows if row['segment'] == 'all')
    run_row = dict(run_base)
    for key, value in all_metrics.items():
        if key not in run_row:
            run_row[key] = value
    run_row['status'] = 'metrics_ok'
    return run_row, seg_rows


def maybe_load_d12_csv(scorecard_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    scorecard = scorecard_dir / 'd12_s3_scorecard.csv'
    segments = scorecard_dir / 'd12_s3_segment_metrics.csv'
    if scorecard.exists():
        for r in read_csv(scorecard):
            row = {k: v for k, v in r.items()}
            row['dataset'] = 'D12S3_23profile_40ks_metadata_ablation'
            row['profile'] = row.get('metadata_profile_id') or row.get('profile') or ''
            row['mae_V'] = safe_float(row.get('mae_V'))
            row['rmse_V'] = safe_float(row.get('rmse_V'))
            row['corr'] = safe_float(row.get('corr'))
            row['bias_V'] = safe_float(row.get('bias_V'))
            row['n'] = safe_int(row.get('n'))
            row['status'] = row.get('status', 'metrics_ok')
            run_rows.append(row)
    if segments.exists():
        for r in read_csv(segments):
            row = {k: v for k, v in r.items()}
            row['dataset'] = 'D12S3_23profile_40ks_metadata_ablation'
            row['profile'] = row.get('metadata_profile_id') or row.get('profile') or ''
            # D12 uses label, D13 uses segment.
            row['segment'] = row.get('segment') or row.get('label') or ''
            row['mae_V'] = safe_float(row.get('mae_V'))
            row['rmse_V'] = safe_float(row.get('rmse_V'))
            row['corr'] = safe_float(row.get('corr'))
            row['bias_V'] = safe_float(row.get('bias_V'))
            row['n'] = safe_int(row.get('n'))
            seg_rows.append(row)
    return run_rows, seg_rows


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.nanmean(vals)) if vals else math.nan


def group_summary(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = tuple(str(r.get(k, 'unknown')) for k in group_keys)
        buckets.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        ok = [r for r in group if str(r.get('status', 'metrics_ok')) in {'metrics_ok', 'strict_completed_metrics_ok', 'strict_completed_metrics_review', ''}]
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row.update({
            'n_rows': len(group),
            'n_ok': len(ok),
            'mean_mae_V': mean(r.get('mae_V') for r in ok),
            'mean_rmse_V': mean(r.get('rmse_V') for r in ok),
            'mean_corr': mean(r.get('corr') for r in ok),
            'mean_bias_V': mean(r.get('bias_V') for r in ok),
            'max_mae_V': max([safe_float(r.get('mae_V')) for r in ok if np.isfinite(safe_float(r.get('mae_V')))] or [math.nan]),
        })
        out.append(row)
    return out


def top_rows(rows: list[dict[str, Any]], key: str = 'mae_V', n: int = 20, predicate=None) -> list[dict[str, Any]]:
    selected = [r for r in rows if predicate is None or predicate(r)]
    selected.sort(key=lambda r: safe_float(r.get(key)), reverse=True)
    return selected[:n]


def comparison_from_mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    d12 = [r for r in rows if r.get('dataset') == 'D12S3_23profile_40ks_metadata_ablation']
    by_mode = {str(r.get('mode')): [] for r in d12}
    for r in d12:
        by_mode.setdefault(str(r.get('mode')), []).append(r)
    summary: dict[str, Any] = {}
    for mode, group in sorted(by_mode.items()):
        summary[mode] = {
            'n': len(group),
            'mean_mae_V': mean(r.get('mae_V') for r in group),
            'mean_corr': mean(r.get('corr') for r in group),
            'mean_bias_V': mean(r.get('bias_V') for r in group),
        }
    if 'on' in summary and 'off' in summary:
        summary['on_minus_off_mae_V'] = summary['on']['mean_mae_V'] - summary['off']['mean_mae_V']
        summary['on_minus_off_corr'] = summary['on']['mean_corr'] - summary['off']['mean_corr']
    if 'zero' in summary and 'off' in summary:
        summary['zero_minus_off_mae_V'] = summary['zero']['mean_mae_V'] - summary['off']['mean_mae_V']
        summary['zero_minus_off_corr'] = summary['zero']['mean_corr'] - summary['off']['mean_corr']
    return summary


def build_recommendation(summary: dict[str, Any], protocol_summary: list[dict[str, Any]], segment_summary: list[dict[str, Any]], worst_runs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append('# D13 segment/protocol diagnosis recommendation')
    lines.append('')
    lines.append('## Status')
    lines.append('')
    lines.append(f"- D10/D12 data loaded: `{summary.get('datasets_loaded')}`")
    lines.append(f"- Total run rows: `{summary.get('run_count')}`")
    lines.append(f"- Total segment rows: `{summary.get('segment_row_count')}`")
    lines.append('')
    d12_cmp = summary.get('d12_mode_comparison', {})
    if d12_cmp:
        lines.append('## D12-S3 metadata conclusion')
        lines.append('')
        lines.append('```json')
        lines.append(json.dumps(d12_cmp, ensure_ascii=False, indent=2, allow_nan=True))
        lines.append('```')
        delta = safe_float(d12_cmp.get('on_minus_off_mae_V'))
        if np.isfinite(delta) and delta > 0.0:
            lines.append('')
            lines.append('**Recommendation:** `metadata_on` increases MAE relative to `metadata_off`; keep D9.6/D9.5.1 as the GV1 mainline and do not promote metadata_on.')
        elif np.isfinite(delta):
            lines.append('')
            lines.append('**Recommendation:** metadata_on does not show a harmful MAE delta here, but should still be treated as ablation unless confirmed on longer mainline windows.')
    lines.append('')
    lines.append('## Worst protocol/segment groups')
    lines.append('')
    lines.append('| dataset | mode | protocol | segment | n_rows | mean_MAE_V | mean_corr | mean_bias_V |')
    lines.append('|---|---|---|---|---:|---:|---:|---:|')
    for r in top_rows(segment_summary, 'mean_mae_V', 15):
        lines.append(f"| {r.get('dataset','')} | {r.get('mode','')} | {r.get('protocol','')} | {r.get('segment','')} | {r.get('n_rows','')} | {r.get('mean_mae_V','')} | {r.get('mean_corr','')} | {r.get('mean_bias_V','')} |")
    lines.append('')
    lines.append('## Worst individual runs')
    lines.append('')
    lines.append('| dataset | mode | protocol | profile | MAE_V | corr | bias_V |')
    lines.append('|---|---|---|---|---:|---:|---:|')
    for r in worst_runs[:15]:
        profile = r.get('profile') or r.get('metadata_profile_id') or ''
        lines.append(f"| {r.get('dataset','')} | {r.get('mode','')} | {r.get('protocol','')} | {profile} | {r.get('mae_V','')} | {r.get('corr','')} | {r.get('bias_V','')} |")
    lines.append('')
    lines.append('## Next action')
    lines.append('')
    lines.append('1. Keep D9.6/D9.5.1 trend-first warmup rare-regime as the mainline.')
    lines.append('2. Keep B1_2C battery-8 flagged/excluded unless a separate target-probe model is designed.')
    lines.append('3. Stop metadata_on as a mainline candidate after D12-S3 unless a future mechanism-level reason is introduced.')
    lines.append('4. Use the worst protocol/segment table to decide D14: protocol-specific adapter, low-tail correction, or P2D-like/high-rate correction.')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='D13 segment/protocol diagnosis from existing GV1 predictions and D12 scorecards.')
    ap.add_argument('--cache_root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--d10_p1_dir', default=None, help='D10-P1 23-profile 200ks mainline directory. Default uses cache_root/xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8.')
    ap.add_argument('--d12_s3_scorecard_dir', default=None, help='D12-S3 scorecard directory. Default uses cache_root/xjtu_batch134_d12_s3_metadata_ablation_scorecard.')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--skip_d10', action='store_true')
    ap.add_argument('--skip_d12', action='store_true')
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    d10_dir = Path(args.d10_p1_dir) if args.d10_p1_dir else cache_root / 'xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8'
    d12_dir = Path(args.d12_s3_scorecard_dir) if args.d12_s3_scorecard_dir else cache_root / 'xjtu_batch134_d12_s3_metadata_ablation_scorecard'
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / 'xjtu_batch134_d13_segment_protocol_diagnosis'
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    datasets_loaded: list[str] = []
    notes: list[str] = []

    if not args.skip_d10:
        predictions = find_prediction_files(d10_dir)
        if predictions:
            datasets_loaded.append('D10P1_mainline_23profile_200ks')
            for pred in predictions:
                try:
                    rr, ss = compute_prediction_segments(pred, dataset='D10P1_mainline_23profile_200ks', mode='off')
                    run_rows.append(rr)
                    segment_rows.extend(ss)
                except Exception as exc:
                    run_rows.append({
                        'dataset': 'D10P1_mainline_23profile_200ks',
                        'run_name': pred.parent.name,
                        'run_dir': str(pred.parent),
                        'prediction_npz': str(pred),
                        'mode': 'off',
                        'profile': profile_from_path(pred),
                        'protocol': protocol_from_profile(profile_from_path(pred)),
                        'status': 'metrics_compute_error',
                        'error': str(exc),
                    })
        else:
            notes.append(f'D10-P1 predictions not found under {d10_dir}; continuing with D12-S3 if available.')

    if not args.skip_d12:
        d12_runs, d12_segments = maybe_load_d12_csv(d12_dir)
        if d12_runs or d12_segments:
            datasets_loaded.append('D12S3_23profile_40ks_metadata_ablation')
            run_rows.extend(d12_runs)
            segment_rows.extend(d12_segments)
        else:
            # Fallback: compute D12 predictions from run dirs if scorecard CSV does not exist.
            d12_run_parent = cache_root
            d12_preds = [p for p in find_prediction_files(d12_run_parent) if 'd12_s3_metadata_' in str(p) and 'STRICT_40ks' in str(p)]
            if d12_preds:
                datasets_loaded.append('D12S3_23profile_40ks_metadata_ablation')
                for pred in d12_preds:
                    try:
                        rr, ss = compute_prediction_segments(pred, dataset='D12S3_23profile_40ks_metadata_ablation')
                        run_rows.append(rr)
                        segment_rows.extend(ss)
                    except Exception as exc:
                        run_rows.append({'dataset':'D12S3_23profile_40ks_metadata_ablation','run_name':pred.parent.name,'run_dir':str(pred.parent),'status':'metrics_compute_error','error':str(exc)})
            else:
                notes.append(f'D12-S3 scorecard/predictions not found under {d12_dir} or {cache_root}.')

    # Write raw rows.
    write_csv(out_dir / 'D13_run_metrics.csv', run_rows)
    write_csv(out_dir / 'D13_segment_metrics.csv', segment_rows)

    protocol_summary = group_summary(run_rows, ['dataset', 'mode', 'protocol'])
    segment_summary = group_summary(segment_rows, ['dataset', 'mode', 'protocol', 'segment'])
    mode_summary = group_summary(run_rows, ['dataset', 'mode'])
    profile_summary = group_summary(run_rows, ['dataset', 'mode', 'protocol', 'profile'])
    charge_discharge_summary = [r for r in segment_summary if r.get('segment') in {'charge_I_pos', 'discharge_I_neg', 'rest_I_zero'}]
    voltage_tail_summary = [r for r in segment_summary if r.get('segment') in {'low_target_le_2p75', 'mid_target_2p75_4p10', 'high_target_ge_4p10', 'pred_high_overshoot_gt_4p35', 'pred_upper_ge_4p269'}]
    time_summary = [r for r in segment_summary if r.get('segment') in {'early_time_0_33', 'mid_time_33_66', 'late_time_66_100'}]
    worst_runs = top_rows(run_rows, 'mae_V', 30)
    worst_segments = top_rows(segment_rows, 'mae_V', 50)

    write_csv(out_dir / 'D13_mode_summary.csv', mode_summary)
    write_csv(out_dir / 'D13_protocol_summary.csv', protocol_summary)
    write_csv(out_dir / 'D13_mode_protocol_summary.csv', group_summary(run_rows, ['dataset', 'mode', 'protocol']))
    write_csv(out_dir / 'D13_mode_protocol_segment_summary.csv', segment_summary)
    write_csv(out_dir / 'D13_charge_discharge_summary.csv', charge_discharge_summary)
    write_csv(out_dir / 'D13_voltage_tail_summary.csv', voltage_tail_summary)
    write_csv(out_dir / 'D13_time_drift_summary.csv', time_summary)
    write_csv(out_dir / 'D13_worst_runs_by_mae.csv', worst_runs)
    write_csv(out_dir / 'D13_worst_segments_by_mae.csv', worst_segments)
    write_csv(out_dir / 'D13_profile_summary.csv', profile_summary)

    summary = {
        'ok': True,
        'stage': 'D13 segment/protocol diagnosis',
        'datasets_loaded': datasets_loaded,
        'cache_root': str(cache_root),
        'd10_p1_dir': str(d10_dir),
        'd12_s3_scorecard_dir': str(d12_dir),
        'out_dir': str(out_dir),
        'run_count': len(run_rows),
        'segment_row_count': len(segment_rows),
        'notes': notes,
        'd12_mode_comparison': comparison_from_mode_summary(run_rows),
        'verdict': 'd13_diagnosis_completed' if run_rows else 'd13_no_data_found',
    }
    write_json(out_dir / 'D13_segment_protocol_summary.json', summary)
    recommendation = build_recommendation(summary, protocol_summary, segment_summary, worst_runs)
    (out_dir / 'D13_RECOMMENDATION.md').write_text(recommendation, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
