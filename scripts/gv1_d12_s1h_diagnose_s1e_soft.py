#!/usr/bin/env python
"""D12-S1H diagnostic-only audit for S1E-soft high-voltage failure.

This script does NOT train a model and does NOT modify the D9.6/D9.5.1 mainline.
It reads existing S1E 40ks prediction.npz files and evaluates whether the
high-voltage failure of the S1E-soft candidate is caused by:

1) global DC bias,
2) a few high-voltage overshoot points,
3) excessive correction leakage in high_target_ge_4p10,
4) or an irreparable correction/base-branch interaction.

Outputs:
- D12_S1H_diagnostic_summary.json
- D12_S1H_variant_decisions.csv
- D12_S1H_variant_segment_metrics.csv
- D12_S1H_profile_high_diagnostics.csv
- D12_S1H_recommendation.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _as1(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_corr(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) < 3:
        return float('nan')
    yy = y - float(np.nanmean(y))
    pp = p - float(np.nanmean(p))
    den = math.sqrt(float(np.nanmean(yy * yy)) * float(np.nanmean(pp * pp)))
    if not np.isfinite(den) or den <= 1e-12:
        return float('nan')
    return float(np.nanmean(yy * pp) / den)


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {'n': 0, 'MAE_V': float('nan'), 'RMSE_V': float('nan'), 'corr': float('nan'), 'bias_V': float('nan')}
    err = p - y
    return {
        'n': int(len(y)),
        'MAE_V': float(np.nanmean(np.abs(err))),
        'RMSE_V': float(math.sqrt(float(np.nanmean(err * err)))),
        'corr': _safe_corr(y, p),
        'bias_V': float(np.nanmean(err)),
    }


def _q(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float('nan')
    return float(np.nanquantile(arr, q))


def _segment_masks(v: np.ndarray, pred: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    finite = np.isfinite(v) & np.isfinite(pred)
    return {
        'all': finite,
        'low_target': finite & (v <= 3.00),
        'low_target_le_2p75': finite & (v <= 2.75),
        'normal_target_gt_3p20': finite & (v > 3.20),
        'target_mid_3p0_4p1': finite & (v > 3.00) & (v < 4.10),
        'high_target_ge_4p10': finite & (v >= 4.10),
        'pred_high_overshoot_gt_4p35': finite & (pred > 4.35),
        'rest_I_zero': finite & (np.abs(current) <= 1e-10),
        'charge_I_positive': finite & (current > 1e-10),
        'discharge_I_negative': finite & (current < -1e-10),
    }


def _parse_run_dir(path: Path) -> Tuple[str, str]:
    name = path.parent.name
    if '__' in name:
        return tuple(name.split('__', 1))  # type: ignore[return-value]
    # fallback for old dirs
    parts = name.split('_Batch-')
    if len(parts) == 2:
        return parts[0], 'Batch-' + parts[1]
    return name, name


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _prediction_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = _load_npz(path)
    y = _as1(a.get('voltage_exp'))
    p = _as1(a.get('voltage_exp_pred', a.get('phis_c_pred')))
    cur = _as1(a.get('I_profile', np.zeros_like(y)))
    n = min(len(y), len(p), len(cur))
    return y[:n].astype(float), p[:n].astype(float), cur[:n].astype(float)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _variants(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> Dict[str, np.ndarray]:
    # Masks are target/pred defined. Use candidate pred for overshoot detection.
    high_t = y >= 4.10
    over = cand > 4.35
    high_or_over = high_t | over
    delta = cand - base
    out: Dict[str, np.ndarray] = {}
    out['s1e_soft_identity'] = cand.copy()

    # Only remove hard overshoot above 4.35, no other changes.
    p = cand.copy()
    p = np.minimum(p, 4.35)
    out['clip_pred_gt_4p35_only'] = p

    # In high/overshoot regions, do not let the candidate move too far from baseline.
    for budget in [0.00, 0.01, 0.02, 0.03]:
        p = cand.copy()
        if budget <= 0:
            p[high_or_over] = base[high_or_over]
            name = 'high_region_revert_to_baseline'
        else:
            p[high_or_over] = base[high_or_over] + np.clip(delta[high_or_over], -budget, budget)
            name = f'high_region_delta_budget_{int(budget*1000)}mV'
        out[name] = p

    # Remove mean high-region bias only inside high/overshoot region.
    p = cand.copy()
    if np.any(high_or_over):
        high_bias = float(np.nanmean((cand - y)[high_or_over]))
        p[high_or_over] = p[high_or_over] - high_bias
    out['high_region_mean_bias_recenter'] = p

    # Global per-profile recenter diagnostic. If this fixes everything, issue is DC bias.
    p = cand.copy()
    global_bias = float(np.nanmean(cand - y)) if len(y) else 0.0
    p = p - global_bias
    out['global_profile_recenter_diagnostic'] = p

    # Hybrid: cap overshoot, then high-region budget 20 mV.
    p = cand.copy()
    p = np.minimum(p, 4.35)
    high_or_over2 = high_t | (p > 4.35)
    p[high_or_over2] = base[high_or_over2] + np.clip((p - base)[high_or_over2], -0.02, 0.02)
    out['clip_4p35_plus_high_delta_budget_20mV'] = p
    return out


def _mean_delta(rows: List[Dict[str, Any]], key: str) -> float:
    vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float('nan')
    return float(np.nanmean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_root_s1e', required=True, help='S1E 6x40ks runs root containing prediction.npz files')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--baseline_mode', default='baseline_d951')
    ap.add_argument('--candidate_mode', default='d12s1e_p2d_low_anchor_soft')
    ap.add_argument('--min_low_improve_V', type=float, default=0.020)
    ap.add_argument('--max_global_regress_V', type=float, default=0.005)
    ap.add_argument('--max_corr_drop', type=float, default=0.005)
    ap.add_argument('--max_rest_regress_V', type=float, default=0.020)
    ap.add_argument('--max_high_regress_V', type=float, default=0.020)
    ap.add_argument('--max_normal_regress_V', type=float, default=0.005)
    args = ap.parse_args()

    root = Path(args.runs_root_s1e)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_map: Dict[Tuple[str, str], Path] = {}
    for pred in sorted(root.rglob('prediction.npz')):
        mode, profile = _parse_run_dir(pred)
        pred_map[(mode, profile)] = pred

    profiles = sorted({p for (m, p) in pred_map if m == args.baseline_mode} & {p for (m, p) in pred_map if m == args.candidate_mode})
    if not profiles:
        raise SystemExit(f'No overlapping profiles found for baseline={args.baseline_mode} and candidate={args.candidate_mode} under {root}')

    segment_rows: List[Dict[str, Any]] = []
    high_diag_rows: List[Dict[str, Any]] = []
    decision_acc: Dict[str, List[Dict[str, float]]] = {}

    for profile in profiles:
        y, base, cur = _prediction_arrays(pred_map[(args.baseline_mode, profile)])
        y2, cand, cur2 = _prediction_arrays(pred_map[(args.candidate_mode, profile)])
        n = min(len(y), len(y2), len(base), len(cand), len(cur), len(cur2))
        y = y[:n]
        base = base[:n]
        cand = cand[:n]
        cur = cur[:n]
        # target arrays should match but we use the baseline y as source of truth.
        variants = _variants(y, base, cand)
        variants['baseline_d951'] = base.copy()

        for vname, pred in variants.items():
            masks = _segment_masks(y, pred, cur)
            for seg, mask in masks.items():
                mm = _metrics(y[mask], pred[mask]) if np.any(mask) else _metrics(np.asarray([]), np.asarray([]))
                segment_rows.append({'variant': vname, 'profile': profile, 'segment': seg, **mm})

        # High failure diagnostics on original S1E-soft.
        high = y >= 4.10
        over = cand > 4.35
        high_or_over = high | over
        err = cand - y
        base_err = base - y
        high_diag_rows.append({
            'profile': profile,
            'n': int(n),
            'n_high_target_ge_4p10': int(np.sum(high)),
            'n_pred_overshoot_gt_4p35': int(np.sum(over)),
            'n_high_or_overshoot': int(np.sum(high_or_over)),
            'baseline_high_MAE_V': _metrics(y[high], base[high])['MAE_V'] if np.any(high) else float('nan'),
            's1e_soft_high_MAE_V': _metrics(y[high], cand[high])['MAE_V'] if np.any(high) else float('nan'),
            's1e_soft_high_bias_V': _metrics(y[high], cand[high])['bias_V'] if np.any(high) else float('nan'),
            's1e_soft_overshoot_MAE_V': _metrics(y[over], cand[over])['MAE_V'] if np.any(over) else float('nan'),
            's1e_soft_err_p05_V': _q(err[high_or_over], 0.05),
            's1e_soft_err_p50_V': _q(err[high_or_over], 0.50),
            's1e_soft_err_p95_V': _q(err[high_or_over], 0.95),
            's1e_soft_global_bias_V': _metrics(y, cand)['bias_V'],
            'baseline_global_bias_V': _metrics(y, base)['bias_V'],
            'delta_candidate_minus_baseline_high_mean_V': float(np.nanmean((cand-base)[high_or_over])) if np.any(high_or_over) else float('nan'),
            'delta_candidate_minus_baseline_high_p95_V': _q((cand-base)[high_or_over], 0.95),
        })

    # Build decisions using averaged segment deltas vs baseline_d951.
    # Index rows for easier lookup.
    by_variant_profile_seg: Dict[Tuple[str, str, str], Dict[str, Any]] = {(r['variant'], r['profile'], r['segment']): r for r in segment_rows}
    variants = sorted({r['variant'] for r in segment_rows if r['variant'] != 'baseline_d951'})
    decision_rows: List[Dict[str, Any]] = []
    for vname in variants:
        prof_dec: List[Dict[str, float]] = []
        for profile in profiles:
            b_all = by_variant_profile_seg[('baseline_d951', profile, 'all')]
            c_all = by_variant_profile_seg[(vname, profile, 'all')]
            b_low = by_variant_profile_seg[('baseline_d951', profile, 'low_target')]
            c_low = by_variant_profile_seg[(vname, profile, 'low_target')]
            b_deep = by_variant_profile_seg[('baseline_d951', profile, 'low_target_le_2p75')]
            c_deep = by_variant_profile_seg[(vname, profile, 'low_target_le_2p75')]
            b_rest = by_variant_profile_seg[('baseline_d951', profile, 'rest_I_zero')]
            c_rest = by_variant_profile_seg[(vname, profile, 'rest_I_zero')]
            b_high = by_variant_profile_seg[('baseline_d951', profile, 'high_target_ge_4p10')]
            c_high = by_variant_profile_seg[(vname, profile, 'high_target_ge_4p10')]
            b_normal = by_variant_profile_seg[('baseline_d951', profile, 'normal_target_gt_3p20')]
            c_normal = by_variant_profile_seg[(vname, profile, 'normal_target_gt_3p20')]
            prof_dec.append({
                'delta_all_MAE_V': float(c_all['MAE_V']) - float(b_all['MAE_V']),
                'delta_low_target_MAE_V': float(c_low['MAE_V']) - float(b_low['MAE_V']),
                'delta_low_le_2p75_MAE_V': float(c_deep['MAE_V']) - float(b_deep['MAE_V']) if int(b_deep['n']) > 0 else float('nan'),
                'delta_rest_MAE_V': float(c_rest['MAE_V']) - float(b_rest['MAE_V']) if int(b_rest['n']) > 0 else 0.0,
                'delta_high_MAE_V': float(c_high['MAE_V']) - float(b_high['MAE_V']) if int(b_high['n']) > 0 else 0.0,
                'delta_normal_MAE_V': float(c_normal['MAE_V']) - float(b_normal['MAE_V']) if int(b_normal['n']) > 0 else 0.0,
                'delta_corr': float(c_all['corr']) - float(b_all['corr']),
            })
        row = {'variant': vname, 'profile_count': len(prof_dec)}
        for key in ['delta_all_MAE_V','delta_low_target_MAE_V','delta_low_le_2p75_MAE_V','delta_rest_MAE_V','delta_high_MAE_V','delta_normal_MAE_V','delta_corr']:
            row[key] = _mean_delta(prof_dec, key)
        row['low_ok'] = bool(row['delta_low_target_MAE_V'] <= -args.min_low_improve_V)
        row['deep_ok'] = bool(np.isfinite(row['delta_low_le_2p75_MAE_V']) and row['delta_low_le_2p75_MAE_V'] <= -args.min_low_improve_V)
        row['global_ok'] = bool(row['delta_all_MAE_V'] <= args.max_global_regress_V)
        row['corr_ok'] = bool(row['delta_corr'] >= -args.max_corr_drop)
        row['rest_ok'] = bool(row['delta_rest_MAE_V'] <= args.max_rest_regress_V)
        row['high_ok'] = bool(row['delta_high_MAE_V'] <= args.max_high_regress_V)
        row['normal_ok'] = bool(row['delta_normal_MAE_V'] <= args.max_normal_regress_V)
        row['diagnostic_promotion'] = bool(row['low_ok'] and row['deep_ok'] and row['global_ok'] and row['corr_ok'] and row['rest_ok'] and row['high_ok'] and row['normal_ok'])
        decision_rows.append(row)

    _write_csv(out_dir / 'D12_S1H_variant_segment_metrics.csv', segment_rows)
    _write_csv(out_dir / 'D12_S1H_profile_high_diagnostics.csv', high_diag_rows)
    _write_csv(out_dir / 'D12_S1H_variant_decisions.csv', decision_rows)

    promoted = [r['variant'] for r in decision_rows if r.get('diagnostic_promotion')]
    # Determine recommendation.
    best_global = min(decision_rows, key=lambda r: (not bool(r.get('low_ok')), float(r.get('delta_all_MAE_V', 1e9)))) if decision_rows else None
    best_high = min(decision_rows, key=lambda r: float(r.get('delta_high_MAE_V', 1e9))) if decision_rows else None
    identity = next((r for r in decision_rows if r['variant'] == 's1e_soft_identity'), None)
    summary = {
        'ok': True,
        'stage': 'D12-S1H diagnostic-only high failure audit',
        'runs_root_s1e': str(root),
        'output_dir': str(out_dir),
        'baseline_mode': args.baseline_mode,
        'candidate_mode': args.candidate_mode,
        'profile_count': len(profiles),
        'variant_count': len(decision_rows),
        'diagnostic_promoted_variants': promoted,
        'identity_s1e_soft_decision': identity,
        'best_global_candidate': best_global,
        'best_high_candidate': best_high,
        'decision_rule': {
            'low_and_deep_MAE_improve_at_least_V': args.min_low_improve_V,
            'global_MAE_regress_no_more_than_V': args.max_global_regress_V,
            'corr_drop_no_more_than': args.max_corr_drop,
            'rest_high_regress_no_more_than_V': args.max_rest_regress_V,
            'normal_regress_no_more_than_V': args.max_normal_regress_V,
        },
    }
    (out_dir / 'D12_S1H_diagnostic_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    rec_lines: List[str] = []
    rec_lines.append('# D12-S1H diagnostic recommendation')
    rec_lines.append('')
    rec_lines.append('This is a diagnostic-only stage. It does not train a new model.')
    rec_lines.append('')
    rec_lines.append(f'- profiles analyzed: {len(profiles)}')
    rec_lines.append(f'- diagnostic promoted variants: {promoted if promoted else "none"}')
    if identity:
        rec_lines.append('')
        rec_lines.append('## S1E-soft identity decision')
        for k in ['delta_all_MAE_V','delta_low_target_MAE_V','delta_low_le_2p75_MAE_V','delta_normal_MAE_V','delta_high_MAE_V','delta_rest_MAE_V','delta_corr','low_ok','deep_ok','global_ok','normal_ok','high_ok','rest_ok','corr_ok']:
            rec_lines.append(f'- {k}: {identity.get(k)}')
    rec_lines.append('')
    if promoted:
        rec_lines.append('## Next action')
        rec_lines.append('A post-scorecard high-only repair appears feasible. Use the promoted variant(s) as the blueprint for D12-S1I, implemented inside training only after confirming no low-target regression.')
    else:
        rec_lines.append('## Next action')
        rec_lines.append('No diagnostic repair satisfies all thresholds. Do not train 200ks. Inspect D12_S1H_profile_high_diagnostics.csv to decide whether the problem is sparse overshoot or broad high-region bias. If broad bias dominates, stop P2D prediction-branch tuning and redesign the voltage head with explicit additive-bias separation.')
    rec_lines.append('')
    rec_lines.append('## Output files')
    rec_lines.append('- D12_S1H_diagnostic_summary.json')
    rec_lines.append('- D12_S1H_variant_decisions.csv')
    rec_lines.append('- D12_S1H_variant_segment_metrics.csv')
    rec_lines.append('- D12_S1H_profile_high_diagnostics.csv')
    (out_dir / 'D12_S1H_RECOMMENDATION.md').write_text('\n'.join(rec_lines), encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
