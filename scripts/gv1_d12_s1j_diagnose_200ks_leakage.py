#!/usr/bin/env python
"""D12-S1J 200ks normal/rest leakage diagnostic and local wrapper.

This stage is diagnostic-only in the sense that it does not train any neural
network.  It reads the already generated D12-S1E 6-profile 200ks predictions
(baseline_d951 and d12s1e_p2d_low_anchor_soft), then builds several corrected
prediction variants to test where the long-window leakage comes from.

Main question:
    S1I fixed the 40ks high-region leakage, but 200ks still fails because
    normal/rest/global regressed.  Is the S1E correction only useful in the
    genuinely low-voltage region, and should all non-low/rest/high regions be
    reverted or budget-limited to baseline?

Outputs:
    D12_S1J_scorecard_summary.json
    D12_S1J_candidate_decisions.csv
    D12_S1J_mode_summary.csv
    D12_S1J_segment_metrics.csv
    D12_S1J_run_metrics.csv
    D12_S1J_leakage_decomposition.csv
    D12_S1J_RECOMMENDATION.md

The script also writes per-variant prediction.npz files under output_runs_root
so that any promoted diagnostic wrapper can be reused by later stages.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _as1(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float('nan')
    return v if np.isfinite(v) else float('nan')


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
    y = _as1(y).astype(float)
    p = _as1(p).astype(float)
    n = min(len(y), len(p))
    y = y[:n]
    p = p[:n]
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
    arr = _as1(arr).astype(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float('nan')
    return float(np.nanquantile(arr, q))


def _parse_run(path: Path) -> Tuple[str, str]:
    name = path.parent.name
    if '__' in name:
        mode, profile = name.split('__', 1)
        return mode, profile
    parts = name.split('_Batch-')
    if len(parts) == 2:
        return parts[0], 'Batch-' + parts[1]
    return name, name


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _target_pred_current(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 'voltage_exp' not in arrays:
        raise KeyError('prediction.npz missing voltage_exp')
    y = _as1(arrays['voltage_exp']).astype(float)
    if 'voltage_exp_pred' in arrays:
        p = _as1(arrays['voltage_exp_pred']).astype(float)
    elif 'phis_c_pred' in arrays:
        p = _as1(arrays['phis_c_pred']).astype(float)
    else:
        raise KeyError('prediction.npz missing voltage_exp_pred or phis_c_pred')
    cur = _as1(arrays.get('I_profile', np.zeros_like(y))).astype(float)
    n = min(len(y), len(p), len(cur))
    return y[:n], p[:n], cur[:n]


def _segment_masks(v: np.ndarray, pred: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    v = _as1(v).astype(float)
    pred = _as1(pred).astype(float)
    current = _as1(current).astype(float)
    n = min(len(v), len(pred), len(current))
    v = v[:n]
    pred = pred[:n]
    current = current[:n]
    finite = np.isfinite(v) & np.isfinite(pred)
    return {
        'all': finite,
        'low_target': finite & (v <= 3.00),
        'low_target_le_2p75': finite & (v <= 2.75),
        'transition_3p00_3p20': finite & (v > 3.00) & (v <= 3.20),
        'normal_target_gt_3p20': finite & (v > 3.20),
        'mid_normal_3p20_4p10': finite & (v > 3.20) & (v < 4.10),
        'high_target_ge_4p10': finite & (v >= 4.10),
        'pred_high_overshoot_gt_4p35': finite & (pred > 4.35),
        'rest_I_zero': finite & (np.abs(current) <= 1e-10),
        'charge_I_positive': finite & (current > 1e-10),
        'discharge_I_negative': finite & (current < -1e-10),
    }


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


def _save_npz_like(source_arrays: Dict[str, np.ndarray], out_path: Path, corrected_pred: np.ndarray,
                   baseline_pred: np.ndarray, candidate_pred: np.ndarray, apply_mask: np.ndarray,
                   variant_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = dict(source_arrays)
    n = len(corrected_pred)
    if 'voltage_exp_pred' in out:
        old = _as1(out['voltage_exp_pred'])
        if len(old) >= n:
            out['voltage_exp_pred_before_s1j'] = old[:n]
        out['voltage_exp_pred'] = corrected_pred
    else:
        out['voltage_exp_pred'] = corrected_pred
    if 'phis_c_pred' in out:
        old = _as1(out['phis_c_pred'])
        if len(old) >= n:
            out['phis_c_pred_before_s1j'] = old[:n]
            out['phis_c_pred'] = corrected_pred
    out['voltage_exp_pred_baseline_d951'] = baseline_pred
    out['voltage_exp_pred_s1e_soft_source'] = candidate_pred
    out['s1j_apply_s1e_mask'] = apply_mask.astype(np.uint8)
    out['s1j_variant_name'] = np.asarray(variant_name)
    out['s1j_note'] = np.asarray('D12-S1J diagnostic wrapper: diagnose 200ks normal/rest/global leakage; no retraining.')
    np.savez_compressed(out_path, **out)


def _copy_baseline_npz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _build_variants(y: np.ndarray, base: np.ndarray, cand: np.ndarray, current: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {variant_name: (prediction, mask_where_s1e_correction_is_allowed)}.

    These variants are intentionally conservative.  S1I proved that high-region
    can be reverted.  200ks failure shows normal/rest/global leakage, so S1J tests
    whether S1E correction should be applied only in genuine low-voltage regions.
    """
    y = _as1(y).astype(float)
    base = _as1(base).astype(float)
    cand = _as1(cand).astype(float)
    current = _as1(current).astype(float)
    n = min(len(y), len(base), len(cand), len(current))
    y = y[:n]
    base = base[:n]
    cand = cand[:n]
    current = current[:n]
    delta = cand - base
    rest = np.abs(current) <= 1e-10
    low3 = y <= 3.00
    low32 = y <= 3.20
    transition = (y > 3.00) & (y <= 3.20)
    nonlow = y > 3.00
    normal = y > 3.20
    high = y >= 4.10
    over = cand > 4.35
    high_or_over = high | over

    variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Source identity for reference.
    variants['s1e_soft_identity'] = (cand.copy(), np.ones_like(low3, dtype=bool))

    # S1I-like reference: only high region reverted, expected to still fail normal/rest at 200ks.
    p = cand.copy()
    p[high_or_over] = base[high_or_over]
    variants['d12s1j_high_revert_only_reference'] = (p, ~high_or_over)

    # Most conservative: use S1E only on truly low target <=3.0 V, baseline elsewhere.
    p = base.copy()
    p[low3] = cand[low3]
    variants['d12s1j_low_only_revert_nonlow_to_baseline'] = (p, low3)

    # Low plus transition blend: full S1E <=3.0, smooth fade to baseline by 3.2 V.
    p = base.copy()
    w = np.zeros_like(y, dtype=float)
    w[low3] = 1.0
    # transition y 3.0->3.2 uses 1->0.
    if np.any(transition):
        w[transition] = 1.0 - _smoothstep01((y[transition] - 3.00) / 0.20)
    p = base + w * delta
    variants['d12s1j_low_plus_transition_fade_to_baseline'] = (p, w > 1e-6)

    # Low full correction; non-low correction budgeted tightly to protect normal/global.
    for budget in [0.005, 0.010, 0.020]:
        p = cand.copy()
        p[nonlow] = base[nonlow] + np.clip(delta[nonlow], -budget, budget)
        p[rest] = base[rest]  # rest is especially fragile in 200ks
        name = f'd12s1j_low_full_nonlow_budget_{int(budget*1000)}mV_rest_revert'
        variants[name] = (p, low3 | ((~low3) & (np.abs(delta) <= budget) & (~rest)))

    # Low full correction; normal/rest/high all reverted.  This isolates whether low alone is sufficient.
    p = cand.copy()
    protect = normal | rest | high_or_over
    p[protect] = base[protect]
    variants['d12s1j_low_preserve_normal_rest_high_revert'] = (p, ~protect)

    # Rest-only revert on top of high revert: tests whether rest is the dominant 200ks leakage.
    p = cand.copy()
    p[high_or_over | rest] = base[high_or_over | rest]
    variants['d12s1j_high_and_rest_revert'] = (p, ~(high_or_over | rest))

    return variants


def _mean(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float('nan')
    return float(np.nanmean(a))


def _scorecard(output_runs_root: Path, output_dir: Path, baseline_mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    pred_files = sorted(output_runs_root.rglob('prediction.npz'))
    run_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []

    for pred_path in pred_files:
        mode, profile = _parse_run(pred_path)
        try:
            arrays = _load_npz(pred_path)
            y, p, cur = _target_pred_current(arrays)
            m_all = _metrics(y, p)
            run_rows.append({'mode': mode, 'profile': profile, 'prediction_npz': str(pred_path), **m_all, 'status': 'metrics_ok'})
            for seg, mask in _segment_masks(y, p, cur).items():
                mm = _metrics(y[mask], p[mask]) if np.any(mask) else _metrics(np.asarray([]), np.asarray([]))
                seg_rows.append({'mode': mode, 'profile': profile, 'segment': seg, **mm})
        except Exception as exc:  # noqa: BLE001
            run_rows.append({'mode': mode, 'profile': profile, 'prediction_npz': str(pred_path), 'status': 'read_error', 'error': repr(exc)})

    modes = sorted({r['mode'] for r in run_rows})
    mode_rows: List[Dict[str, Any]] = []
    for mode in modes:
        rows = [r for r in run_rows if r.get('mode') == mode and r.get('status') == 'metrics_ok']
        mode_rows.append({
            'mode': mode,
            'n': len(rows),
            'mean_MAE_V': _mean([_safe_float(r.get('MAE_V')) for r in rows]),
            'mean_RMSE_V': _mean([_safe_float(r.get('RMSE_V')) for r in rows]),
            'mean_corr': _mean([_safe_float(r.get('corr')) for r in rows]),
            'mean_bias_V': _mean([_safe_float(r.get('bias_V')) for r in rows]),
            'status': 'metrics_ok' if rows else 'empty',
        })

    by = {(r['mode'], r['profile'], r['segment']): r for r in seg_rows}
    profiles = sorted({r['profile'] for r in run_rows if r.get('mode') == baseline_mode and r.get('status') == 'metrics_ok'})
    decisions: List[Dict[str, Any]] = []
    for mode in modes:
        if mode == baseline_mode:
            continue
        profs = sorted({r['profile'] for r in run_rows if r.get('mode') == mode and r.get('status') == 'metrics_ok'} & set(profiles))
        if not profs:
            continue
        deltas: Dict[str, List[float]] = {k: [] for k in [
            'all', 'low_target', 'low_target_le_2p75', 'normal_target_gt_3p20', 'mid_normal_3p20_4p10',
            'high_target_ge_4p10', 'rest_I_zero', 'charge_I_positive', 'discharge_I_negative'
        ]}
        corr_deltas: List[float] = []
        for profile in profs:
            for seg in deltas:
                b = by.get((baseline_mode, profile, seg), {})
                c = by.get((mode, profile, seg), {})
                deltas[seg].append(_safe_float(c.get('MAE_V')) - _safe_float(b.get('MAE_V')))
            b_all = by.get((baseline_mode, profile, 'all'), {})
            c_all = by.get((mode, profile, 'all'), {})
            corr_deltas.append(_safe_float(c_all.get('corr')) - _safe_float(b_all.get('corr')))

        row: Dict[str, Any] = {
            'mode': mode,
            'profile_count': len(profs),
            'delta_all_MAE_V': _mean(deltas['all']),
            'delta_low_target_MAE_V': _mean(deltas['low_target']),
            'delta_low_le_2p75_MAE_V': _mean(deltas['low_target_le_2p75']),
            'delta_normal_MAE_V': _mean(deltas['normal_target_gt_3p20']),
            'delta_mid_normal_3p20_4p10_MAE_V': _mean(deltas['mid_normal_3p20_4p10']),
            'delta_high_MAE_V': _mean(deltas['high_target_ge_4p10']),
            'delta_rest_MAE_V': _mean(deltas['rest_I_zero']),
            'delta_charge_MAE_V': _mean(deltas['charge_I_positive']),
            'delta_discharge_MAE_V': _mean(deltas['discharge_I_negative']),
            'delta_corr': _mean(corr_deltas),
        }
        low_ok = row['delta_low_target_MAE_V'] <= -args.min_low_improve_V
        deep_ok = row['delta_low_le_2p75_MAE_V'] <= -args.min_low_improve_V
        global_ok = row['delta_all_MAE_V'] <= args.max_global_regress_V
        normal_ok = row['delta_normal_MAE_V'] <= args.max_normal_regress_V
        rest_ok = row['delta_rest_MAE_V'] <= args.max_rest_regress_V
        high_ok = row['delta_high_MAE_V'] <= args.max_high_regress_V
        corr_ok = row['delta_corr'] >= -args.max_corr_drop
        mid_normal_ok = row['delta_mid_normal_3p20_4p10_MAE_V'] <= args.max_normal_regress_V
        # For S1J, mid_normal_ok is diagnostic; promotion uses original rule plus strict normal/rest/high/corr.
        promote = bool(low_ok and deep_ok and global_ok and normal_ok and rest_ok and high_ok and corr_ok)
        row.update({
            'low_ok': bool(low_ok), 'deep_ok': bool(deep_ok), 'global_ok': bool(global_ok),
            'normal_ok': bool(normal_ok), 'mid_normal_ok_diagnostic': bool(mid_normal_ok),
            'rest_ok': bool(rest_ok), 'high_ok': bool(high_ok), 'corr_ok': bool(corr_ok),
            'diagnostic_promotion': promote,
            'promote_to_next': promote,
        })
        decisions.append(row)

    promoted = [r['mode'] for r in decisions if r.get('promote_to_next')]
    summary = {
        'ok': True,
        'stage': 'D12-S1J 200ks normal/rest leakage diagnostic wrapper',
        'source_runs_root': str(args.source_runs_root),
        'output_runs_root': str(output_runs_root),
        'output_dir': str(output_dir),
        'prediction_count': int(sum(1 for r in run_rows if r.get('status') == 'metrics_ok')),
        'metrics_ok_count': int(sum(1 for r in run_rows if r.get('status') == 'metrics_ok')),
        'read_error_count': int(sum(1 for r in run_rows if r.get('status') == 'read_error')),
        'baseline_mode': baseline_mode,
        'source_candidate_mode': args.candidate_mode,
        'promoted_candidates': promoted,
        'decision_rule': {
            'low_and_deep_MAE_improve_at_least_V': args.min_low_improve_V,
            'global_MAE_regress_no_more_than_V': args.max_global_regress_V,
            'corr_drop_no_more_than': args.max_corr_drop,
            'rest_high_regress_no_more_than_V': args.max_rest_regress_V,
            'normal_regress_no_more_than_V': args.max_normal_regress_V,
        },
        'interpretation': 'If low-only or nonlow-budget variants promote, 200ks failure is caused by non-low/rest leakage rather than low-anchor failure.',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / 'D12_S1J_run_metrics.csv', run_rows)
    _write_csv(output_dir / 'D12_S1J_segment_metrics.csv', seg_rows)
    _write_csv(output_dir / 'D12_S1J_mode_summary.csv', mode_rows)
    _write_csv(output_dir / 'D12_S1J_candidate_decisions.csv', decisions)
    (output_dir / 'D12_S1J_scorecard_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def _build_recommendation(summary: Dict[str, Any], decisions: List[Dict[str, Any]], leakage_rows: List[Dict[str, Any]]) -> str:
    promoted = summary.get('promoted_candidates', [])
    lines: List[str] = []
    lines.append('# D12-S1J Recommendation')
    lines.append('')
    lines.append('D12-S1J is a diagnostic-only wrapper stage. It does not train a new model; it reads D12-S1E-soft 200ks predictions and tests whether 200ks failures are caused by normal/rest/global leakage.')
    lines.append('')
    lines.append(f"promoted_candidates = {promoted}")
    lines.append('')
    if promoted:
        lines.append('## Recommendation')
        lines.append('Use the most conservative promoted variant as the next wrapper blueprint. Prefer variants in this order:')
        lines.append('1. d12s1j_low_only_revert_nonlow_to_baseline')
        lines.append('2. d12s1j_low_plus_transition_fade_to_baseline')
        lines.append('3. d12s1j_low_full_nonlow_budget_10mV_rest_revert')
        lines.append('4. d12s1j_low_preserve_normal_rest_high_revert')
        lines.append('')
        lines.append('Do not train another high-safe model until the promoted wrapper is checked on 6-profile 200ks and then 23-profile confirmation.')
    else:
        lines.append('## Recommendation')
        lines.append('No diagnostic wrapper promoted. This means the 200ks leakage cannot be fixed by simple low-only / non-low-budget / rest-revert logic. Next step should be a source training change, not another post-wrapper.')
    lines.append('')
    lines.append('## Top decisions')
    for r in decisions:
        lines.append(f"- {r.get('mode')}: promote={r.get('promote_to_next')}, d_all={_safe_float(r.get('delta_all_MAE_V')):.6g}, d_low={_safe_float(r.get('delta_low_target_MAE_V')):.6g}, d_deep={_safe_float(r.get('delta_low_le_2p75_MAE_V')):.6g}, d_normal={_safe_float(r.get('delta_normal_MAE_V')):.6g}, d_rest={_safe_float(r.get('delta_rest_MAE_V')):.6g}, d_high={_safe_float(r.get('delta_high_MAE_V')):.6g}, d_corr={_safe_float(r.get('delta_corr')):.6g}")
    lines.append('')
    lines.append('## Leakage decomposition')
    lines.append('Use D12_S1J_leakage_decomposition.csv to inspect mean S1E correction vs baseline by segment/profile. Large positive/negative non-low/rest correction means S1E low anchor is leaking outside the intended low-voltage region.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_runs_root', required=True, help='S1E 6x200ks runs root containing baseline and S1E-soft prediction.npz files')
    ap.add_argument('--output_runs_root', required=True, help='Output runs root where S1J corrected prediction.npz files will be written')
    ap.add_argument('--output_dir', required=True, help='Output scorecard/diagnostic directory')
    ap.add_argument('--baseline_mode', default='baseline_d951')
    ap.add_argument('--candidate_mode', default='d12s1e_p2d_low_anchor_soft')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--min_low_improve_V', type=float, default=0.020)
    ap.add_argument('--max_global_regress_V', type=float, default=0.005)
    ap.add_argument('--max_corr_drop', type=float, default=0.005)
    ap.add_argument('--max_rest_regress_V', type=float, default=0.020)
    ap.add_argument('--max_high_regress_V', type=float, default=0.020)
    ap.add_argument('--max_normal_regress_V', type=float, default=0.005)
    args = ap.parse_args()

    source_root = Path(args.source_runs_root)
    output_runs_root = Path(args.output_runs_root)
    output_dir = Path(args.output_dir)
    if args.clean:
        if output_runs_root.exists():
            shutil.rmtree(output_runs_root)
        if output_dir.exists():
            shutil.rmtree(output_dir)
    output_runs_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_map: Dict[Tuple[str, str], Path] = {}
    for pred in sorted(source_root.rglob('prediction.npz')):
        mode, profile = _parse_run(pred)
        pred_map[(mode, profile)] = pred
    profiles = sorted({p for (m, p) in pred_map if m == args.baseline_mode} & {p for (m, p) in pred_map if m == args.candidate_mode})
    if not profiles:
        raise SystemExit(f'No overlapping profiles for baseline={args.baseline_mode} and candidate={args.candidate_mode} under {source_root}')

    leakage_rows: List[Dict[str, Any]] = []

    for profile in profiles:
        base_path = pred_map[(args.baseline_mode, profile)]
        cand_path = pred_map[(args.candidate_mode, profile)]
        base_arrays = _load_npz(base_path)
        cand_arrays = _load_npz(cand_path)
        y, base_pred, cur = _target_pred_current(base_arrays)
        y2, cand_pred, cur2 = _target_pred_current(cand_arrays)
        n = min(len(y), len(y2), len(base_pred), len(cand_pred), len(cur), len(cur2))
        y = y[:n]
        base_pred = base_pred[:n]
        cand_pred = cand_pred[:n]
        cur = cur[:n]
        delta = cand_pred - base_pred

        # Copy baseline into output so scorecard can compare every variant against it.
        _copy_baseline_npz(base_path, output_runs_root / f'{args.baseline_mode}__{profile}' / 'prediction.npz')

        # Write source identity and variants.
        for vname, (pred, allowed_mask) in _build_variants(y, base_pred, cand_pred, cur).items():
            _save_npz_like(cand_arrays, output_runs_root / f'{vname}__{profile}' / 'prediction.npz', pred[:n], base_pred[:n], cand_pred[:n], allowed_mask[:n], vname)

        # Leakage decomposition by segment for original S1E-soft vs baseline.
        masks = _segment_masks(y, cand_pred, cur)
        for seg, mask in masks.items():
            if not np.any(mask):
                leakage_rows.append({'profile': profile, 'segment': seg, 'n': 0})
                continue
            seg_delta = delta[mask]
            seg_err_cand = (cand_pred - y)[mask]
            seg_err_base = (base_pred - y)[mask]
            leakage_rows.append({
                'profile': profile,
                'segment': seg,
                'n': int(np.sum(mask)),
                's1e_minus_baseline_delta_mean_V': float(np.nanmean(seg_delta)),
                's1e_minus_baseline_delta_absmean_V': float(np.nanmean(np.abs(seg_delta))),
                's1e_minus_baseline_delta_p05_V': _q(seg_delta, 0.05),
                's1e_minus_baseline_delta_p50_V': _q(seg_delta, 0.50),
                's1e_minus_baseline_delta_p95_V': _q(seg_delta, 0.95),
                'baseline_MAE_V': _metrics(y[mask], base_pred[mask])['MAE_V'],
                's1e_soft_MAE_V': _metrics(y[mask], cand_pred[mask])['MAE_V'],
                'delta_MAE_V': _metrics(y[mask], cand_pred[mask])['MAE_V'] - _metrics(y[mask], base_pred[mask])['MAE_V'],
                'baseline_bias_V': float(np.nanmean(seg_err_base)),
                's1e_soft_bias_V': float(np.nanmean(seg_err_cand)),
            })

    _write_csv(output_dir / 'D12_S1J_leakage_decomposition.csv', leakage_rows)
    summary = _scorecard(output_runs_root, output_dir, args.baseline_mode, args)

    # Load decisions back for recommendation.
    decisions: List[Dict[str, Any]] = []
    dec_path = output_dir / 'D12_S1J_candidate_decisions.csv'
    if dec_path.exists():
        with dec_path.open('r', encoding='utf-8', newline='') as f:
            decisions = list(csv.DictReader(f))
    rec = _build_recommendation(summary, decisions, leakage_rows)
    (output_dir / 'D12_S1J_RECOMMENDATION.md').write_text(rec, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
