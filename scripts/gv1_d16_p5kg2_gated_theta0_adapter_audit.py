from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# D16-P5K-G2 gated theta0 adapter audit
# No training. No checkpoint loading. No model modification.
#
# This audit is intentionally FAST: it reuses G1 by-profile exact-metric rows
# instead of rereading 50+ GB soft-label arrays. It composes candidate gated
# baselines by choosing one already-audited profile-level row per profile:
#   - default: P5K-C-baseline
#   - repair source: G1-theta0_oracle, G1-ridge_core_fit,
#                    G1-ridge_core_plus_hard_fit, or G1-rule_v1
#
# Important boundary:
#   * Any candidate using theta0_oracle or ridge_* is diagnostic only.
#   * G1-rule_v1 and gates based only on profile metadata are deployability
#     probes, but still require subsequent validation before training.
# -----------------------------------------------------------------------------

PROTOCOL_BY_BATCH = {
    'Batch-1': '2C',
    'Batch-2': '3C',
    'Batch-3': 'R2.5',
    'Batch-4': 'R3',
    'Batch-5': 'random_walk',
    'Batch-6': 'GEO',
}

REFERENCE_EVAL = {
    'theta_a_mean_mae': 0.139017,
    'theta_a_mean_r2': 0.474238,
    'theta_c_mean_mae': 0.123569,
    'theta_c_mean_r2': 0.391913,
}

METRICS = [
    'theta_a_mean', 'theta_c_mean',
    'theta_a', 'theta_c',
    'cs_a_mean', 'cs_c_mean',
    'grad_a_surface_center', 'grad_c_surface_center',
]

REQUIRED_MODELS = [
    'P5K-C-baseline',
    'G1-theta0_oracle',
    'G1-rule_v1',
    'G1-ridge_core_fit',
    'G1-ridge_core_plus_hard_fit',
]

KNOWN_HARD_PROFILE_IDS = {
    # hard_probe by design
    'profiles/Batch-5_battery-8',
    'profiles/Batch-1_battery-8',
    'profiles/Batch-6_battery-6',
    'profiles/Batch-2_battery-2',
    # recurring severe non-hard-probe / core-train/eval profiles from G0/G1/P5K-F
    'profiles/Batch-6_battery-5',
    'profiles/Batch-2_battery-3',
    'profiles/Batch-5_battery-2',
}


def finite_float(x: Any, default: float = float('nan')) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def fmt(v: Any, nd: int = 6) -> str:
    try:
        x = float(v)
        if math.isnan(x):
            return 'nan'
        if abs(x) >= 1e4 or (abs(x) < 1e-4 and x != 0):
            return f'{x:.6e}'
        return f'{x:.{nd}f}'
    except Exception:
        return str(v)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8'); return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def natural_battery_num(battery: str) -> int:
    m = re.search(r'(\d+)', str(battery))
    return int(m.group(1)) if m else -1


def find_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def infer_protocol(row: Dict[str, Any]) -> str:
    p = str(row.get('protocol', '') or '').strip()
    if p:
        return p
    return PROTOCOL_BY_BATCH.get(str(row.get('batch', '')), str(row.get('batch', '')))


def row_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row.get('model', ''), row.get('profile_id', ''))


def require_models(by_model_profile: Dict[Tuple[str, str], Dict[str, str]], profiles: List[str]) -> List[str]:
    missing: List[str] = []
    for m in REQUIRED_MODELS:
        n = sum(1 for p in profiles if (m, p) in by_model_profile)
        if n != len(profiles):
            missing.append(f'{m}: {n}/{len(profiles)} rows')
    return missing


def metric_count(row: Dict[str, str], metric: str) -> float:
    return finite_float(row.get(f'{metric}_count', '0'), 0.0)


def metric_sse(row: Dict[str, str], metric: str) -> float:
    return finite_float(row.get(f'{metric}_sum_err_sq', 'nan'))


def metric_sum_true(row: Dict[str, str], metric: str) -> float:
    return finite_float(row.get(f'{metric}_sum_true', 'nan'))


def metric_sum_true_sq(row: Dict[str, str], metric: str) -> float:
    return finite_float(row.get(f'{metric}_sum_true_sq', 'nan'))


def metric_sum_abs(row: Dict[str, str], metric: str) -> float:
    n = metric_count(row, metric)
    mae = finite_float(row.get(f'{metric}_mae', 'nan'))
    return mae * n if math.isfinite(mae) else float('nan')


def metric_sum_err(row: Dict[str, str], metric: str) -> float:
    n = metric_count(row, metric)
    bias = finite_float(row.get(f'{metric}_bias', 'nan'))
    return bias * n if math.isfinite(bias) else float('nan')


def metric_max_abs(row: Dict[str, str], metric: str) -> float:
    return finite_float(row.get(f'{metric}_max_abs', 'nan'))


def aggregate_rows(rows: List[Dict[str, str]], candidate: str, group: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {'candidate': candidate, 'group': group, 'profile_count': len(rows)}
    for metric in METRICS:
        n = 0.0; sum_abs = 0.0; sum_err = 0.0; sse = 0.0; st = 0.0; st2 = 0.0; mx = 0.0
        any_row = False
        for r in rows:
            rn = metric_count(r, metric)
            if rn <= 0:
                continue
            vals = [metric_sum_abs(r, metric), metric_sum_err(r, metric), metric_sse(r, metric), metric_sum_true(r, metric), metric_sum_true_sq(r, metric)]
            if not all(math.isfinite(x) for x in vals):
                continue
            any_row = True
            n += rn; sum_abs += vals[0]; sum_err += vals[1]; sse += vals[2]; st += vals[3]; st2 += vals[4]
            mxa = metric_max_abs(r, metric)
            if math.isfinite(mxa):
                mx = max(mx, mxa)
        if any_row and n > 0:
            sst = st2 - st * st / max(n, 1.0)
            out[f'{metric}_count'] = int(n)
            out[f'{metric}_mae'] = sum_abs / n
            out[f'{metric}_rmse'] = math.sqrt(max(sse, 0.0) / n)
            out[f'{metric}_bias'] = sum_err / n
            out[f'{metric}_max_abs'] = mx
            out[f'{metric}_r2'] = 1.0 - sse / sst if sst > 1e-20 else float('nan')
            out[f'{metric}_sum_err_sq'] = sse
            out[f'{metric}_sum_true'] = st
            out[f'{metric}_sum_true_sq'] = st2
    return out


class Candidate:
    def __init__(self, name: str, repair_model: str, gate_name: str, deployability: str, note: str):
        self.name = name
        self.repair_model = repair_model
        self.gate_name = gate_name
        self.deployability = deployability
        self.note = note


def oracle_shift_mag(row: Dict[str, str]) -> float:
    a = abs(finite_float(row.get('oracle_shift_a', 'nan'), 0.0))
    c = abs(finite_float(row.get('oracle_shift_c', 'nan'), 0.0))
    return max(a, c)


def gate(profile: str, base_row: Dict[str, str], gate_name: str, args: argparse.Namespace) -> bool:
    split = str(base_row.get('split', ''))
    batch = str(base_row.get('batch', ''))
    battery = str(base_row.get('battery', ''))
    batt_num = natural_battery_num(battery)
    pid = str(base_row.get('profile_id', profile))
    shift_mag = oracle_shift_mag(base_row)

    if gate_name == 'none':
        return False
    if gate_name == 'all':
        return True
    if gate_name == 'split_hard_probe':
        return split == 'hard_probe'
    if gate_name == 'known_hard_profile_list':
        return pid in KNOWN_HARD_PROFILE_IDS
    if gate_name == 'oracle_shift_mag_0p25':
        return shift_mag >= 0.25
    if gate_name == 'oracle_shift_mag_0p20':
        return shift_mag >= 0.20
    if gate_name == 'observed_metadata_v1':
        # Pure metadata-only rule derived from repeated audit failures, no soft-label values.
        # Conservative: target only highly recurrent hard regimes.
        return (
            (batch == 'Batch-1' and batt_num == 8) or
            (batch == 'Batch-2' and batt_num in (2, 3)) or
            (batch == 'Batch-5' and batt_num in (8, 2)) or
            (batch == 'Batch-6' and batt_num in (5, 6))
        )
    if gate_name == 'observed_metadata_strict_v1':
        return (
            (batch == 'Batch-1' and batt_num == 8) or
            (batch == 'Batch-5' and batt_num == 8) or
            (batch == 'Batch-6' and batt_num == 6) or
            (batch == 'Batch-2' and batt_num == 2)
        )
    if gate_name == 'observed_protocol_batch5_geo_v1':
        return (
            (batch == 'Batch-5' and batt_num >= 7) or
            (batch == 'Batch-6' and batt_num >= 5) or
            (batch == 'Batch-1' and batt_num == 8) or
            (batch == 'Batch-2' and batt_num in (2, 3))
        )
    raise ValueError(f'unknown gate_name: {gate_name}')


def build_candidates() -> List[Candidate]:
    cands: List[Candidate] = []
    cands.append(Candidate('P5K-C-baseline', 'P5K-C-baseline', 'none', 'deployable_reference', 'Default P5K-C hard baseline only.'))
    cands.append(Candidate('G2-deployable_rule_v1_gated_observed_metadata_strict', 'G1-rule_v1', 'observed_metadata_strict_v1', 'deployability_probe', 'Use G1 pure rule only for strict observed metadata hard regimes; diagnostic deployability probe.'))
    cands.append(Candidate('G2-deployable_rule_v1_gated_observed_metadata_v1', 'G1-rule_v1', 'observed_metadata_v1', 'deployability_probe', 'Use G1 pure rule for broader metadata hard regimes; diagnostic deployability probe.'))
    # Diagnostic gates using theta0 oracle / ridge rows.
    for repair in ['G1-theta0_oracle', 'G1-ridge_core_fit', 'G1-ridge_core_plus_hard_fit']:
        tag = repair.replace('G1-', '')
        cands.append(Candidate(f'G2-split_hard_probe__{tag}', repair, 'split_hard_probe', 'diagnostic_only', 'Apply repair only to manifest hard_probe split.'))
        cands.append(Candidate(f'G2-known_hard_profile_list__{tag}', repair, 'known_hard_profile_list', 'diagnostic_only', 'Apply repair to recurring known hard profile list from audits.'))
        cands.append(Candidate(f'G2-observed_metadata_strict_v1__{tag}', repair, 'observed_metadata_strict_v1', 'diagnostic_only', 'Apply repair to strict metadata hard regimes.'))
        cands.append(Candidate(f'G2-observed_metadata_v1__{tag}', repair, 'observed_metadata_v1', 'diagnostic_only', 'Apply repair to broader metadata hard regimes.'))
        cands.append(Candidate(f'G2-oracle_shift_mag_0p25__{tag}', repair, 'oracle_shift_mag_0p25', 'oracle_gate_diagnostic', 'Gate by soft-label oracle shift magnitude >=0.25. Upper-bound classifier.'))
        cands.append(Candidate(f'G2-oracle_shift_mag_0p20__{tag}', repair, 'oracle_shift_mag_0p20', 'oracle_gate_diagnostic', 'Gate by soft-label oracle shift magnitude >=0.20. Upper-bound classifier.'))
    return cands


def eval_candidate(cand: Candidate, profiles: List[str], by_mp: Dict[Tuple[str, str], Dict[str, str]], args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    selected: List[Dict[str, str]] = []
    gate_rows: List[Dict[str, Any]] = []
    for pid in profiles:
        base = by_mp[('P5K-C-baseline', pid)]
        use_gate = gate(pid, base, cand.gate_name, args)
        src_model = cand.repair_model if use_gate else 'P5K-C-baseline'
        row = by_mp.get((src_model, pid))
        if row is None:
            row = base
            src_model = 'P5K-C-baseline'
            use_gate = False
        selected.append(row)
        gate_rows.append({
            'candidate': cand.name,
            'profile_id': pid,
            'batch': base.get('batch',''),
            'battery': base.get('battery',''),
            'split': base.get('split',''),
            'gate': bool(use_gate),
            'gate_name': cand.gate_name,
            'selected_model': src_model,
            'oracle_shift_a': finite_float(base.get('oracle_shift_a','nan')),
            'oracle_shift_c': finite_float(base.get('oracle_shift_c','nan')),
            'oracle_shift_mag': oracle_shift_mag(base),
        })
    return selected, gate_rows


def profile_count_by_group(rows: List[Dict[str, str]], group: str) -> int:
    if group == 'ALL':
        return len({r.get('profile_id','') for r in rows})
    if group.startswith('split:'):
        s = group.split(':',1)[1]
        return len({r.get('profile_id','') for r in rows if r.get('split','') == s})
    return 0


def candidate_split_rows(cand: Candidate, selected: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups = ['ALL'] + [f'split:{s}' for s in ['eval','core_train','hard_probe','train']]
    for g in groups:
        if g == 'ALL':
            subset = selected
            group_label = 'ALL'
        else:
            split = g.split(':',1)[1]
            subset = [r for r in selected if r.get('split','') == split]
            group_label = split
        if not subset:
            continue
        row = aggregate_rows(subset, cand.name, group_label)
        row['deployability'] = cand.deployability
        row['gate_name'] = cand.gate_name
        row['repair_model'] = cand.repair_model
        row['note'] = cand.note
        out.append(row)
    return out


def pass_eval_no_regression(row: Dict[str, Any], tol_mae: float, tol_r2: float) -> bool:
    return (
        finite_float(row.get('theta_a_mean_mae')) <= REFERENCE_EVAL['theta_a_mean_mae'] + tol_mae and
        finite_float(row.get('theta_c_mean_mae')) <= REFERENCE_EVAL['theta_c_mean_mae'] + tol_mae and
        finite_float(row.get('theta_a_mean_r2')) >= REFERENCE_EVAL['theta_a_mean_r2'] - tol_r2 and
        finite_float(row.get('theta_c_mean_r2')) >= REFERENCE_EVAL['theta_c_mean_r2'] - tol_r2
    )


def hard_improved(row: Dict[str, Any]) -> bool:
    return (
        finite_float(row.get('theta_a_mean_mae')) < 0.20 and
        finite_float(row.get('theta_c_mean_mae')) < 0.20 and
        finite_float(row.get('theta_a_mean_r2')) > -0.25 and
        finite_float(row.get('theta_c_mean_r2')) > -0.50
    )


def score_eval(row: Dict[str, Any]) -> float:
    # Lower is better; combines MAE and lack of R2.
    return (
        finite_float(row.get('theta_a_mean_mae'), 9) + finite_float(row.get('theta_c_mean_mae'), 9)
        + max(0.0, 0.85 - finite_float(row.get('theta_a_mean_r2'), -9)) * 0.10
        + max(0.0, 0.85 - finite_float(row.get('theta_c_mean_r2'), -9)) * 0.10
    )


def md_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for r in rows:
        vals = []
        for c in columns:
            vals.append(fmt(r.get(c,'')))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-G2 gated theta0 adapter audit from G1 by-profile metrics. No training.')
    ap.add_argument('--cache-root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--g1-by-profile', default='')
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--allow-overwrite', action='store_true')
    ap.add_argument('--tolerate-eval-mae', type=float, default=0.002)
    ap.add_argument('--tolerate-eval-r2', type=float, default=0.02)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    default_g1_paths = [
        cache_root / 'xjtu_d16_p5kg1_MINI_EVIDENCE' / 'D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv',
        cache_root / 'xjtu_d16_p5kg1_observed_theta0_audit' / 'D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv',
    ]
    g1_path = Path(args.g1_by_profile) if args.g1_by_profile.strip() else find_existing(default_g1_paths)
    if g1_path is None or not g1_path.exists():
        raise FileNotFoundError(
            'Cannot find G1 by-profile CSV. Expected one of:\n  ' +
            '\n  '.join(str(p) for p in default_g1_paths) +
            '\nRestore it from xjtu_d16_p5kg1_MINI_EVIDENCE or rerun G1 audit.'
        )
    out_dir = Path(args.out_dir) if args.out_dir.strip() else cache_root / 'xjtu_d16_p5kg2_gated_theta0_adapter_audit'
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and non-empty: {out_dir}; pass --allow-overwrite')
    if out_dir.exists() and args.allow_overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(g1_path)
    by_mp: Dict[Tuple[str, str], Dict[str, str]] = {row_key(r): r for r in rows}
    profiles = sorted({r.get('profile_id','') for r in rows if r.get('model','') == 'P5K-C-baseline'})
    missing = require_models(by_mp, profiles)
    failures: List[Dict[str, Any]] = []
    if missing:
        failures.append({'error': 'missing_required_rows', 'details': '; '.join(missing)})

    all_split_rows: List[Dict[str, Any]] = []
    all_gate_rows: List[Dict[str, Any]] = []
    candidate_summary: List[Dict[str, Any]] = []
    candidates = build_candidates()

    for cand in candidates:
        selected, gate_rows = eval_candidate(cand, profiles, by_mp, args)
        rows_c = candidate_split_rows(cand, selected)
        all_split_rows.extend(rows_c)
        all_gate_rows.extend(gate_rows)
        eval_row = next((r for r in rows_c if r['group'] == 'eval'), None)
        hard_row = next((r for r in rows_c if r['group'] == 'hard_probe'), None)
        core_row = next((r for r in rows_c if r['group'] == 'core_train'), None)
        gated_count = sum(1 for g in gate_rows if g['gate'])
        eval_pass = bool(eval_row and pass_eval_no_regression(eval_row, args.tolerate_eval_mae, args.tolerate_eval_r2))
        hard_pass = bool(hard_row and hard_improved(hard_row))
        candidate_summary.append({
            'candidate': cand.name,
            'deployability': cand.deployability,
            'repair_model': cand.repair_model,
            'gate_name': cand.gate_name,
            'gated_count': gated_count,
            'eval_profile_count': eval_row.get('profile_count') if eval_row else 0,
            'hard_profile_count': hard_row.get('profile_count') if hard_row else 0,
            'eval_theta_a_mean_mae': eval_row.get('theta_a_mean_mae') if eval_row else float('nan'),
            'eval_theta_a_mean_r2': eval_row.get('theta_a_mean_r2') if eval_row else float('nan'),
            'eval_theta_c_mean_mae': eval_row.get('theta_c_mean_mae') if eval_row else float('nan'),
            'eval_theta_c_mean_r2': eval_row.get('theta_c_mean_r2') if eval_row else float('nan'),
            'hard_theta_a_mean_mae': hard_row.get('theta_a_mean_mae') if hard_row else float('nan'),
            'hard_theta_a_mean_r2': hard_row.get('theta_a_mean_r2') if hard_row else float('nan'),
            'hard_theta_c_mean_mae': hard_row.get('theta_c_mean_mae') if hard_row else float('nan'),
            'hard_theta_c_mean_r2': hard_row.get('theta_c_mean_r2') if hard_row else float('nan'),
            'eval_no_regression_pass': eval_pass,
            'hard_improvement_pass': hard_pass,
            'both_gates_pass': eval_pass and hard_pass,
            'eval_score_lower_better': score_eval(eval_row) if eval_row else float('nan'),
            'note': cand.note,
        })

    by_prof_out = out_dir / 'D16_P5KG2_GATED_THETA0_BY_PROFILE_SELECTION.csv'
    split_out = out_dir / 'D16_P5KG2_GATED_THETA0_SPLIT_METRICS.csv'
    summary_out = out_dir / 'D16_P5KG2_GATED_THETA0_CANDIDATE_SUMMARY.csv'
    failures_out = out_dir / 'D16_P5KG2_GATED_THETA0_FAILURES.json'
    report_out = out_dir / 'D16_P5KG2_GATED_THETA0_ADAPTER_AUDIT_REPORT.md'

    write_csv(all_gate_rows, by_prof_out)
    write_csv(all_split_rows, split_out)
    write_csv(candidate_summary, summary_out)
    write_json(failures, failures_out)

    # Report
    lines: List[str] = []
    lines.append('# D16-P5K-G2 Gated Theta0 Adapter Audit Report')
    lines.append('')
    lines.append('This is a **no-training** audit. It does not load checkpoints and does not modify any model. It composes gated candidates from the G1 by-profile exact metrics to test whether a hard-regime gate can preserve normal eval while repairing hard_probe theta0/OCP-phase failures.')
    lines.append('')
    lines.append('Important boundary: candidates using `theta0_oracle` or `ridge_*` are diagnostic only. `rule_v1` is the only pure observed-only theta0 correction from G1, and the G2 candidates using it are deployability probes, not promotion-ready training baselines.')
    lines.append('')
    lines.append('## 0. Run metadata')
    lines.append(f'- g1_by_profile_csv: `{g1_path}`')
    lines.append(f'- out_dir: `{out_dir}`')
    lines.append(f'- profile_count: `{len(profiles)}`')
    lines.append(f'- failure_count: `{len(failures)}`')
    lines.append(f'- eval_no_regression_tolerance: MAE +{args.tolerate_eval_mae}, R2 -{args.tolerate_eval_r2}')
    lines.append('')
    lines.append('## 1. Reference gates')
    ref_rows = [{'reference':'P5K-C-baseline eval from G1/G0', **REFERENCE_EVAL}]
    lines.extend(md_table(ref_rows, ['reference','theta_a_mean_mae','theta_a_mean_r2','theta_c_mean_mae','theta_c_mean_r2']))
    lines.append('')
    lines.append('## 2. Candidate summary')
    # sort: pass both first, then eval score
    sorted_summary = sorted(candidate_summary, key=lambda r: (not bool(r.get('both_gates_pass')), finite_float(r.get('eval_score_lower_better'), 999)))
    cols = ['candidate','deployability','gated_count','eval_theta_a_mean_mae','eval_theta_a_mean_r2','eval_theta_c_mean_mae','eval_theta_c_mean_r2','hard_theta_a_mean_mae','hard_theta_a_mean_r2','hard_theta_c_mean_mae','hard_theta_c_mean_r2','eval_no_regression_pass','hard_improvement_pass','both_gates_pass']
    lines.extend(md_table(sorted_summary, cols))
    lines.append('')
    lines.append('## 3. Split metrics')
    split_cols = ['candidate','group','profile_count','theta_a_mean_mae','theta_a_mean_bias','theta_a_mean_r2','theta_c_mean_mae','theta_c_mean_bias','theta_c_mean_r2','cs_a_mean_r2','cs_c_mean_r2']
    display_splits = [r for r in all_split_rows if r.get('group') in ('eval','hard_probe','core_train')]
    lines.extend(md_table(display_splits, split_cols))
    lines.append('')
    lines.append('## 4. Automatic verdict')
    both_pass = [r for r in candidate_summary if r.get('both_gates_pass')]
    deployable_pass = [r for r in both_pass if r.get('deployability') in ('deployability_probe','deployable_reference') and r.get('candidate') != 'P5K-C-baseline']
    diagnostic_pass = [r for r in both_pass if 'diagnostic' in str(r.get('deployability')) or 'oracle' in str(r.get('deployability'))]
    if deployable_pass:
        best = sorted(deployable_pass, key=lambda r: finite_float(r.get('eval_score_lower_better'), 999))[0]
        lines.append(f"- **DEPLOYABILITY-PROBE PASS:** `{best['candidate']}` satisfies the loose G2 audit gates. This still requires full baseline-only array audit and then training no-regression before promotion.")
    elif diagnostic_pass:
        best = sorted(diagnostic_pass, key=lambda r: finite_float(r.get('eval_score_lower_better'), 999))[0]
        lines.append(f"- **DIAGNOSTIC PASS ONLY:** `{best['candidate']}` satisfies gates but uses oracle/ridge information. It shows that gated theta0 repair could work if a valid observed-only classifier/adapter is built.")
    else:
        lines.append('- **NO CANDIDATE PASSED BOTH GATES.** Do not start P5K-G training. Improve hard-regime classifier / observed theta0 estimator first.')
    best_eval = sorted(candidate_summary, key=lambda r: finite_float(r.get('eval_score_lower_better'), 999))[0]
    best_hard = sorted(candidate_summary, key=lambda r: finite_float(r.get('hard_theta_a_mean_mae'), 999)+finite_float(r.get('hard_theta_c_mean_mae'), 999))[0]
    lines.append(f"- Best eval candidate by combined score: `{best_eval['candidate']}`.")
    lines.append(f"- Best hard_probe candidate by MAE sum: `{best_hard['candidate']}`.")
    lines.append('- If the best hard_probe candidate is oracle/ridge but deployability probes fail, the next step is a stronger observed-only hard-regime classifier, not long residual training.')
    lines.append('')
    lines.append('## 5. Gated profile counts by candidate and split')
    gate_count_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        grs = [g for g in all_gate_rows if g['candidate'] == cand.name]
        for split in sorted(set(g['split'] for g in grs)):
            gate_count_rows.append({'candidate': cand.name, 'split': split, 'profile_count': len([g for g in grs if g['split']==split]), 'gated_count': len([g for g in grs if g['split']==split and g['gate']])})
    lines.extend(md_table(gate_count_rows, ['candidate','split','profile_count','gated_count']))
    lines.append('')
    lines.append('## 6. Output files')
    lines.append(f'- by_profile_selection_csv: `{by_prof_out}`')
    lines.append(f'- split_metrics_csv: `{split_out}`')
    lines.append(f'- candidate_summary_csv: `{summary_out}`')
    lines.append(f'- failures_json: `{failures_out}`')
    report_out.write_text('\n'.join(lines), encoding='utf-8')

    print(f'[D16-P5K-G2] wrote report: {report_out}', flush=True)
    print(f'[D16-P5K-G2] failure_count={len(failures)} profile_count={len(profiles)}', flush=True)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
