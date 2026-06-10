from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_rows(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def run_cmd(cmd: List[str], stdout_path: Path, stderr_path: Path) -> Tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(stdout_path, 'w', encoding='utf-8', errors='replace') as out, open(stderr_path, 'w', encoding='utf-8', errors='replace') as err:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    return int(p.returncode), time.perf_counter() - t0


def make_candidate_prior(base_prior_path: Path, cand: Dict[str, Any], out_path: Path) -> None:
    prior = load_json(base_prior_path)
    rg = prior.setdefault('radial_gradient', {})
    rg['alpha_D_negative'] = float(cand['alpha_D_negative'])
    rg['alpha_J_negative'] = float(cand.get('alpha_J_negative', rg.get('alpha_J_negative', 1.0)))
    rg['targeted_fix_note'] = {
        'stage': 'D15-P4D-targeted-fix',
        'target_cell': 'Batch-5_battery-8',
        'candidate': cand.get('name', ''),
        'reason': 'Original D15-P4D audit warned for weak active anode radial gradient in Batch-5_battery-8.',
        'only_negative_electrode_parameters_changed': True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(prior, out_path)


def extract_target_audit_metrics(audit_dir: Path) -> Dict[str, Any]:
    summary_path = audit_dir / 'radial_gradient_audit_summary.json'
    by_path = audit_dir / 'radial_gradient_audit_by_profile.csv'
    summary: Dict[str, Any] = load_json(summary_path) if summary_path.exists() else {}
    rows = read_csv_rows(by_path) if by_path.exists() else []
    row = rows[0] if rows else {}
    out: Dict[str, Any] = {
        'audit_overall_status': summary.get('overall_status', ''),
        'audit_profile_count': summary.get('profile_count', ''),
        'audit_pass_count': summary.get('pass_count', ''),
        'audit_warn_count': summary.get('warn_count', ''),
        'audit_fail_count': summary.get('fail_count', ''),
        'audit_read_error_count': summary.get('read_error_count', ''),
    }
    for k, v in row.items():
        if k in {
            'profile', 'status', 'a_status', 'c_status',
            'a_mean_active_abs_gradient_norm', 'a_p95_active_abs_gradient_norm',
            'a_direction_match_fraction', 'a_theta_outside_fraction', 'a_mass_cbar_mae_norm',
            'c_mean_active_abs_gradient_norm', 'c_p95_active_abs_gradient_norm',
            'c_direction_match_fraction', 'c_theta_outside_fraction', 'c_mass_cbar_mae_norm'
        }:
            out[k] = v
    return out


def is_candidate_pass(metrics: Dict[str, Any]) -> bool:
    return str(metrics.get('audit_overall_status', '')).upper() == 'PASS' and str(metrics.get('status', '')).upper() in ('PASS', '')


def copy_replace_cell(candidate_cell_dir: Path, final_cell_dir: Path, backup_dir: Path | None) -> None:
    if not candidate_cell_dir.exists():
        raise FileNotFoundError(f'Candidate cell dir does not exist: {candidate_cell_dir}')
    if backup_dir is not None and final_cell_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(final_cell_dir, backup_dir)
    if final_cell_dir.exists():
        shutil.rmtree(final_cell_dir)
    final_cell_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate_cell_dir, final_cell_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='D15-P4D targeted fix for Batch-5_battery-8 weak anode radial gradient.')
    ap.add_argument('--config', default='configs/d15_p4d_batch5_battery8_targeted_fix_config.json')
    ap.add_argument('--allow-overwrite', action='store_true', help='Allow overwriting the targeted fix run directory and target cell output after a candidate passes.')
    ap.add_argument('--candidate-limit', type=int, default=0, help='Try only the first N candidates; 0 means all candidates.')
    ap.add_argument('--backup-old', action='store_true', help='Copy the old target cell directory before replacement. This may take disk space/time.')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(ROOT / args.config)
    target = cfg['target_cell']
    run_dir = Path(cfg['target_fix_run_dir'])
    logs_dir = run_dir / 'logs'
    candidates_dir = run_dir / 'candidates'
    selected_dir = run_dir / 'selected_candidate'
    summary_json = run_dir / 'D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_SUMMARY.json'
    candidates_csv = run_dir / 'D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_CANDIDATES.csv'

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Run directory exists and is not empty: {run_dir}. Use --allow-overwrite for deliberate rerun.')
    if args.allow_overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    base_prior = Path(cfg['base_prior_json'])
    manifest_csv = Path(cfg['p4c_replay_manifest_csv'])
    final_root = Path(cfg['final_softlabels_dir'])
    final_cell_dir = final_root / 'profiles' / target
    gen_script = ROOT / 'scripts' / 'd15_p4d_full_generate_one_rg_softlabel.py'
    audit_script = ROOT / 'scripts' / 'd15_p0_radial_gradient_audit.py'

    preflight_failures: List[str] = []
    for p in [base_prior, manifest_csv, gen_script, audit_script]:
        if not p.exists():
            preflight_failures.append(str(p))
    if preflight_failures:
        write_json({'final_status': 'FAIL', 'failures': ['missing required file'] + preflight_failures}, summary_json)
        print('[D15-P4D targeted] FAIL missing files:', preflight_failures, flush=True)
        return 2

    cands = list(cfg.get('candidates', []))
    if args.candidate_limit and args.candidate_limit > 0:
        cands = cands[:args.candidate_limit]
    rows: List[Dict[str, Any]] = []
    selected: Dict[str, Any] | None = None
    selected_candidate_cell_dir: Path | None = None

    for idx, cand in enumerate(cands, start=1):
        name = cand.get('name') or f'candidate_{idx:02d}'
        cand_dir = candidates_dir / f'{idx:02d}_{name}'
        prior_path = cand_dir / 'P2Dlite_prior_targeted.json'
        soft_dir = cand_dir / 'softlabels'
        audit_dir = cand_dir / 'audit'
        make_candidate_prior(base_prior, cand, prior_path)

        print(f'[D15-P4D targeted] candidate {idx}/{len(cands)} {name}: alpha_D_negative={cand["alpha_D_negative"]} alpha_J_negative={cand.get("alpha_J_negative", 1.0)}', flush=True)

        gen_stdout = logs_dir / f'{idx:02d}_{name}_generate.out.log'
        gen_stderr = logs_dir / f'{idx:02d}_{name}_generate.err.log'
        gen_cmd = [
            sys.executable, str(gen_script),
            '--config', 'configs/d15_p4d_full_remaining14_config.json',
            '--manifest-csv', str(manifest_csv),
            '--cell-id', target,
            '--prior-json', str(prior_path),
            '--output-root', str(soft_dir),
            '--save-mode', cfg.get('save_mode', 'uncompressed'),
            '--status-dir', str(cand_dir / 'status'),
            '--overwrite-existing',
        ]
        gen_rc, gen_sec = run_cmd(gen_cmd, gen_stdout, gen_stderr)
        if gen_rc != 0:
            row = {
                'candidate_index': idx,
                'candidate_name': name,
                'alpha_D_negative': cand.get('alpha_D_negative'),
                'alpha_J_negative': cand.get('alpha_J_negative', 1.0),
                'generation_status': 'FAIL',
                'generation_returncode': gen_rc,
                'generation_seconds': round(gen_sec, 3),
            }
            rows.append(row)
            print(f'[D15-P4D targeted] generation FAIL candidate={name}; rc={gen_rc}', flush=True)
            continue

        audit_stdout = logs_dir / f'{idx:02d}_{name}_audit.out.log'
        audit_stderr = logs_dir / f'{idx:02d}_{name}_audit.err.log'
        audit_cmd = [
            sys.executable, str(audit_script),
            '--source-dir', str(soft_dir),
            '--prior-json', str(prior_path),
            '--out-dir', str(audit_dir),
        ]
        audit_rc, audit_sec = run_cmd(audit_cmd, audit_stdout, audit_stderr)
        metrics = extract_target_audit_metrics(audit_dir)
        cand_pass = audit_rc == 0 and is_candidate_pass(metrics)
        row = {
            'candidate_index': idx,
            'candidate_name': name,
            'alpha_D_negative': cand.get('alpha_D_negative'),
            'alpha_J_negative': cand.get('alpha_J_negative', 1.0),
            'generation_status': 'PASS',
            'generation_returncode': gen_rc,
            'generation_seconds': round(gen_sec, 3),
            'audit_returncode': audit_rc,
            'audit_seconds': round(audit_sec, 3),
            'candidate_selected': bool(cand_pass),
        }
        row.update(metrics)
        rows.append(row)
        print(f'[D15-P4D targeted] candidate={name} audit={metrics.get("audit_overall_status")} a_grad={metrics.get("a_mean_active_abs_gradient_norm")} selected={cand_pass}', flush=True)
        if cand_pass:
            selected = row
            selected_candidate_cell_dir = soft_dir / 'profiles' / target
            break

    write_csv_rows(rows, candidates_csv)

    if selected is None or selected_candidate_cell_dir is None:
        summary = {
            'stage': cfg['stage'],
            'target_cell': target,
            'final_status': 'REVIEW',
            'reason': 'No candidate reached targeted radial-audit PASS; old cell output was not replaced.',
            'target_output_replaced': False,
            'candidate_count': len(rows),
            'candidates_csv': str(candidates_csv),
            'run_dir': str(run_dir),
        }
        write_json(summary, summary_json)
        print('[D15-P4D targeted] REVIEW: no candidate selected; old output not replaced.', flush=True)
        return 1

    backup_dir: Path | None = None
    backup_requested = args.backup_old or bool(cfg.get('backup_old_cell_dir', False))
    if backup_requested:
        backup_dir = run_dir / 'backup_old_cell_dir' / target
    copy_replace_cell(selected_candidate_cell_dir, final_cell_dir, backup_dir)
    # Keep a copy of selected audit for review convenience.
    selected_audit_src = candidates_dir / f"{int(selected['candidate_index']):02d}_{selected['candidate_name']}" / 'audit'
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(selected_candidate_cell_dir, selected_dir / 'profile' / target)
    if selected_audit_src.exists():
        shutil.copytree(selected_audit_src, selected_dir / 'audit')

    summary = {
        'stage': cfg['stage'],
        'target_cell': target,
        'final_status': 'PASS',
        'target_output_replaced': True,
        'replacement_policy': 'Only Batch-5_battery-8 was regenerated and copied over the old P4D output. No other cell directory was modified by this script.',
        'selected_candidate': selected,
        'final_cell_dir': str(final_cell_dir),
        'backup_old_cell_dir': str(backup_dir) if backup_dir else None,
        'run_dir': str(run_dir),
        'candidates_csv': str(candidates_csv),
        'selected_audit_dir': str(selected_dir / 'audit'),
        'allowed_claim_if_pass': 'Batch-5_battery-8 targeted P2Dlite-RG soft label was regenerated with stronger anode radial-gradient prior and passed targeted radial-gradient audit.',
        'not_allowed_claims': [
            'Do not claim experimental internal-state truth for cs_a/cs_c.',
            'Do not claim other cells were regenerated by this targeted fix.',
            'Do not claim held-out generalization from this targeted fix.'
        ]
    }
    write_json(summary, summary_json)
    print('[D15-P4D targeted] PASS: replaced target cell output:', final_cell_dir, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
