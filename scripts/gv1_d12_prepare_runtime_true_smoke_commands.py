#!/usr/bin/env python
"""Prepare TRUE D12 runtime metadata smoke commands.

This rescue script replaces the mistaken long-run generated commands from the
first D12 runtime patch.  It generates only short 1-profile smoke commands:
50 epochs, 40 ks window, 1024 sampled time points, batch size 512.
It does not run training by itself.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path


def read_csv(p: Path):
    with p.open('r', newline='', encoding='utf-8-sig') as f:
        return [dict(r) for r in csv.DictReader(f)]

def first(row, cols):
    for c in cols:
        v=str(row.get(c,'')).strip()
        if v:
            return v
    return ''

def pid(row):
    return first(row,['profile_id','cell_uid','profile_key','label']) or '_'.join(str(row.get(c,'')).strip() for c in ['batch_id','protocol','battery_id'])

def safe(s):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s)).strip('_') or 'profile'

def q(s):
    return '"'+str(s).replace('"','`"')+'"'

def sol(row):
    v=first(row,['profile_npz','solution_npz','npz_path','source_npz'])
    if v:
        return v
    d=first(row,['prepared_dir','profile_dir'])
    return str(Path(d)/'solution_replay_profile.npz') if d else ''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project_root', default=r'C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1')
    ap.add_argument('--cache_root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--d12_plan_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--profile_limit', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--time_window_s', type=float, default=40000.0)
    ap.add_argument('--max_time_points', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--prediction_time_points', type=int, default=1024)
    ap.add_argument('--prediction_radial_points', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--python', default=r'D:\Anaconda\envs\torchgpu\python.exe')
    args=ap.parse_args()

    # Hard safety guard: this script must not be used to prepare long runs.
    if args.epochs > 200 or args.time_window_s > 60000 or args.max_time_points > 2048 or args.batch_size > 1024:
        raise SystemExit('Refusing to generate non-smoke D12 runtime commands. Use <=200 epochs, <=60ks, <=2048 points, <=1024 batch.')

    cr=Path(args.cache_root)
    d12=Path(args.d12_plan_dir) if args.d12_plan_dir else cr/'xjtu_batch134_d12_metadata_on_off_ablation_plan'
    out=Path(args.out_dir) if args.out_dir else cr/'xjtu_batch134_d12_runtime_metadata_true_smoke_commands'
    out.mkdir(parents=True, exist_ok=True)
    on_manifest=d12/'d12_metadata_on_23profile_manifest.csv'
    off_manifest=d12/'d12_metadata_off_23profile_manifest.csv'
    rows=read_csv(on_manifest)
    n=max(1,min(args.profile_limit,len(rows)))
    selected=rows[:n]
    wrapper='scripts\\gv1_train_conditioned_pinn_d12_metadata_runtime.py'

    def lines(mode, manifest, tag):
        L=["$ErrorActionPreference = 'Stop'", f"Set-Location {q(args.project_root)}", f"$Python = {q(args.python)}", '']
        for i,r in enumerate(selected,1):
            p=pid(r); s=sol(r)
            runout=cr/f'xjtu_batch134_d12_runtime_{tag}_{safe(p)}_TRUE_SMOKE_40ks_e{args.epochs}'
            L += [f"Write-Host 'D12 TRUE SMOKE {tag} {i}/{n}: {p}'",
                  f"& $Python {wrapper} `",
                  f"  --metadata_mode {mode} `",
                  f"  --metadata_manifest {q(str(manifest))} `",
                  f"  --metadata_profile_id {q(p)} `",
                  f"  --metadata_strict_profile_match true `",
                  f"  --metadata_allow_target_probe false `",
                  f"  --solution_npz {q(s)} `",
                  f"  --output_dir {q(str(runout))} `",
                  f"  --profile_adaptive_mode auto `",
                  f"  --epochs {args.epochs} `",
                  f"  --batch_size {args.batch_size} `",
                  f"  --seed {args.seed} `",
                  f"  --device {args.device} `",
                  f"  --max_time_points {args.max_time_points} `",
                  f"  --time_window_s {args.time_window_s} `",
                  f"  --prediction_time_points {args.prediction_time_points} `",
                  f"  --prediction_radial_points {args.prediction_radial_points}",
                  "if ($LASTEXITCODE -ne 0) { throw 'D12 TRUE SMOKE runtime command failed.' }", '']
        return L
    scripts={
        'metadata_off': out/f'run_d12_runtime_TRUE_SMOKE_metadata_off_{n}profile.generated.ps1',
        'metadata_zero': out/f'run_d12_runtime_TRUE_SMOKE_metadata_zero_{n}profile.generated.ps1',
        'metadata_on': out/f'run_d12_runtime_TRUE_SMOKE_metadata_on_{n}profile.generated.ps1'}
    scripts['metadata_off'].write_text('\n'.join(lines('off',off_manifest,'metadata_off')),encoding='utf-8')
    scripts['metadata_zero'].write_text('\n'.join(lines('zero',on_manifest,'metadata_zero')),encoding='utf-8')
    scripts['metadata_on'].write_text('\n'.join(lines('on',on_manifest,'metadata_on')),encoding='utf-8')
    summary={
        'ok': True,
        'stage': 'D12 runtime TRUE SMOKE command preparation rescue',
        'verdict': 'd12_runtime_true_smoke_commands_prepared',
        'profile_limit': n,
        'selected_profile_ids': [pid(r) for r in selected],
        'epochs': args.epochs,
        'time_window_s': args.time_window_s,
        'max_time_points': args.max_time_points,
        'batch_size': args.batch_size,
        'prediction_time_points': args.prediction_time_points,
        'out_dir': str(out),
        'generated_scripts': {k:str(v) for k,v in scripts.items()},
        'invalid_previous_defaults': {'epochs': 40000, 'time_window_s': 200000.0, 'max_time_points': 8192, 'batch_size': 2048},
        'note': 'Generated only; not executed. Run preflight before executing.'}
    (out/'d12_runtime_TRUE_SMOKE_command_preparation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'D12_RUNTIME_TRUE_SMOKE_COMMANDS.md').write_text('# D12 Runtime TRUE SMOKE Commands\n\n```text\n'+summary['verdict']+'\n```\n\nThis rescue command set is intentionally short: 50 epochs / 40 ks / 1024 points / batch 512 by default.\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
