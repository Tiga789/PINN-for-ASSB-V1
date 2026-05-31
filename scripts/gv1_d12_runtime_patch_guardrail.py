#!/usr/bin/env python
"""Read-only guardrail audit for the D12 runtime metadata patch."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def read_text(p: Path) -> str:
    try: return p.read_text(encoding='utf-8', errors='ignore')
    except Exception: return ''

def read_json(p: Path) -> dict:
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}

def csv_count(p: Path) -> int:
    try:
        with p.open('r', newline='', encoding='utf-8-sig') as f: return sum(1 for _ in csv.DictReader(f))
    except Exception: return -1

def row(cid, ok, desc): return {'check_id': cid, 'status': 'pass' if ok else 'fail', 'description': desc}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project_root', default=r'C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1')
    ap.add_argument('--cache_root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--d12_plan_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    args=ap.parse_args()
    pr=Path(args.project_root); cr=Path(args.cache_root)
    d12=Path(args.d12_plan_dir) if args.d12_plan_dir else cr/'xjtu_batch134_d12_metadata_on_off_ablation_plan'
    out=Path(args.out_dir) if args.out_dir else cr/'xjtu_batch134_d12_runtime_metadata_patch_guardrail'
    out.mkdir(parents=True, exist_ok=True)
    summary=read_json(d12/'d12_metadata_on_off_ablation_summary.json')
    train=read_text(pr/'scripts/gv1_train_conditioned_pinn.py')
    trainer=read_text(pr/'gv1/trainer.py')
    wrapper=read_text(pr/'scripts/gv1_train_conditioned_pinn_d12_metadata_runtime.py')
    runtime=read_text(pr/'gv1/d12_metadata_runtime.py')
    rows=[
        row('R01',(pr/'gv1/d12_metadata_runtime.py').exists(),'new runtime module exists'),
        row('R02',(pr/'scripts/gv1_train_conditioned_pinn_d12_metadata_runtime.py').exists(),'new metadata wrapper exists'),
        row('R03','D9.5.1' in train and 'trend-first' in train,'D9.6/D9.5.1 train script signature remains present'),
        row('R04','D9.5.1' in trainer and 'trend-first' in trainer,'D9.6/D9.5.1 trainer signature remains present'),
        row('R05',summary.get('verdict')=='d12_metadata_on_off_ablation_plan_ready_no_mainline_overwrite','D12 plan verdict is ready'),
        row('R06',csv_count(d12/'d12_metadata_on_23profile_manifest.csv')==23,'metadata_on manifest has 23 non-target profiles'),
        row('R07',csv_count(d12/'d12_metadata_off_23profile_manifest.csv')==23,'metadata_off manifest has 23 non-target profiles'),
        row('R08',csv_count(d12/'d12_battery8_target_probe_manifest_not_mainline.csv')==1,'battery-8 target probe manifest has exactly 1 row'),
        row('R09','runpy.run_path' in wrapper and 'gv1_train_conditioned_pinn.py' in wrapper,'wrapper delegates to existing D9 trainer'),
        row('R10','self.condition' in runtime and 'concatenate' in runtime,'runtime augments condition vector process-locally'),
        row('R11','24-profile' not in wrapper.lower() and '200ks' not in wrapper.lower(),'wrapper does not generate direct 24-profile 200ks command'),
    ]
    counts={'pass':sum(r['status']=='pass' for r in rows),'fail':sum(r['status']=='fail' for r in rows)}
    verdict='d12_runtime_metadata_patch_guardrail_pass' if counts['fail']==0 else 'd12_runtime_metadata_patch_guardrail_fail'
    outsum={'ok': counts['fail']==0, 'stage':'D12 runtime metadata patch guardrail', 'verdict':verdict, 'guard_counts':counts, 'd12_verdict':summary.get('verdict'), 'out_dir':str(out)}
    with (out/'d12_runtime_metadata_patch_guardrail_checklist.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['check_id','status','description']); w.writeheader(); w.writerows(rows)
    (out/'d12_runtime_metadata_patch_guardrail_summary.json').write_text(json.dumps(outsum, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'D12_RUNTIME_PATCH_RECOMMENDATION.md').write_text('# D12 Runtime Metadata Patch Guardrail\n\n```text\n'+verdict+'\n```\n', encoding='utf-8')
    print(json.dumps(outsum, ensure_ascii=False, indent=2))
    if counts['fail']:
        raise SystemExit(2)

if __name__=='__main__': main()
