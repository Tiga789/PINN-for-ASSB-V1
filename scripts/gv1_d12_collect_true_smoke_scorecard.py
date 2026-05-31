#!/usr/bin/env python
"""Collect D12 TRUE SMOKE off/zero/on metrics if training completed."""
from __future__ import annotations
import argparse, json, csv, math, re
from pathlib import Path
import numpy as np

def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def f(x, default=math.nan):
    try: return float(x)
    except Exception: return default

def write_csv(path: Path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        w=csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

def classify(r):
    mae=f(r.get('mae_V')); corr=f(r.get('corr')); upper=f(r.get('pred_upper_frac_ge_4p269'),0); over=f(r.get('pred_overshoot_frac_gt_4p35'),0)
    if np.isfinite(mae) and np.isfinite(corr) and mae <= 0.14 and corr >= 0.85 and upper <= 0.15 and over <= 0.05:
        return 'smoke_pass'
    if np.isfinite(mae) and np.isfinite(corr): return 'smoke_review'
    return 'read_error'

def mode_from_name(name):
    m=re.search(r'd12_runtime_metadata_(off|zero|on)_', name)
    return m.group(1) if m else 'unknown'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache_root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--out_dir', default=None)
    args=ap.parse_args()
    cr=Path(args.cache_root)
    out=Path(args.out_dir) if args.out_dir else cr/'xjtu_batch134_d12_runtime_metadata_true_smoke_scorecard'
    dirs=sorted([p for p in cr.iterdir() if p.is_dir() and p.name.startswith('xjtu_batch134_d12_runtime_metadata_') and 'TRUE_SMOKE' in p.name])
    rows=[]
    for d in dirs:
        row={'run_dir':str(d),'run_name':d.name,'mode':mode_from_name(d.name)}
        meta=d/'d12_metadata_runtime_summary.json'
        if meta.exists():
            try:
                md=read_json(meta).get('metadata',{})
                row.update({'metadata_dim':md.get('metadata_dim'), 'metadata_profile_id':md.get('profile_id'), 'metadata_mode':md.get('mode')})
            except Exception as e:
                row['metadata_read_error']=str(e)
        mp=d/'d10_voltage_metrics.json'
        if mp.exists():
            try:
                data=read_json(mp); allm=data.get('metrics',{}).get('all',data.get('all',{}))
                row.update(allm)
            except Exception as e:
                row['metrics_read_error']=str(e)
        else:
            row['metrics_missing']='true'
        row['status']=classify(row)
        rows.append(row)
    counts={}
    for r in rows: counts[r['status']]=counts.get(r['status'],0)+1
    maes=[f(r.get('mae_V')) for r in rows if np.isfinite(f(r.get('mae_V')))]
    corrs=[f(r.get('corr')) for r in rows if np.isfinite(f(r.get('corr')))]
    summary={'ok':True,'stage':'D12 TRUE SMOKE scorecard','profile_count':len(rows),'counts':counts,'mean_mae_V':float(np.mean(maes)) if maes else math.nan,'mean_corr':float(np.mean(corrs)) if corrs else math.nan,'out_dir':str(out),'verdict':'d12_true_smoke_complete_review_scorecard' if rows else 'd12_true_smoke_no_runs_found'}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'d12_true_smoke_scorecard.csv', rows)
    (out/'d12_true_smoke_scorecard_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
