#!/usr/bin/env python
"""Collect D12 runtime metadata on/off prediction scorecard."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def metrics(pred: Path):
    if not pred.exists(): return {'read_error':1}
    try:
        with np.load(pred,allow_pickle=True) as z:
            y=np.asarray(z['voltage_exp'],float).reshape(-1); p=np.asarray(z['voltage_exp_pred'],float).reshape(-1)
        n=min(len(y),len(p)); y=y[:n]; p=p[:n]; m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
        if len(y)<3: return {'read_error':1}
        e=p-y
        return {'read_error':0,'n':len(y),'mae_V':float(np.mean(abs(e))),'rmse_V':float(np.sqrt(np.mean(e*e))),'bias_V':float(np.mean(e)),'corr':float(np.corrcoef(p,y)[0,1]) if np.std(p)>1e-12 and np.std(y)>1e-12 else math.nan,'pred_max_V':float(np.max(p)),'pred_upper_frac_ge_4p269':float(np.mean(p>=4.269)),'pred_overshoot_frac_gt_4p35':float(np.mean(p>4.35))}
    except Exception as exc:
        return {'read_error':1,'error':str(exc)}

def status(r):
    if r.get('read_error'): return 'read_error'
    mae=float(r.get('mae_V',999)); corr=float(r.get('corr',0)); over=float(r.get('pred_overshoot_frac_gt_4p35',0))
    if mae<=0.11 and corr>=0.89 and over<=0.03: return 'pass'
    if mae<=0.14 and corr>=0.85: return 'borderline'
    return 'fail'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache_root',default=r'E:\XJTU battery dataset\_gv1_cache'); ap.add_argument('--out_dir',default=None); args=ap.parse_args()
    cr=Path(args.cache_root); out=Path(args.out_dir) if args.out_dir else cr/'xjtu_batch134_d12_runtime_metadata_ablation_scorecard'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode in ['metadata_off','metadata_zero','metadata_on']:
        for d in sorted(cr.glob(f'xjtu_batch134_d12_runtime_{mode}_*_200ks')):
            r={'mode':mode,'profile_id':d.name.replace(f'xjtu_batch134_d12_runtime_{mode}_','').replace('_200ks',''),'run_dir':str(d)}; r.update(metrics(d/'prediction.npz')); r['status']=status(r); rows.append(r)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    csvp=out/'d12_runtime_onoff_scorecard.csv'
    with csvp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['mode','profile_id','status']); w.writeheader(); w.writerows(rows)
    counts={}
    for r in rows: counts[f"{r.get('mode')}:{r.get('status')}"]=counts.get(f"{r.get('mode')}:{r.get('status')}",0)+1
    summ={'ok':True,'stage':'D12 runtime metadata on/off scorecard','run_count':len(rows),'counts':counts,'csv':str(csvp),'out_dir':str(out)}
    (out/'d12_runtime_onoff_scorecard_summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summ,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
