# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path


def get(d, path, default=None):
    cur=d
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur=cur[p]
        else:
            return default
    return cur


def main():
    ap=argparse.ArgumentParser(description='Inspect D17-P3.4 final forward-core promotion summary')
    ap.add_argument('summary_json')
    args=ap.parse_args()
    p=Path(args.summary_json)
    d=json.loads(p.read_text(encoding='utf-8'))
    out={
        'protocol': d.get('protocol'),
        'status': d.get('status'),
        'promotion_status': d.get('promotion_status'),
        'p4_ready': d.get('p4_ready'),
        'p4_blockers': d.get('p4_blockers'),
        'resolved_spec': d.get('resolved_spec'),
        'spec_voltage_fit_rmse_V': get(d,'p34_resolved_spec_alignment.voltage_only_fit.rmse_V'),
        'train_forward_mae_V': get(d,'voltage_recovery.train_forward_voltage_mae_mean_V'),
        'train_corrected_mae_V': get(d,'voltage_recovery.train_corrected_voltage_mae_mean_V'),
        'validation_forward_mae_V': get(d,'voltage_recovery.validation_forward_voltage_mae_mean_V'),
        'validation_corrected_mae_V': get(d,'voltage_recovery.validation_corrected_voltage_mae_mean_V'),
        'forward_core_reliability_status': get(d,'residual_budget_audit.forward_core_reliability_status'),
        'residual_budget_status': get(d,'residual_budget_audit.residual_budget_status'),
        'no_state_label_training': get(d,'no_state_label_policy.training_uses_state_softlabels'),
        'validation_uses_state_softlabels': get(d,'validation_adaptation.uses_state_softlabels'),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
