# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/d17_p34v_inspect_summary.py <D17_P34V_FINAL_VALIDATION_POLISH_SUMMARY.json>')
        raise SystemExit(2)
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding='utf-8'))
    agg = d.get('validation_aggregate_polished', {})
    print(json.dumps({
        'protocol': d.get('protocol'),
        'status': d.get('status'),
        'promotion_status': d.get('promotion_status'),
        'p4_ready': d.get('p4_ready'),
        'promotion_reasons': d.get('promotion_reasons'),
        'target_mae_V': d.get('target_mae_V'),
        'validation_mae_before_V': d.get('validation_aggregate_before', {}).get('corrected_voltage_mae_mean_V'),
        'validation_mae_polished_V': agg.get('voltage_mae_polished_V_mean'),
        'validation_mae_polished_max_V': agg.get('voltage_mae_polished_V_max'),
        'forward_voltage_mae_mean_V': agg.get('forward_voltage_mae_V_mean'),
        'residual_total_abs_mean_V': agg.get('new_total_residual_abs_mean_V_mean'),
        'residual_total_abs_max_V': agg.get('new_total_residual_abs_max_V_max'),
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
