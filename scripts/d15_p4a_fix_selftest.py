#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import re

def canonicalize_from_text(text: str):
    t = str(text).replace('\\\\', '/')
    m = re.findall(r'Batch[-_](?P<batch>[1-6])(?:\D|$)', t)
    batch = int(m[-1]) if m else None
    batt = re.findall(r'battery[-_](?P<battery>\d+)', t, flags=re.IGNORECASE)
    battery = int(batt[-1]) if batt else None
    if batch is None:
        for tok, bb in [('R2.5', 3), ('R2_5', 3), ('R3', 4), ('3C', 2), ('2C', 1), ('random_walk', 5), ('random-walk', 5), ('GEO', 6)]:
            if tok in t:
                batch = bb
                break
    if batch and battery:
        return f'Batch-{batch}_battery-{battery}'
    return None

cases = {
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0019_battery-3_R3_battery-3\solution_replay_profile.npz': 'Batch-4_battery-3',
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0014_battery-6_R2.5_battery-6\solution_replay_profile.npz': 'Batch-3_battery-6',
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0003_battery-3_2C_battery-3\solution_replay_profile.npz': 'Batch-1_battery-3',
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_batch2_replay_profiles_d15p3\profiles\Batch-2_3C_battery-15\solution_replay_profile.npz': 'Batch-2_battery-15',
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3b_batch56_replay_smoke\profiles\Batch-6_Batch-6_battery-3\solution_replay_profile.npz': 'Batch-6_battery-3',
    r'E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3b_batch56_replay_smoke\profiles\Batch-5_Batch-5_battery-7\solution_replay_profile.npz': 'Batch-5_battery-7',
}
for text, expected in cases.items():
    got = canonicalize_from_text(text)
    if got != expected:
        raise AssertionError(f'{text!r}: expected {expected}, got {got}')
print('[D15-P4A-fix selftest] PASS')
