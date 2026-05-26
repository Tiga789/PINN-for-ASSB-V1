from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.measured_replay.profile_builder import build_replay_profile
from gv1.measured_replay.replay_audit import audit_replay_profile


def test_replay_profile_minimal():
    df = pd.DataFrame({
        'time_s': [0, 1, 2, 3],
        'current_A': [1.0, 1.0, 0.0, -1.0],
        'voltage_V': [3.6, 3.7, 3.7, 3.5],
        'cycle_id': [1, 1, 1, 1],
    })
    p = build_replay_profile(df)
    assert len(p.t_s) == 4
    assert p.q_charge_Ah[-1] > 0
    assert p.q_discharge_Ah[-1] > 0
    audit = audit_replay_profile(p)
    assert audit.ok
    assert audit.metrics['has_charge']
    assert audit.metrics['has_discharge']
