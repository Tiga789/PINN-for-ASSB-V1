from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn_precision.audit import audit_prediction_file, write_json


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pred_dir = root / 'eval' / 'predictions'
        pred_dir.mkdir(parents=True)
        soft = root / 'soft' / 'profiles' / 'fake_profile'
        soft.mkdir(parents=True)
        n = 80
        nr = 17
        t = np.arange(n, dtype=np.float32)
        I = np.sin(np.linspace(0, 8 * np.pi, n)).astype(np.float32)
        cycle = np.repeat(np.arange(20, 40), 4)[:n].astype(np.int32)
        theta_a = np.clip(0.45 + 0.05 * np.sin(t[:, None] / 7) + np.linspace(-0.02, 0.02, nr)[None, :], 0, 1).astype(np.float32)
        theta_c = np.clip(0.55 - 0.04 * np.sin(t[:, None] / 8) + np.linspace(0.015, -0.015, nr)[None, :], 0, 1).astype(np.float32)
        phie = (0.1 * np.sin(t / 9)).astype(np.float32)
        phis = (3.6 + 0.2 * np.sin(t / 11)).astype(np.float32)
        np.savez_compressed(soft / 'solution_softlabels.npz', t_global_s=t, I_profile=I, cycle_id=cycle, theta_a=theta_a, theta_c=theta_c, phie=phie, phis_c=phis)
        Y = np.concatenate([theta_a, theta_c, phie[:, None], phis[:, None]], axis=1).astype(np.float32)
        names = [f'theta_a_r{i:02d}' for i in range(nr)] + [f'theta_c_r{i:02d}' for i in range(nr)] + ['phie', 'phis_c']
        Yp = Y + np.random.default_rng(1).normal(0, 1e-4, size=Y.shape).astype(np.float32)
        np.savez_compressed(pred_dir / 'fake_prediction.npz', t_global_s=t, y_true=Y, y_pred=Yp, target_names=np.array(names), feature_names=np.array(['dummy']), profile_id=np.array('profiles/fake_profile'))
        cfg = {'theta_outside_eps': 1e-5, 'topk_errors': 5, 'transition_abs_dI_fraction': 0.2, 'transition_window_points': 2}
        row, top, cyc = audit_prediction_file(pred_dir / 'fake_prediction.npz', root / 'soft', cfg)
        if row.get('phis_c_mae', 1.0) > 0.002 or row.get('theta_a_mae', 1.0) > 0.002:
            raise RuntimeError(f'selftest metrics unexpectedly large: {row}')
        if not top:
            raise RuntimeError('top-k audit did not produce rows')
        write_json({'row': row, 'top_count': len(top), 'cycle_count': len(cyc)}, root / 'selftest.json')
    print('[D15-P2 selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
