from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import build_dataset
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics


def make_fake_profile(root: Path, name: str, phase: float) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    n = 240
    nr = 17
    t = np.linspace(0, 1000, n).astype(np.float32)
    I = (2.0 * np.sin(2 * np.pi * t / 400 + phase)).astype(np.float32)
    q = np.cumsum(I * np.diff(t, prepend=t[0])) / 3600
    q = q / (np.max(np.abs(q)) + 1e-6)
    r = np.linspace(0.03, 0.97, nr).astype(np.float32)
    theta_a_mean = 0.45 + 0.15 * q
    theta_c_mean = 0.55 - 0.12 * q
    grad_a = 0.06 * np.tanh(I / 2.0)
    grad_c = -0.05 * np.tanh(I / 2.0)
    shape = (r - np.sum(r * ((np.linspace(0,1,nr+1)[1:]**3 - np.linspace(0,1,nr+1)[:-1]**3))))
    theta_a = theta_a_mean[:, None] + grad_a[:, None] * shape[None, :]
    theta_c = theta_c_mean[:, None] + grad_c[:, None] * shape[None, :]
    phie = (0.02 * I + 0.01 * np.sin(t / 100)).astype(np.float32)
    phis_c = (3.6 + 0.2 * q + 0.03 * I).astype(np.float32)
    np.savez_compressed(
        d / 'solution_softlabels.npz',
        t_global_s=t,
        I_profile=I,
        voltage_exp=phis_c + 0.005 * np.sin(t/50),
        temperature_C=np.full(n, 25.0, dtype=np.float32),
        theta_a=theta_a.astype(np.float32),
        theta_c=theta_c.astype(np.float32),
        cs_a=(theta_a * 31000).astype(np.float32),
        cs_c=(theta_c * 50000).astype(np.float32),
        phie=phie,
        phis_c=phis_c,
        r_a=r,
        r_c=r,
        radial_solver_version=np.array('P2Dlite-RG-v1-selftest'),
        cell_uid=np.array(name),
        batch=np.array('selftest'),
        protocol=np.array('selftest'),
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='d15p1_selftest_') as td:
        root = Path(td) / 'softlabels'
        make_fake_profile(root, 'p0', 0.0)
        make_fake_profile(root, 'p1', 0.7)
        bundle = build_dataset(root, max_train_per_profile=64, max_val_per_profile=32, seed=1)
        assert bundle.X_train.shape[0] > 0
        assert bundle.Y_train.shape[1] == 36
        # Identity-like metric smoke.
        m = compute_rg_metrics(bundle.Y_val, bundle.Y_val.copy(), bundle.target_slices)
        assert m['theta_a_mae'] < 1e-12
        assert m['phis_c_mae'] < 1e-12
    print('[D15-P1 selftest] PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
