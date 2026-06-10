from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def clean_cell_id(s: str) -> str:
    keep = []
    for ch in str(s):
        keep.append(ch if ch.isalnum() or ch in ('-', '_', '.') else '_')
    return ''.join(keep).strip('_') or 'unknown_cell'


def _electrode_params(prior: Dict[str, Any], electrode: str) -> ElectrodeRGParams:
    rg = prior.get('radial_gradient', {})
    if electrode == 'a':
        spec = prior['electrodes']['negative']
        alpha_D = float(rg.get('alpha_D_negative', 1.0))
        alpha_J = float(rg.get('alpha_J_negative', 1.0))
        name = 'negative_graphite_d15p4d'
    else:
        spec = prior['electrodes']['positive']
        alpha_D = float(rg.get('alpha_D_positive', 1.0))
        alpha_J = float(rg.get('alpha_J_positive', 1.0))
        name = 'positive_NCM523_d15p4d'
    return ElectrodeRGParams(
        name=name,
        radius_m=float(spec['particle_radius_m']),
        diffusivity_m2_s=float(spec['solid_diffusivity_m2_s']),
        csmax_mol_m3=float(spec['csmax_mol_m3']),
        alpha_D=alpha_D,
        alpha_J=alpha_J,
        gradient_clip_normalized=float(spec.get('gradient_clip_normalized', rg.get('gradient_clip_normalized', 0.12))),
        theta_min_clip=0.0,
        theta_max_clip=1.0,
    )


def _infer_n(d: Dict[str, Any]) -> int:
    for k in ['t_global_s', 'time_s', 't']:
        if k in d:
            return int(np.asarray(d[k]).reshape(-1).size)
    raise KeyError('No time key found in replay profile')


def _get_1d(d: Dict[str, Any], keys: List[str], n: int | None = None, required: bool = True, fill: float = 0.0) -> np.ndarray:
    if n is None:
        n = _infer_n(d)
    for k in keys:
        if k in d:
            arr = np.asarray(d[k])
            if arr.dtype.kind in {'U', 'S', 'O'}:
                continue
            arr = arr.astype(float).reshape(-1)
            if arr.size >= n:
                return arr[:n]
    if required:
        raise KeyError('Missing required keys: ' + ','.join(keys))
    return np.full(n, fill, dtype=float)


def _copy_common_arrays(src: Dict[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keep = [
        't_global_s', 'time_s', 'I_profile', 'current_A', 'voltage_exp', 'voltage_V',
        'temperature_C', 'cycle_id', 'step_id', 'step_type', 'batch', 'protocol',
        'cell_uid', 'source_file', 'battery_id', 'Q_ref_Ah_replay'
    ]
    for k in keep:
        if k not in src:
            continue
        arr = np.asarray(src[k])
        if arr.ndim >= 1 and arr.shape[0] >= n:
            out[k] = arr[:n]
        else:
            out[k] = arr
    return out


def _cum_theta_from_current(t: np.ndarray, I: np.ndarray, theta0: float, window: float, capacity_Ah: float, sign: float) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt[dt < 0] = 0.0
    q_Ah = np.cumsum(I * dt) / 3600.0
    theta = theta0 + sign * (q_Ah / max(capacity_Ah, 1e-12)) * window
    return np.clip(theta, 0.0, 1.0)


def _find_row(manifest_csv: str | Path, cell_id: str) -> Dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    rows = [r for r in rows if str(r.get('status', '')).upper() in ('PASS', 'READY', 'READY_RAW', '')]
    matches = []
    for r in rows:
        can = r.get('canonical_cell_id') or r.get('cell_id') or r.get('profile_id') or ''
        if can == cell_id:
            matches.append(r)
    if len(matches) == 1:
        return matches[0]
    # fallback substring match, but require unique
    matches = [r for r in rows if cell_id in ' '.join([r.get('canonical_cell_id',''), r.get('cell_id',''), r.get('profile_id',''), r.get('npz_path',''), r.get('replay_npz','')])]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f'Could not find unique cell {cell_id} in {manifest_csv}; matches={len(matches)}')


def _extract_npz_path(row: Dict[str, str]) -> Path:
    for k in ['npz_path', 'replay_npz', 'profile_npz', 'output_npz']:
        v = row.get(k, '')
        if v:
            return Path(v)
    raise KeyError('Manifest row has no replay npz path column: expected npz_path/replay_npz/profile_npz/output_npz')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='D15-P4D full: generate one Batch-5/6 remaining-cell P2Dlite-RG soft-label profile.')
    ap.add_argument('--config', default='configs/d15_p4d_full_remaining14_config.json')
    ap.add_argument('--manifest-csv', required=True)
    ap.add_argument('--cell-id', required=True)
    ap.add_argument('--prior-json', default=None)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--save-mode', choices=['compressed', 'uncompressed'], default=None)
    ap.add_argument('--status-dir', required=True)
    ap.add_argument('--skip-if-complete', action='store_true')
    ap.add_argument('--overwrite-existing', action='store_true')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    gen = cfg['generation']
    prior = load_json(args.prior_json or cfg['prior_json'])
    row = _find_row(args.manifest_csv, args.cell_id)
    npz_path = _extract_npz_path(row)
    if not npz_path.exists():
        raise FileNotFoundError(f'Replay npz not found for {args.cell_id}: {npz_path}')
    save_mode = args.save_mode or gen.get('save_mode', 'uncompressed')
    out_dir = Path(args.output_root) / 'profiles' / args.cell_id
    out_npz = out_dir / 'solution_softlabels.npz'
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / (clean_cell_id(args.cell_id) + '.json')

    if args.skip_if_complete and out_npz.exists() and (out_dir / 'soft_label_summary.json').exists() and not args.overwrite_existing:
        try:
            prev = load_json(out_dir / 'soft_label_summary.json')
        except Exception:
            prev = {}
        prev.update({
            'standalone_process_status': 'SKIPPED_ALREADY_COMPLETE',
            'cell_id_requested': args.cell_id,
            'output_npz': str(out_npz),
            'status': 'PASS',
        })
        write_json(prev, status_path)
        print(f'[D15-P4D one] SKIP already complete {args.cell_id}', flush=True)
        return 0

    if args.overwrite_existing and out_npz.exists():
        try:
            out_npz.unlink()
        except Exception:
            pass

    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        n = _infer_n(d)
        t = _get_1d(d, ['t_global_s', 'time_s', 't'], n=n)
        I = _get_1d(d, ['I_profile', 'current_A', 'I'], n=n)
        V = _get_1d(d, ['voltage_exp', 'voltage_V', 'V'], n=n)
        T = _get_1d(d, ['temperature_C', 'T_C', 'T'], n=n, required=False, fill=float(gen.get('default_temperature_C', 25.0)))
        common = _copy_common_arrays(d, n)
        load_wall = time.perf_counter() - wall0

        compute_wall0 = time.perf_counter(); compute_cpu0 = time.process_time()
        cap_Ah = float(gen.get('capacity_scale_Ah', 2.0))
        p_a = _electrode_params(prior, 'a')
        p_c = _electrode_params(prior, 'c')
        pos_spec = prior['electrodes']['positive']
        neg_spec = prior['electrodes']['negative']
        win_c = float(pos_spec.get('theta_max', 0.9149)) - float(pos_spec.get('theta_min', 0.2535))
        win_a = float(neg_spec.get('theta_max', 0.8544)) - float(neg_spec.get('theta_min', 0.0079))
        theta_c_mean = _cum_theta_from_current(t, I, float(gen.get('theta_positive_initial', 0.90)), win_c, cap_Ah, sign=-1.0)
        theta_a_mean = _cum_theta_from_current(t, I, float(gen.get('theta_negative_initial', 0.08)), win_a, cap_Ah, sign=+1.0)
        cbar_c = theta_c_mean * p_c.csmax_mol_m3
        cbar_a = theta_a_mean * p_a.csmax_mol_m3
        J_c = infer_surface_flux_from_cbar(t, cbar_c, p_c.R)
        J_a = infer_surface_flux_from_cbar(t, cbar_a, p_a.R)
        nr = int(gen.get('n_r', 17))
        max_sub = float(prior.get('radial_gradient', {}).get('implicit_step_subdivide_dt_s', 10.0))
        cs_a, diag_a = generate_rg_profile(t, cbar_a, J_a, np.full(nr, cbar_a[0], dtype=float), p_a, nr=nr, max_substep_s=max_sub)
        cs_c, diag_c = generate_rg_profile(t, cbar_c, J_c, np.full(nr, cbar_c[0], dtype=float), p_c, nr=nr, max_substep_s=max_sub)
        phis_c_soft = V.astype(float)
        phie = float(gen.get('phie_ohmic_scale_V_per_A', -0.015)) * I.astype(float)
        compute_wall = time.perf_counter() - compute_wall0
        compute_cpu = time.process_time() - compute_cpu0

        out = common
        out.update({
            't_global_s': t.astype(np.float32),
            'I_profile': I.astype(np.float32),
            'voltage_exp': V.astype(np.float32),
            'temperature_C': T.astype(np.float32),
            'cs_a': cs_a.astype(np.float32),
            'cs_c': cs_c.astype(np.float32),
            'theta_a': (cs_a / p_a.csmax_mol_m3).astype(np.float32),
            'theta_c': (cs_c / p_c.csmax_mol_m3).astype(np.float32),
            'cbar_a': cbar_a.astype(np.float32),
            'cbar_c': cbar_c.astype(np.float32),
            'cs_a_surface': cs_a[:, -1].astype(np.float32),
            'cs_c_surface': cs_c[:, -1].astype(np.float32),
            'cs_a_center': cs_a[:, 0].astype(np.float32),
            'cs_c_center': cs_c[:, 0].astype(np.float32),
            'grad_a_surface_center': diag_a['surface_center'].astype(np.float32),
            'grad_c_surface_center': diag_c['surface_center'].astype(np.float32),
            'grad_a_surface_mean': diag_a['surface_mean'].astype(np.float32),
            'grad_c_surface_mean': diag_c['surface_mean'].astype(np.float32),
            'J_a_eff_rg': diag_a['J_used'].astype(np.float32),
            'J_c_eff_rg': diag_c['J_used'].astype(np.float32),
            'D_a_eff_rg': diag_a['D_eff'].astype(np.float32),
            'D_c_eff_rg': diag_c['D_eff'].astype(np.float32),
            'r_a': diag_a['r_norm_centers'].astype(np.float32),
            'r_c': diag_c['r_norm_centers'].astype(np.float32),
            'radial_volume_weights_a': diag_a['volume_weights'].astype(np.float32),
            'radial_volume_weights_c': diag_c['volume_weights'].astype(np.float32),
            'phis_c': phis_c_soft.astype(np.float32),
            'phis_c_soft': phis_c_soft.astype(np.float32),
            'phie': phie.astype(np.float32),
            'phis_c_base': phis_c_soft.astype(np.float32),
            'voltage_residual_s1k': np.zeros_like(phis_c_soft, dtype=np.float32),
            'radial_solver_version': np.array('D15-P4D-full-remaining14-P2Dlite-RG-v1'),
            'radial_gradient_quality_flag': np.array('D15P4D_FULL_GENERATED_NEEDS_AUDIT'),
            'source_profile_npz': np.array(str(npz_path)),
            'source_replay_profile_npz': np.array(str(npz_path)),
            'softlabel_boundary_note': np.array('D15-P4D model-consistent soft labels; not experimental internal-state truth.'),
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        write_wall0 = time.perf_counter()
        if save_mode == 'compressed':
            np.savez_compressed(out_npz, **out)
        else:
            np.savez(out_npz, **out)
        write_wall = time.perf_counter() - write_wall0
        wall_total = time.perf_counter() - wall0
        cpu_total = time.process_time() - cpu0
        cyc_count = ''
        if 'cycle_id' in d:
            try:
                cyc_count = int(len(np.unique(np.asarray(d['cycle_id']).reshape(-1))))
            except Exception:
                cyc_count = ''
        summary = {
            'stage': 'D15-P4D full remaining14 P2Dlite-RG generation',
            'standalone_process_status': 'PASS',
            'status': 'PASS',
            'profile_id': args.cell_id,
            'canonical_cell_id': args.cell_id,
            'source_replay_npz': str(npz_path),
            'output_npz': str(out_npz),
            'save_mode': save_mode,
            'time_points': int(n),
            'cycle_count': cyc_count,
            'nr_a': nr,
            'nr_c': nr,
            'load_wall_seconds': float(load_wall),
            'compute_wall_seconds': float(compute_wall),
            'compute_cpu_seconds': float(compute_cpu),
            'write_wall_seconds': float(write_wall),
            'total_wall_seconds': float(wall_total),
            'total_cpu_seconds': float(cpu_total),
            'cpu_core_equivalent_during_process': float(cpu_total / max(wall_total, 1e-9)),
            'output_size_mb': float(out_npz.stat().st_size / (1024 * 1024)),
            'phis_c_source': 'voltage_exp_preserved_as_phis_c_soft',
            'phie_source': f'lumped_current_ohmic_scale_{gen.get("phie_ohmic_scale_V_per_A", -0.015)}',
            'theta_a_min': float(np.nanmin(out['theta_a'])),
            'theta_a_max': float(np.nanmax(out['theta_a'])),
            'theta_c_min': float(np.nanmin(out['theta_c'])),
            'theta_c_max': float(np.nanmax(out['theta_c'])),
            'mean_abs_grad_a_norm': float(np.mean(np.abs(out['grad_a_surface_center'])) / p_a.csmax_mol_m3),
            'mean_abs_grad_c_norm': float(np.mean(np.abs(out['grad_c_surface_center'])) / p_c.csmax_mol_m3),
        }
        write_json(summary, out_dir / 'soft_label_summary.json')
        write_json(summary, status_path)
        print(f'[D15-P4D one] PASS {args.cell_id} wall={wall_total:.1f}s sizeMB={summary["output_size_mb"]:.1f}', flush=True)
        return 0
    except Exception as exc:
        wall_total = time.perf_counter() - wall0
        cpu_total = time.process_time() - cpu0
        err = {
            'stage': 'D15-P4D full remaining14 P2Dlite-RG generation',
            'standalone_process_status': 'FAIL',
            'status': 'FAIL',
            'profile_id': args.cell_id,
            'canonical_cell_id': args.cell_id,
            'source_replay_npz': str(npz_path),
            'error': repr(exc),
            'traceback': traceback.format_exc(),
            'total_wall_seconds': float(wall_total),
            'total_cpu_seconds': float(cpu_total),
        }
        write_json(err, status_path)
        print(f'[D15-P4D one] FAIL {args.cell_id}: {exc!r}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
