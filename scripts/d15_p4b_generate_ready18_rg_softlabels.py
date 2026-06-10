from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar


def pwin(path: str | Path) -> Path:
    return Path(str(path))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def scalar_string(x: Any, fallback: str = '') -> str:
    try:
        arr = np.asarray(x)
        val = arr.item() if arr.shape == () else arr.reshape(-1)[0]
        if isinstance(val, bytes):
            return val.decode('utf-8', errors='ignore')
        return str(val)
    except Exception:
        return fallback


def _electrode_params(prior: Dict[str, Any], electrode: str) -> ElectrodeRGParams:
    rg = prior.get('radial_gradient', {})
    if electrode == 'a':
        spec = prior['electrodes']['negative']
        alpha_D = float(rg.get('alpha_D_negative', 1.0))
        alpha_J = float(rg.get('alpha_J_negative', 1.0))
        name = 'negative_graphite_d15p4b'
    else:
        spec = prior['electrodes']['positive']
        alpha_D = float(rg.get('alpha_D_positive', 1.0))
        alpha_J = float(rg.get('alpha_J_positive', 1.0))
        name = 'positive_NCM523_d15p4b'
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


def _get_1d(d: Dict[str, Any], keys: List[str], required: bool = True, fill: float = 0.0) -> np.ndarray:
    n = None
    for k in ['t_global_s', 'time_s', 't']:
        if k in d:
            n = np.asarray(d[k]).reshape(-1).size
            break
    for k in keys:
        if k in d:
            arr = np.asarray(d[k])
            if arr.dtype.kind in {'U', 'S', 'O'}:
                continue
            arr = arr.astype(float).reshape(-1)
            if n is None or arr.size == n:
                return arr
    if required:
        raise KeyError('Missing required keys: ' + ','.join(keys))
    if n is None:
        return np.array([], dtype=float)
    return np.full(n, fill, dtype=float)


def _cum_theta_from_current(t: np.ndarray, I: np.ndarray, theta0: float, window: float, capacity_Ah: float, sign: float) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt[dt < 0] = 0.0
    q_Ah = np.cumsum(I * dt) / 3600.0
    theta = theta0 + sign * (q_Ah / max(capacity_Ah, 1e-12)) * window
    return np.clip(theta, 0.0, 1.0)


def _copy_common_arrays(src: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keep = [
        't_global_s', 'time_s', 'I_profile', 'current_A', 'voltage_exp', 'voltage_V',
        'temperature_C', 'cycle_id', 'step_id', 'step_type', 'batch', 'protocol',
        'cell_uid', 'source_file', 'battery_id', 'Q_ref_Ah_replay'
    ]
    for k in keep:
        if k in src:
            out[k] = src[k]
    return out


def _process_one(row: Dict[str, str], cfg: Dict[str, Any], prior: Dict[str, Any], output_root_str: str, save_mode: str) -> Dict[str, Any]:
    npz_path = pwin(row['replay_npz'])
    profile_id = row.get('canonical_cell_id') or row.get('profile_id') or npz_path.parent.name
    with np.load(npz_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}

    t = _get_1d(d, ['t_global_s', 'time_s', 't'])
    I = _get_1d(d, ['I_profile', 'current_A', 'I'])
    V = _get_1d(d, ['voltage_exp', 'voltage_V', 'V'])
    T = _get_1d(d, ['temperature_C', 'T_C', 'T'], required=False, fill=float(cfg['generation'].get('default_temperature_C', 25.0)))

    gen = cfg['generation']
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

    out = _copy_common_arrays(d)
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
        'radial_solver_version': np.array('D15-P4B-ready18-P2Dlite-RG-v1'),
        'radial_gradient_quality_flag': np.array('D15P4B_READY18_GENERATED_NEEDS_AUDIT'),
        'source_profile_npz': np.array(str(npz_path)),
        'source_replay_profile_npz': np.array(str(npz_path)),
        'softlabel_boundary_note': np.array(cfg.get('boundary_note', 'D15-P4B model-consistent soft labels; not experimental internal-state truth.')),
    })
    out_dir = Path(output_root_str) / 'profiles' / profile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / 'solution_softlabels.npz'
    if save_mode == 'compressed':
        np.savez_compressed(out_npz, **out)
    else:
        np.savez(out_npz, **out)

    cyc_count = ''
    if 'cycle_id' in d:
        try:
            cyc_count = int(len(np.unique(np.asarray(d['cycle_id']).reshape(-1))))
        except Exception:
            cyc_count = ''
    summary = {
        'stage': 'D15-P4B ready18 P2Dlite-RG generation',
        'profile_id': profile_id,
        'canonical_cell_id': row.get('canonical_cell_id', profile_id),
        'source_replay_npz': str(npz_path),
        'output_npz': str(out_npz),
        'save_mode': save_mode,
        'time_points': int(len(t)),
        'cycle_count': cyc_count,
        'nr_a': nr,
        'nr_c': nr,
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
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='D15-P4B generate P2Dlite-RG soft labels for 18 replay-ready remaining cells.')
    ap.add_argument('--config', default='configs/d15_p4b_ready18_generation_config.json')
    ap.add_argument('--manifest-csv', default=None)
    ap.add_argument('--prior-json', default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--save-mode', choices=['compressed', 'uncompressed'], default=None)
    ap.add_argument('--allow-overwrite', action='store_true')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    manifest_csv = pwin(args.manifest_csv or cfg['p4a_fix_manifest_csv'])
    prior_json = Path(args.prior_json or cfg['prior_json'])
    output_root = pwin(args.output_dir or cfg['output_softlabels_dir'])
    workers = int(args.workers or cfg.get('generation', {}).get('workers', 2))
    workers = max(1, workers)
    save_mode = args.save_mode or cfg.get('generation', {}).get('save_mode', 'uncompressed')

    if output_root.exists() and any(output_root.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {output_root}. Use --allow-overwrite for deliberate rerun.')
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(manifest_csv)
    rows = [r for r in rows if str(r.get('p4b_ready', '')).lower() in ('true', '1', 'yes', 'pass')]
    prior = load_json(prior_json)
    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f'[D15-P4B generate] ready profile count: {len(rows)}; workers={workers}; save_mode={save_mode}', flush=True)
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            print(f'[D15-P4B generate] {idx}/{len(rows)} {row.get("canonical_cell_id")}', flush=True)
            try:
                summaries.append(_process_one(row, cfg, prior, str(output_root), save_mode))
            except Exception as exc:
                errors.append({'canonical_cell_id': row.get('canonical_cell_id', ''), 'replay_npz': row.get('replay_npz', ''), 'error': repr(exc), 'traceback': traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one, row, cfg, prior, str(output_root), save_mode): row for row in rows}
            done = 0
            for fut in as_completed(futs):
                done += 1
                row = futs[fut]
                can = row.get('canonical_cell_id', '')
                try:
                    summaries.append(fut.result())
                    print(f'[D15-P4B generate] done {done}/{len(rows)} {can}', flush=True)
                except Exception as exc:
                    errors.append({'canonical_cell_id': can, 'replay_npz': row.get('replay_npz', ''), 'error': repr(exc), 'traceback': traceback.format_exc()})
                    print(f'[D15-P4B generate] ERROR {done}/{len(rows)} {can}: {exc!r}', flush=True)

    summaries = sorted(summaries, key=lambda r: str(r.get('canonical_cell_id') or r.get('profile_id')))
    report = {
        'stage': 'D15-P4B ready18 P2Dlite-RG soft-label generation',
        'manifest_csv': str(manifest_csv),
        'prior_json': str(prior_json),
        'output_dir': str(output_root),
        'requested_profile_count': len(rows),
        'generated_count': len(summaries),
        'error_count': len(errors),
        'workers': workers,
        'save_mode': save_mode,
        'overall_status': 'PASS' if len(errors) == 0 and len(summaries) == len(rows) else 'FAIL',
        'errors': errors,
    }
    write_json(report, output_root / 'D15_P4B_READY18_RG_GENERATION_REPORT.json')
    write_csv(summaries, output_root / 'D15_P4B_READY18_RG_GENERATION_REPORT.csv')
    if errors:
        write_json(errors, output_root / 'D15_P4B_READY18_RG_GENERATION_ERRORS.json')
    print('[D15-P4B generate] overall_status:', report['overall_status'])
    return 0 if report['overall_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
