#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

def pfc(v: str) -> Path:
    return Path(str(v).replace('/', os.sep))

def ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def wjson(p: Path, o: Any) -> None:
    ensure(p.parent)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding='utf-8')

def wcsv(p: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    ensure(p.parent)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, '')
                out[k] = ';'.join(map(str, v)) if isinstance(v, (list, tuple)) else v
            writer.writerow(out)

def parse_batch(text: str) -> Optional[int]:
    m = re.findall(r'Batch[-_ ]?(?P<b>[1-6])(?:\D|$)', str(text))
    return int(m[-1]) if m else None

def parse_batt(text: str) -> Optional[int]:
    m = re.findall(r'battery[-_ ]?(?P<n>\d+)', str(text), flags=re.IGNORECASE)
    return int(m[-1]) if m else None

def infer_proto_batch(text: str) -> Optional[int]:
    for tok, b in [('R2.5', 3), ('R2_5', 3), ('R3', 4), ('3C', 2), ('2C', 1), ('random_walk', 5), ('random-walk', 5), ('GEO', 6), ('geo', 6)]:
        if tok in str(text):
            return b
    return None

def canonical_text(text: str) -> Optional[str]:
    t = str(text).replace('\\', '/')
    b = parse_batch(t)
    n = parse_batt(t)
    if b is None:
        b = infer_proto_batch(t)
    return f'Batch-{b}_battery-{n}' if b is not None and n is not None else None

def protocol(can: str) -> str:
    m = re.match(r'Batch-(\d+)_', str(can))
    b = int(m.group(1)) if m else 0
    return {1: '2C', 2: '3C', 3: 'R2.5', 4: 'R3', 5: 'random_walk', 6: 'GEO'}.get(b, '')

def scalar(z: Any, key: str) -> Optional[str]:
    if key not in z.files:
        return None
    x = z[key]
    val = x.item() if getattr(x, 'shape', None) == () else np.asarray(x).reshape(-1)[0]
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='ignore')
    return str(val)

def npz_can(path: Path):
    meta = {'batch': None, 'protocol': None, 'cell_uid': None, 'source_file': None}
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in meta:
                meta[k] = scalar(z, k)
    except Exception:
        return canonical_text(str(path)), None, meta
    for txt in [meta.get('cell_uid') or '', meta.get('source_file') or '', str(path)]:
        can = canonical_text(txt)
        if can:
            return can, protocol(can), meta
    b = parse_batch(meta.get('batch') or '')
    n = parse_batt(str(path))
    if b and n:
        can = f'Batch-{b}_battery-{n}'
        return can, protocol(can), meta
    return None, None, meta

def safe_keys(path: Path):
    try:
        with np.load(path, allow_pickle=True) as z:
            return True, list(z.files), ''
    except Exception as e:
        return False, [], repr(e)

def plen(path: Path):
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in ['t_global_s', 't']:
                if k in z.files:
                    return int(np.asarray(z[k]).shape[0])
    except Exception:
        return ''
    return ''

def stats(path: Path, ok: bool):
    if not ok:
        return {}
    out = {}
    try:
        with np.load(path, allow_pickle=True) as z:
            t = np.asarray(z['t_global_s'] if 't_global_s' in z.files else z['t'])
            i = np.asarray(z['I_profile'] if 'I_profile' in z.files else z['current_A'])
            v = np.asarray(z['voltage_exp'] if 'voltage_exp' in z.files else z['voltage_V'])
            cyc = np.asarray(z['cycle_id']) if 'cycle_id' in z.files else None
            out = dict(
                time_points=int(t.shape[0]),
                current_min_A=float(np.nanmin(i)),
                current_max_A=float(np.nanmax(i)),
                voltage_min_V=float(np.nanmin(v)),
                voltage_max_V=float(np.nanmax(v)),
                time_monotonic_nondec=bool(np.all(np.diff(t.astype(float)) >= -1e-9)) if t.size > 1 else True,
                finite_core_ok=bool(np.isfinite(t).all() and np.isfinite(i).all() and np.isfinite(v).all()),
                cycle_count=int(len(np.unique(cyc))) if cyc is not None and cyc.size else ''
            )
    except Exception as e:
        out = dict(error=repr(e), time_monotonic_nondec=False, finite_core_ok=False)
    return out

def discover(roots, fname: str):
    out = []
    seen = set()
    for r in roots:
        root = pfc(r)
        if not root.exists():
            continue
        for p in root.rglob(fname):
            s = str(p.resolve()).lower()
            if s not in seen:
                seen.add(s)
                out.append(p)
    return sorted(out, key=lambda p: str(p))

def infer_root(path: Path, roots):
    sp = str(path).replace('\\', '/').lower()
    best = ''
    for r in roots:
        rr = str(pfc(r)).replace('\\', '/').lower()
        if sp.startswith(rr) and len(rr) > len(best):
            best = rr
    return best

def dedup(rows: List[Dict[str, Any]], key: str):
    def score(r):
        s = 100 if r.get(key) is True or r.get(key) == 'PASS' else 0
        if re.match(r'^Batch-[1-6]_battery-\d+$', str(r.get('canonical_cell_id', ''))):
            s += 50
        try:
            n = int(r.get('time_points') or r.get('softlabel_time_points') or 0)
        except Exception:
            n = 0
        smoke = -1 if 'smoke' in str(r.get('replay_npz') or r.get('softlabel_npz') or '').lower() else 0
        return (s, n, smoke)
    best = {}
    nocan = []
    for r in rows:
        can = str(r.get('canonical_cell_id') or '')
        if not can:
            nocan.append(r)
            continue
        if can not in best or score(r) > score(best[can]):
            best[can] = r
    return sorted(list(best.values()) + nocan, key=lambda r: (str(r.get('canonical_cell_id') or 'ZZZ'), str(r.get('replay_npz') or r.get('softlabel_npz') or '')))

def raw_index(cfg):
    root = pfc(cfg['xjtu_root'])
    rows = []
    names = {1: '2C_battery-{n}.mat', 2: '3C_battery-{n}.mat', 3: 'R2.5_battery-{n}.mat', 4: 'R3_battery-{n}.mat'}
    for bs, spec in sorted(cfg['expected_batches'].items(), key=lambda kv: int(kv[0])):
        b = int(bs)
        proto = spec['protocol']
        for n in range(1, int(spec['cell_count']) + 1):
            d = root / f'Batch-{b}'
            p = d / (names.get(b, 'battery-{n}.mat').format(n=n))
            if not p.exists() and d.exists():
                hits = sorted(d.glob(f'*battery-{n}.mat'))
                if hits:
                    p = hits[0]
            rows.append(dict(canonical_cell_id=f'Batch-{b}_battery-{n}', batch_id=b, battery_id=n, protocol_inferred=proto, raw_mat_path=str(p), raw_mat_exists=p.exists(), raw_mat_size_bytes=p.stat().st_size if p.exists() else ''))
    return rows

def audit_soft(cfg):
    req = cfg.get('required_softlabel_keys', [])
    rows = []
    for p in discover(cfg.get('existing_rg_softlabel_roots', []), 'solution_softlabels.npz'):
        ok, keys, err = safe_keys(p)
        can, proto, meta = npz_can(p)
        miss = [k for k in req if k not in keys]
        rows.append(dict(canonical_cell_id=can or '', batch_id=int(re.match(r'Batch-(\d+)_', can).group(1)) if can and re.match(r'Batch-(\d+)_', can) else '', battery_id=int(re.search(r'battery-(\d+)', can).group(1)) if can and re.search(r'battery-(\d+)', can) else '', protocol_inferred=proto or (protocol(can) if can else ''), softlabel_npz=str(p), softlabel_root=infer_root(p, cfg.get('existing_rg_softlabel_roots', [])), softlabel_size_bytes=p.stat().st_size if p.exists() else '', softlabel_read_ok=ok, softlabel_keys_ok=ok and not miss, missing_required_keys=';'.join(miss), softlabel_time_points=plen(p), cell_uid_meta=meta.get('cell_uid') if meta else '', batch_meta=meta.get('batch') if meta else '', protocol_meta=meta.get('protocol') if meta else '', softlabel_error=err or ''))
    return dedup(rows, 'softlabel_keys_ok')

def audit_replay(cfg):
    req = cfg.get('required_replay_keys', [])
    rows = []
    for p in discover(cfg.get('replay_profile_roots', []), 'solution_replay_profile.npz'):
        ok, keys, err = safe_keys(p)
        can, proto, meta = npz_can(p)
        miss = [k for k in req if k not in keys]
        st = stats(p, ok)
        malformed = (not can) or ('Batch-134' in str(can)) or ('Batch-56' in str(can))
        rows.append(dict(canonical_cell_id=can or '', batch_id=int(re.match(r'Batch-(\d+)_', can).group(1)) if can and re.match(r'Batch-(\d+)_', can) else '', battery_id=int(re.search(r'battery-(\d+)', can).group(1)) if can and re.search(r'battery-(\d+)', can) else '', protocol_inferred=proto or (protocol(can) if can else ''), replay_npz=str(p), replay_root=infer_root(p, cfg.get('replay_profile_roots', [])), replay_size_bytes=p.stat().st_size if p.exists() else '', replay_read_ok=ok, required_keys_ok=ok and not miss, missing_required_keys=';'.join(miss), canonical_malformed=malformed, cell_uid_meta=meta.get('cell_uid') if meta else '', batch_meta=meta.get('batch') if meta else '', protocol_meta=meta.get('protocol') if meta else '', error=err or st.get('error', ''), status='PASS' if ok and not miss and not malformed and st.get('time_monotonic_nondec', True) and st.get('finite_core_ok', True) else 'FAIL', **st))
    return rows, dedup(rows, 'status')

def make_cov(raw, soft, rep):
    sb = {r['canonical_cell_id']: r for r in soft if r.get('canonical_cell_id')}
    rb = {r['canonical_cell_id']: r for r in rep if r.get('canonical_cell_id')}
    cov = []
    rem = []
    ready = []
    miss = []
    for r in raw:
        can = r['canonical_cell_id']
        s = sb.get(can)
        rp = rb.get(can)
        has = bool(s and s.get('softlabel_keys_ok') is True)
        p4b = (not has) and rp is not None and rp.get('status') == 'PASS'
        row = dict(r, has_existing_rg_softlabel=has, existing_softlabel_npz=s.get('softlabel_npz') if s else '', replay_status=rp.get('status') if rp else 'MISSING', replay_npz=rp.get('replay_npz') if rp else '', p4b_ready=p4b)
        cov.append(row)
        if not has:
            rem.append(row)
            (ready if p4b else miss).append(row)
    return cov, rem, ready, miss

def batch_matrix(cov):
    out = []
    for b in range(1, 7):
        rows = [r for r in cov if int(r['batch_id']) == b]
        rem = [r for r in rows if not r.get('has_existing_rg_softlabel')]
        ready = [r for r in rem if r.get('p4b_ready')]
        miss = [r for r in rem if not r.get('p4b_ready')]
        out.append(dict(batch_id=b, protocol=rows[0].get('protocol_inferred') if rows else '', raw_cell_count=len(rows), existing_rg_softlabel_cell_count=sum(1 for r in rows if r.get('has_existing_rg_softlabel')), remaining_cell_count=len(rem), remaining_p4b_ready_count=len(ready), remaining_missing_or_bad_replay_count=len(miss), remaining_cells=';'.join(r['canonical_cell_id'] for r in rem), p4b_ready_cells=';'.join(r['canonical_cell_id'] for r in ready), missing_or_bad_replay_cells=';'.join(r['canonical_cell_id'] for r in miss)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d15_p4a_fix_replay_readiness_config.json')
    ap.add_argument('--out-dir')
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()
    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    out = Path(args.out_dir) if args.out_dir else pfc(cfg['output_dir'])
    if out.exists() and any(out.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out}. Use --allow-overwrite for deliberate rerun.')
    out.mkdir(parents=True, exist_ok=True)
    raw = raw_index(cfg)
    soft = audit_soft(cfg)
    rep_all, rep = audit_replay(cfg)
    cov, rem, p4b, missing = make_cov(raw, soft, rep)
    mat = batch_matrix(cov)
    bad = [r for r in rep if not re.match(r'^Batch-[1-6]_battery-\d+$', str(r.get('canonical_cell_id') or ''))]
    fake = [r for r in rep if 'Batch-134' in str(r.get('canonical_cell_id')) or 'Batch-56' in str(r.get('canonical_cell_id'))]
    raw_count = len(raw)
    existing = sum(1 for r in cov if r.get('has_existing_rg_softlabel'))
    remaining = len(rem)
    ready = len(p4b)
    miss = len(missing)
    mapping = 'PASS' if not bad and not fake else 'FAIL'
    softstatus = 'PASS' if raw_count == 55 and existing == 23 and remaining == 32 else 'REVIEW'
    readiness = 'PASS' if ready > 0 else 'REVIEW'
    full = 'PASS' if ready == remaining == 32 else 'REVIEW'
    final = 'PASS' if mapping == 'PASS' and softstatus == 'PASS' and readiness == 'PASS' else 'REVIEW'
    score = dict(stage='D15-P4A-fix', final_status=final, mapping_status=mapping, softlabel_coverage_status=softstatus, partial_p4b_readiness_status=readiness, full_remaining32_readiness_status=full, raw_cell_count=raw_count, existing_rg_softlabel_cell_count=existing, remaining_cell_count=remaining, p4b_ready_remaining_cell_count=ready, missing_or_bad_replay_remaining_cell_count=miss, replay_profile_dedup_count=len(rep), replay_profile_all_count=len(rep_all), bad_canonical_replay_count=len(bad), fake_batch_replay_count=len(fake), p4b_ready_cells=[r['canonical_cell_id'] for r in p4b], missing_or_bad_replay_cells=[r['canonical_cell_id'] for r in missing], important_note='PASS means canonicalization/readiness audit is fixed and a non-empty P4B manifest is available. If full_remaining32_readiness_status is REVIEW, missing replay profiles must be generated before completing all 32 cells.', created_at_unix=time.time())
    wjson(out / 'D15_P4A_FIX_FINAL_SCORECARD.json', score)
    wcsv(out / 'D15_P4A_FIX_RAW_CELL_INDEX.csv', raw)
    wcsv(out / 'D15_P4A_FIX_EXISTING_RG_SOFTLABEL_INDEX.csv', soft)
    wcsv(out / 'D15_P4A_FIX_REPLAY_PROFILE_AUDIT_ALL.csv', rep_all)
    wcsv(out / 'D15_P4A_FIX_REPLAY_PROFILE_AUDIT_DEDUP.csv', rep)
    wcsv(out / 'D15_P4A_FIX_EXISTING_CELL_COVERAGE.csv', cov)
    wcsv(out / 'D15_P4A_FIX_REMAINING32_CELL_MANIFEST.csv', rem)
    wcsv(out / 'D15_P4A_FIX_P4B_INPUT_MANIFEST.csv', p4b)
    wcsv(out / 'D15_P4A_FIX_MISSING_OR_BAD_REPLAY_MANIFEST.csv', missing)
    wcsv(out / 'D15_P4A_FIX_BATCH_COVERAGE_MATRIX.csv', mat)
    print('[D15-P4A-fix] final_status:', final)
    print('[D15-P4A-fix] mapping_status:', mapping)
    print('[D15-P4A-fix] existing_rg_softlabel_cell_count:', existing)
    print('[D15-P4A-fix] remaining_cell_count:', remaining)
    print('[D15-P4A-fix] p4b_ready_remaining_cell_count:', ready)
    print('[D15-P4A-fix] missing_or_bad_replay_remaining_cell_count:', miss)
    print('[D15-P4A-fix] wrote:', out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
