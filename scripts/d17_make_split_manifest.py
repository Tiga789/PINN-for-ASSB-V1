# -*- coding: utf-8 -*-
"""
D17-P1: create locked cell-level train/validation/frozen-test split manifest.

Patch v3 (2026-06-15):
- Robustly canonicalizes D15 ALL55 soft-label names such as Batch-1_battery-8
  to replay-compatible aliases such as Batch-1_2C_battery-8.
- Forces Batch-1/battery-8 to flagged_probe even when the soft-label directory
  omits the protocol token.
- Improves replay matching for mixed replay directory conventions:
  Batch-1_2C_battery-k, profiles/000x_battery-k_2C_battery-k, etc.
- Auto-discovers sibling replay-profile roots under _gv1_cache, so initial 8-cell
  replay profiles such as Batch-5_battery-7 / Batch-6_battery-3 are not missed
  when only D15-P4C remaining14 roots are passed.
- Prints exact missing replay IDs in the console when status=REVIEW.
- Does NOT read state soft-label arrays. Only directory names and small JSON
  summaries are inspected.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUMMARY_NAMES = ("soft_label_summary.json", "summary.json", "softlabel_summary.json")
SOFTLABEL_NPZ_NAMES = ("solution_softlabels.npz", "solution_softlabel.npz")
REPLAY_NPZ_HINTS = ("solution_replay_profile.npz", "replay_profile.npz", "solution.npz")

# D17-P1 canonical softlabel-dir -> replay-dir conventions.
# These names are used only for IDs/metadata matching; they are not state labels.
BATCH_PROTOCOL_MAP = {
    "Batch-1": "2C",
    "Batch-2": "3C",
    "Batch-3": "R2.5",
    "Batch-4": "R3",
    "Batch-5": "random_walk",
    "Batch-6": "GEO",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_optional(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def find_summary(cell_dir: Path) -> Optional[Path]:
    for name in SUMMARY_NAMES:
        p = cell_dir / name
        if p.exists():
            return p
    found = sorted(cell_dir.glob("*summary*.json"))
    return found[0] if found else None


def find_softlabel_npz(cell_dir: Path) -> Optional[Path]:
    for name in SOFTLABEL_NPZ_NAMES:
        p = cell_dir / name
        if p.exists():
            return p
    found = sorted(cell_dir.glob("*.npz"))
    soft = [p for p in found if "soft" in p.name.lower()]
    return soft[0] if soft else (found[0] if found else None)


def natural_sort_key(s: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


def norm_text(text: str) -> str:
    """Normalize text for robust path/name matching."""
    t = str(text).replace("\\", "/")
    t = t.split("/")[-1]
    t = re.sub(r"\s+", "_", t.strip())
    t = re.sub(r"Batch[_ ]?(\d+)", r"Batch-\1", t, flags=re.I)
    t = re.sub(r"battery[_ ]?(\d+)", r"battery-\1", t, flags=re.I)
    t = t.replace("R2_5", "R2.5")
    return t


def batch_battery_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = norm_text(text)
    mb = re.search(r"Batch[-_ ]?(\d+)", t, flags=re.I)
    mc = re.search(r"battery[-_ ]?(\d+)", t, flags=re.I)
    batch = f"Batch-{mb.group(1)}" if mb else None
    battery = f"battery-{mc.group(1)}" if mc else None
    return batch, battery


def canonical_cell_id(text: str, *, fallback_dir_name: str = "") -> str:
    """
    Canonicalize a cell/profile identifier.

    Important D15 ALL55 convention:
      softlabel directory: Batch-1_battery-8
      replay directory:    Batch-1_2C_battery-8 or 0008_battery-8_2C_battery-8

    We canonicalize the manifest to the replay-compatible alias when the batch
    has a known protocol. This also makes flag_cell matching deterministic.
    """
    candidates = [fallback_dir_name, text]
    # Prefer explicit D15 softlabel directory form because some summary JSONs
    # inherited old replay profile ids such as profiles\\0003_battery-3_2C_battery-3.
    for cand in candidates:
        t = norm_text(cand)
        batch, battery = batch_battery_from_text(t)
        if batch and battery:
            proto = BATCH_PROTOCOL_MAP.get(batch)
            if proto and proto not in t:
                return f"{batch}_{proto}_{battery}"
            return t
    return norm_text(text or fallback_dir_name)


def infer_batch_protocol_battery(cell_uid: str, summary: Dict[str, Any], *, fallback_dir_name: str = "") -> Tuple[str, str, str]:
    text = " ".join([str(fallback_dir_name), str(cell_uid)])
    batch = str(summary.get("batch") or summary.get("batch_id") or "")
    protocol = str(summary.get("protocol") or summary.get("protocol_id") or "")
    battery = str(summary.get("battery") or summary.get("battery_id") or "")

    b2, bat2 = batch_battery_from_text(text)
    if not batch and b2:
        batch = b2
    if not battery and bat2:
        battery = bat2
    if not batch:
        batch = "Batch-UNKNOWN"
    if not battery:
        battery = "battery-UNKNOWN"

    # D15 ALL55 softlabel dirs often omit protocol for B1/B3/B4 ready profiles.
    if not protocol or protocol == "protocol-UNKNOWN":
        mapped = BATCH_PROTOCOL_MAP.get(batch)
        if mapped:
            protocol = mapped
        else:
            normalized = text.lower().replace("_", ".")
            for tag in ("2C", "3C", "R2.5", "R3", "random_walk", "GEO"):
                if tag.lower().replace("_", ".") in normalized:
                    protocol = tag
                    break
    protocol = protocol or "protocol-UNKNOWN"
    return batch, protocol, battery


def should_prefer_dir_name(dir_name: str, summary_cell_uid: str) -> bool:
    """D15 final softlabel directory names are more stable than old summary profile ids."""
    d = norm_text(dir_name)
    if re.fullmatch(r"Batch-\d+_battery-\d+", d, flags=re.I):
        return True
    # Also prefer dir if summary uid contains a numeric replay profile id.
    s = norm_text(summary_cell_uid)
    if re.search(r"^\d+_battery-\d+_", s):
        return True
    return False


def discover_cell_dirs(softlabel_root: Path) -> List[Path]:
    if not softlabel_root.exists():
        raise FileNotFoundError(f"softlabel_root does not exist: {softlabel_root}")

    profiles_dir = softlabel_root / "profiles"
    search_roots = [profiles_dir] if profiles_dir.exists() else [softlabel_root]
    candidates: List[Path] = []
    for root in search_roots:
        # Prefer direct children under profiles; fall back to rglob only if needed.
        direct = [p for p in root.iterdir() if p.is_dir() and (find_softlabel_npz(p) or find_summary(p))]
        if direct:
            candidates.extend(direct)
        else:
            candidates.extend([p for p in root.rglob("*") if p.is_dir() and (find_softlabel_npz(p) or find_summary(p))])

    unique: List[Path] = []
    seen = set()
    for d in sorted(candidates):
        key = str(d.resolve()).lower()
        if key not in seen:
            unique.append(d)
            seen.add(key)
    return unique


def discover_softlabel_cells(softlabel_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for d in discover_cell_dirs(softlabel_root):
        summary_path = find_summary(d)
        summary = load_json_optional(summary_path)
        summary_uid = str(summary.get("cell_uid") or summary.get("profile_id") or summary.get("cell_id") or "")
        if should_prefer_dir_name(d.name, summary_uid):
            cell_uid = norm_text(d.name)
        else:
            cell_uid = norm_text(summary_uid or d.name)
        batch, protocol, battery = infer_batch_protocol_battery(cell_uid, summary, fallback_dir_name=d.name)
        canonical = canonical_cell_id(cell_uid, fallback_dir_name=d.name)
        npz_path = find_softlabel_npz(d)
        records.append({
            "cell_uid": cell_uid,
            "canonical_cell_uid": canonical,
            "batch": batch,
            "protocol": protocol,
            "battery": battery,
            "softlabel_dir": str(d),
            "softlabel_npz": str(npz_path) if npz_path else "",
            "softlabel_summary": str(summary_path) if summary_path else "",
            "source_stage": summary.get("source_stage") or summary.get("stage") or "",
            "resolved_spec_hash": summary.get("resolved_spec_hash") or "",
        })

    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        cid = rec["canonical_cell_uid"]
        if cid in by_id:
            duplicates.setdefault(cid, [by_id[cid]]).append(rec)
            # Prefer records with softlabel npz and a non-legacy cell_uid.
            cur_score = int(bool(by_id[cid].get("softlabel_npz"))) + int("profiles" not in by_id[cid].get("cell_uid", "").lower())
            new_score = int(bool(rec.get("softlabel_npz"))) + int("profiles" not in rec.get("cell_uid", "").lower())
            if new_score > cur_score:
                by_id[cid] = rec
        else:
            by_id[cid] = rec

    out = sorted(by_id.values(), key=lambda r: natural_sort_key(r["canonical_cell_uid"]))
    for rec in out:
        rec["duplicate_candidates"] = max(0, len(duplicates.get(rec["canonical_cell_uid"], [])))
    return out


def find_replay_profiles(replay_roots: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in replay_roots:
        if not root or not root.exists():
            continue
        for p in root.rglob("*.npz"):
            lname = p.name.lower()
            if any(h.lower() == lname for h in REPLAY_NPZ_HINTS) or "replay" in lname or "profile" in lname:
                paths.append(p)
    return sorted(set(paths), key=lambda p: str(p).lower())


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def discover_sibling_replay_roots(replay_roots: Iterable[Path], explicit_search_roots: Iterable[Path]) -> List[Path]:
    """
    Recover replay roots that were not explicitly passed on the command line.

    D15 ALL55 FINAL mixes cells generated in several stages. Two cells in the
    initial cross-batch 8-cell set (commonly Batch-5_battery-7 and Batch-6_battery-3)
    may live in an earlier replay-profile cache rather than the later
    xjtu_batch56_remaining14_replay_profiles_d15p4c root. This function searches
    only likely replay/profile cache directories and never opens state soft-label arrays.
    """
    bases: List[Path] = []
    for x in explicit_search_roots:
        if x and x.exists():
            bases.append(x)
    for root in replay_roots:
        if not root:
            continue
        # The normal case is E:/XJTU battery dataset/_gv1_cache/<replay_root>.
        # root.parent is therefore _gv1_cache and is the correct sibling-search base.
        if root.exists():
            bases.append(root)
            bases.append(root.parent)
            if root.name.lower() == "profiles":
                bases.append(root.parent.parent)
        else:
            # Even if one supplied root is missing, its parent may still be useful.
            bases.append(root.parent)
    bases = _unique_paths([b for b in bases if b and b.exists()])

    roots: List[Path] = []
    for base in bases:
        if not base.exists():
            continue
        name_l = base.name.lower()
        if ("replay" in name_l or "profile" in name_l) and base.is_dir():
            roots.append(base)
        # Scan only direct children. Avoid a full _gv1_cache rglob over the 52GB
        # soft-label tree unless the child itself looks like a replay/profile cache.
        try:
            children = list(base.iterdir())
        except Exception:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            cl = child.name.lower()
            if "replay" in cl or "profile" in cl:
                roots.append(child)
    return _unique_paths(roots)


def replay_key(path: Path) -> str:
    parts = list(path.parts[-5:])
    return " ".join(parts).lower().replace("\\", "/")


def aliases_for_record(rec: Dict[str, Any]) -> List[str]:
    aliases = set()
    for field in ("canonical_cell_uid", "cell_uid"):
        val = str(rec.get(field, ""))
        if val:
            aliases.add(norm_text(val))
            aliases.add(canonical_cell_id(val, fallback_dir_name=val))
    batch = rec.get("batch", "")
    protocol = rec.get("protocol", "")
    battery = rec.get("battery", "")
    if batch and battery:
        mapped = BATCH_PROTOCOL_MAP.get(str(batch), str(protocol) or "")
        if mapped and mapped != "protocol-UNKNOWN":
            aliases.add(f"{batch}_{mapped}_{battery}")
            aliases.add(f"{battery}_{mapped}_{battery}")
            aliases.add(f"{mapped}_{battery}")
        aliases.add(f"{batch}_{battery}")
    return sorted(a for a in aliases if a)


def attach_replay_paths(records: List[Dict[str, Any]], replay_roots: Iterable[Path]) -> None:
    replays = find_replay_profiles(replay_roots)
    replay_keys = [(p, replay_key(p)) for p in replays]
    for rec in records:
        aliases = [a.lower().replace("\\", "/") for a in aliases_for_record(rec)]
        batch_low = str(rec.get("batch", "")).lower()
        battery_low = str(rec.get("battery", "")).lower()
        protocol_low = str(rec.get("protocol", "")).lower().replace("_", ".")
        best = ""

        # 1) Exact/alias containment.
        for p, key in replay_keys:
            key2 = key.replace("_", ".")
            if any(alias.replace("_", ".") in key2 for alias in aliases):
                best = str(p)
                break

        # 2) Fallback: protocol + battery. Necessary for old profiles/000x_battery-k_2C_battery-k dirs.
        if not best and battery_low and protocol_low and protocol_low != "protocol-unknown":
            for p, key in replay_keys:
                key2 = key.replace("_", ".")
                if battery_low in key2 and protocol_low in key2:
                    best = str(p)
                    break

        # 3) Fallback: batch + battery for Batch-5/6 direct names.
        if not best and batch_low and battery_low:
            for p, key in replay_keys:
                key2 = key.replace("_", ".")
                if batch_low in key2 and battery_low in key2:
                    best = str(p)
                    break

        rec["replay_npz"] = best


def normalize_flag(flag: str) -> List[str]:
    flag = norm_text(flag)
    out = {flag, canonical_cell_id(flag, fallback_dir_name=flag)}
    b, bat = batch_battery_from_text(flag)
    if b and bat:
        proto = BATCH_PROTOCOL_MAP.get(b)
        out.add(f"{b}_{bat}")
        if proto:
            out.add(f"{b}_{proto}_{bat}")
            out.add(f"{proto}_{bat}")
    return sorted(out)


def is_flagged(rec: Dict[str, Any], flags: Iterable[str]) -> bool:
    rec_aliases = {a.lower() for a in aliases_for_record(rec)}
    flag_aliases = set()
    for flag in flags:
        flag_aliases.update(a.lower() for a in normalize_flag(flag))
    if rec_aliases.intersection(flag_aliases):
        return True
    # Explicit D17 rule: Batch-1 battery-8 is the late-2C flagged probe.
    return rec.get("batch") == "Batch-1" and rec.get("battery") == "battery-8"


def stratified_split(records: List[Dict[str, Any]], seed: int, train_frac: float, val_frac: float) -> None:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        if rec.get("split") == "flagged_probe":
            continue
        groups.setdefault((rec["batch"], rec["protocol"]), []).append(rec)

    rng = random.Random(seed)
    for key in sorted(groups.keys(), key=lambda k: (natural_sort_key(k[0]), natural_sort_key(k[1]))):
        group = sorted(groups[key], key=lambda r: natural_sort_key(r["canonical_cell_uid"]))
        rng.shuffle(group)
        n = len(group)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train = max(1, int(round(n * train_frac)))
            n_val = max(1, int(round(n * val_frac)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        for i, rec in enumerate(group):
            if i < n_train:
                rec["split"] = "train"
            elif i < n_train + n_val:
                rec["split"] = "validation"
            else:
                rec["split"] = "frozen_test"


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "split", "cell_uid", "canonical_cell_uid", "batch", "protocol", "battery",
        "is_flagged_probe", "replay_npz", "softlabel_dir", "softlabel_npz",
        "softlabel_summary", "resolved_spec_hash", "source_stage", "duplicate_candidates",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--softlabel_root", required=True)
    ap.add_argument("--replay_root", action="append", default=[], help="May be provided multiple times")
    ap.add_argument("--replay_search_root", action="append", default=[], help="Optional cache root(s) used to auto-discover sibling replay profile directories, e.g. E:/XJTU battery dataset/_gv1_cache")
    ap.add_argument("--no_auto_replay_sibling_search", action="store_true", help="Disable automatic sibling replay-root discovery under _gv1_cache")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--flag_cell", action="append", default=["Batch-1_2C_battery-8"])
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    softlabel_root = Path(ns.softlabel_root)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("d17_split_manifest.json", "d17_split_manifest.csv", "d17_split_audit.json"):
        p = out_dir / name
        if p.exists() and not ns.force:
            raise SystemExit(f"[ABORT] {p} already exists. Use --force only if you intentionally lock a new split.")

    records = discover_softlabel_cells(softlabel_root)
    explicit_replay_roots = [Path(x) for x in ns.replay_root]
    extra_replay_roots: List[Path] = []
    if not ns.no_auto_replay_sibling_search:
        extra_replay_roots = discover_sibling_replay_roots(
            explicit_replay_roots, [Path(x) for x in ns.replay_search_root]
        )
    effective_replay_roots = _unique_paths(explicit_replay_roots + extra_replay_roots)
    attach_replay_paths(records, effective_replay_roots)

    for rec in records:
        rec["is_flagged_probe"] = bool(is_flagged(rec, ns.flag_cell))
        if rec["is_flagged_probe"]:
            rec["split"] = "flagged_probe"
    stratified_split(records, ns.seed, ns.train_frac, ns.val_frac)

    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["canonical_cell_uid"]))

    stable_payload = {
        "seed": ns.seed,
        "softlabel_root": str(softlabel_root),
        "flag_cell": sorted(ns.flag_cell),
        "records": [
            {k: rec.get(k, "") for k in [
                "split", "canonical_cell_uid", "batch", "protocol", "battery",
                "is_flagged_probe", "softlabel_dir", "softlabel_npz", "replay_npz"
            ]}
            for rec in records_sorted
        ],
    }
    stable_json = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_hash = sha256_text(stable_json)

    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec["split"]] = counts.get(rec["split"], 0) + 1

    duplicate_ids = [r["canonical_cell_uid"] for r in records if int(r.get("duplicate_candidates") or 0) > 0]
    missing_replay = [
        r["canonical_cell_uid"] for r in records
        if not r.get("replay_npz") and r.get("split") in ("train", "validation", "frozen_test")
    ]
    flagged_cells = [r["canonical_cell_uid"] for r in records if r.get("is_flagged_probe")]
    battery8_flagged = any(
        r.get("batch") == "Batch-1" and r.get("battery") == "battery-8" and r.get("is_flagged_probe")
        for r in records
    )
    expected_has_train_validation_frozen = all(counts.get(k, 0) > 0 for k in ("train", "validation", "frozen_test"))

    notes = [
        "State soft-label arrays were not read while creating this manifest.",
        "softlabel_npz paths are included only for later frozen/report-only audit. D17 dataset must not use them for training.",
        "Patch v3 canonicalizes Batch-1/2/3/4/5/6 protocol aliases and auto-discovers sibling replay-profile roots under _gv1_cache.",
        f"effective_replay_root_count={len(effective_replay_roots)}",
    ]
    audit_pass = expected_has_train_validation_frozen and battery8_flagged and len(missing_replay) == 0
    if not battery8_flagged:
        notes.append("Batch-1 battery-8 was not flagged. Check canonical cell names and --flag_cell.")
    if missing_replay:
        notes.append("Some normal split records still have no replay_npz; check replay_root paths or replay naming.")

    audit = {
        "protocol": "D17-P1_SPLIT_AUDIT",
        "created_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "pass": audit_pass,
        "manifest_hash_sha256": manifest_hash,
        "split_manifest_locked": True,
        "counts": counts,
        "expected_has_train_validation_frozen": expected_has_train_validation_frozen,
        "flagged_cells": flagged_cells,
        "battery8_flagged": battery8_flagged,
        "duplicate_warning_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "missing_replay_count_for_normal_splits": len(missing_replay),
        "missing_replay_for_normal_splits": missing_replay[:100],
        "notes": notes,
    }

    manifest = {
        "protocol": "D17-P1_SPLIT_MANIFEST",
        "created_at_utc": audit["created_at_utc"],
        "seed": ns.seed,
        "softlabel_root": str(softlabel_root),
        "replay_roots": ns.replay_root,
        "auto_discovered_replay_roots": [str(p) for p in effective_replay_roots if str(p) not in set(ns.replay_root)],
        "flag_cell": ns.flag_cell,
        "split_policy": {
            "level": "cell",
            "stratify_by": ["batch", "protocol"],
            "train_frac": ns.train_frac,
            "val_frac": ns.val_frac,
            "frozen_test_frac": 1.0 - ns.train_frac - ns.val_frac,
            "flagged_probe_policy": "excluded_from_normal_promotion",
        },
        "manifest_hash_sha256": manifest_hash,
        "counts": counts,
        "records": records_sorted,
    }

    (out_dir / "d17_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "d17_split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "d17_split_manifest.csv", records_sorted)
    print(json.dumps({
        "status": "PASS" if audit_pass else "REVIEW",
        "out_dir": str(out_dir),
        "manifest_hash_sha256": manifest_hash,
        "counts": counts,
        "flagged_cells": flagged_cells,
        "missing_replay_count_for_normal_splits": len(missing_replay),
        "missing_replay_for_normal_splits": missing_replay[:50],
        "effective_replay_root_count": len(effective_replay_roots),
        "auto_discovered_replay_roots": [str(p) for p in effective_replay_roots if str(p) not in set(ns.replay_root)][:20],
    }, ensure_ascii=False, indent=2))
    return 0 if audit_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
