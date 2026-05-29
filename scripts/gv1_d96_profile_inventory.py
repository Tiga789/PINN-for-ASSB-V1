#!/usr/bin/env python
"""D9.6 profile inventory and deterministic selection for GV1 multi-cell checks.

This script does not touch training data. It scans D8 replay-profile folders and
selects profile-level runs for sequential single-profile verification.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ProfileItem:
    run_id: str
    protocol: str
    batch_hint: str
    cell_hint: str
    profile_dir: str
    solution_npz: str


def _protocol_from_text(text: str) -> str:
    s = text.replace("\\", "/")
    # Keep R2.5 before R2/R3 style checks.
    if re.search(r"(?:^|[_\-/])R2\.5(?:[_\-/]|$)", s, flags=re.I):
        return "R2.5"
    if re.search(r"(?:^|[_\-/])R3(?:[_\-/]|$)", s, flags=re.I):
        return "R3"
    if re.search(r"(?:^|[_\-/])R2(?:[_\-/]|$)", s, flags=re.I):
        return "R2"
    if re.search(r"(?:^|[_\-/])2C(?:[_\-/]|$)", s, flags=re.I):
        return "2C"
    if re.search(r"(?:^|[_\-/])3C(?:[_\-/]|$)", s, flags=re.I):
        return "3C"
    return "unknown"


def _batch_hint_from_text(text: str, protocol: str) -> str:
    # Historical naming convention in this project: 2C≈Batch-1, R2.5≈Batch-3, R3≈Batch-4.
    if protocol == "2C":
        return "B1"
    if protocol == "R2.5":
        return "B3"
    if protocol == "R3":
        return "B4"
    m = re.search(r"(?:Batch[-_ ]?)(\d+)", text, flags=re.I)
    return f"B{m.group(1)}" if m else "B?"


def _cell_hint_from_text(text: str) -> str:
    name = Path(text).parent.name
    m = re.search(r"battery[-_ ]?(\d+)", name, flags=re.I)
    if m:
        return f"battery-{m.group(1)}"
    m = re.search(r"cell[-_ ]?(\d+)", name, flags=re.I)
    if m:
        return f"cell-{m.group(1)}"
    return name


def _safe_id(text: str) -> str:
    text = text.replace("R2.5", "R25")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:140] if len(text) > 140 else text


def scan_profiles(profile_root: Path) -> list[ProfileItem]:
    paths = sorted(profile_root.rglob("solution_replay_profile.npz"), key=lambda p: str(p).lower())
    items: list[ProfileItem] = []
    seen_run_ids: dict[str, int] = {}
    for idx, p in enumerate(paths, start=1):
        text = str(p)
        protocol = _protocol_from_text(text)
        batch = _batch_hint_from_text(text, protocol)
        cell = _cell_hint_from_text(text)
        base = _safe_id(f"{batch}_{protocol}_{cell}_{p.parent.name}")
        if base in seen_run_ids:
            seen_run_ids[base] += 1
            run_id = f"{base}_{seen_run_ids[base]:02d}"
        else:
            seen_run_ids[base] = 1
            run_id = base
        items.append(ProfileItem(
            run_id=run_id,
            protocol=protocol,
            batch_hint=batch,
            cell_hint=cell,
            profile_dir=str(p.parent),
            solution_npz=str(p),
        ))
    return items


def parse_quota(text: str) -> dict[str, int] | None:
    if not text or text.strip().lower() in {"all", "*"}:
        return None
    out: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad quota item {part!r}; expected e.g. 2C:2,R2.5:2,R3:2")
        k, v = part.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


def select_profiles(items: list[ProfileItem], quota: dict[str, int] | None, max_profiles: int | None) -> list[ProfileItem]:
    if quota is None:
        selected = list(items)
        return selected[:max_profiles] if max_profiles and max_profiles > 0 else selected

    selected: list[ProfileItem] = []
    used: set[str] = set()
    for protocol, n in quota.items():
        matches = [it for it in items if it.protocol.lower() == protocol.lower()]
        selected.extend(matches[: max(n, 0)])
        used.update(it.solution_npz for it in matches[: max(n, 0)])

    # If requested count is larger than available quota matches, fill deterministically with remaining profiles.
    target = max_profiles if max_profiles and max_profiles > 0 else len(selected)
    if len(selected) < target:
        for it in items:
            if it.solution_npz in used:
                continue
            selected.append(it)
            used.add(it.solution_npz)
            if len(selected) >= target:
                break
    return selected[:target]


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan and select D9.6 GV1 replay profiles.")
    ap.add_argument("--profile_root", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--quota", default="2C:2,R2.5:2,R3:2", help="Protocol quotas, e.g. 2C:2,R2.5:2,R3:2, or all")
    ap.add_argument("--max_profiles", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.profile_root)
    if not root.exists():
        raise FileNotFoundError(f"Profile root does not exist: {root}")
    items = scan_profiles(root)
    quota = parse_quota(args.quota)
    selected = select_profiles(items, quota=quota, max_profiles=args.max_profiles)
    counts: dict[str, int] = {}
    for it in items:
        counts[it.protocol] = counts.get(it.protocol, 0) + 1
    selected_counts: dict[str, int] = {}
    for it in selected:
        selected_counts[it.protocol] = selected_counts.get(it.protocol, 0) + 1

    payload = {
        "ok": bool(items),
        "profile_root": str(root),
        "total_profiles_found": len(items),
        "protocol_counts_found": counts,
        "selection_quota": args.quota,
        "max_profiles": args.max_profiles,
        "selected_count": len(selected),
        "selected_protocol_counts": selected_counts,
        "profiles": [asdict(it) for it in selected],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
