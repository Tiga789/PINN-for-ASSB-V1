from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .common import ConfigError, flatten_records, load_json, normalize_path_text, read_csv


_CANONICAL_RE = re.compile(r"^Batch-(?P<batch>\d+)_(?P<protocol>.+)_battery-(?P<battery>\d+)$", re.IGNORECASE)
_CELL_RE = re.compile(r"^Batch-(?P<batch>\d+)_battery-(?P<battery>\d+)$", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class CanonicalUID:
    batch: int
    protocol: str
    battery: int

    @property
    def canonical(self) -> str:
        return f"Batch-{self.batch}_{self.protocol}_battery-{self.battery}"

    @property
    def cell_uid(self) -> str:
        return f"Batch-{self.batch}_battery-{self.battery}"

    @property
    def branch_family(self) -> str:
        return "P4D" if self.batch in {5, 6} or self.protocol.lower() in {"random_walk", "geo"} else "RG"


def parse_canonical_uid(value: str) -> CanonicalUID:
    text = str(value).strip()
    match = _CANONICAL_RE.fullmatch(text)
    if not match:
        raise ConfigError(f"Invalid canonical_cell_uid: {value!r}")
    return CanonicalUID(int(match.group("batch")), match.group("protocol"), int(match.group("battery")))


def parse_cell_uid(value: str) -> tuple[int, int]:
    text = str(value).strip()
    match = _CELL_RE.fullmatch(text)
    if not match:
        raise ConfigError(f"Invalid cell_uid: {value!r}")
    return int(match.group("batch")), int(match.group("battery"))


def canonical_from_record(record: Mapping[str, Any]) -> CanonicalUID:
    canonical = str(record.get("canonical_cell_uid", "")).strip()
    if canonical:
        return parse_canonical_uid(canonical)
    batch_text = str(record.get("batch", "")).strip()
    battery_text = str(record.get("battery", "")).strip()
    protocol = str(record.get("protocol", "")).strip()
    bmatch = re.fullmatch(r"Batch-(\d+)", batch_text, flags=re.IGNORECASE)
    cmatch = re.fullmatch(r"battery-(\d+)", battery_text, flags=re.IGNORECASE)
    if not (bmatch and cmatch and protocol):
        raise ConfigError(f"Record does not define an exact canonical UID: {record}")
    return CanonicalUID(int(bmatch.group(1)), protocol, int(cmatch.group(1)))


def load_split_index(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_json(path)
    records = flatten_records(manifest)
    index: dict[str, dict[str, Any]] = {}
    for raw in records:
        uid = canonical_from_record(raw)
        key = uid.canonical.lower()
        if key in index:
            raise ConfigError(f"Duplicate canonical UID in split manifest: {uid.canonical}")
        record = dict(raw)
        record["canonical_cell_uid"] = uid.canonical
        record["cell_uid"] = uid.cell_uid
        record["batch"] = f"Batch-{uid.batch}"
        record["battery"] = f"battery-{uid.battery}"
        record["protocol"] = uid.protocol
        index[key] = record
    return manifest, index


def load_role_index(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    index: dict[str, dict[str, Any]] = {}
    stage_to_role = {
        "g2_train_fit": "fit_train",
        "g2_train_internal_heldout": "internal_heldout",
        "g2_validation_report_only": "validation_report_only",
    }
    for row in rows:
        uid_text = str(row.get("canonical_cell_uid", "")).strip()
        # D17-G2 release manifests use the column name ``split`` while some
        # development snapshots used ``stage``. Accept only the exact known
        # tokens; do not infer roles from paths or substring matches.
        stage_raw = row.get("stage") or row.get("split") or ""
        stage = str(stage_raw).strip().lower()
        if not uid_text or stage not in stage_to_role:
            continue
        uid = parse_canonical_uid(uid_text)
        key = uid.canonical.lower()
        if key in index:
            raise ConfigError(f"Duplicate role row for {uid.canonical}")
        index[key] = {**row, "role": stage_to_role[stage], "canonical_cell_uid": uid.canonical}
    return index


def select_exact_records(
    *,
    split_index: Mapping[str, Mapping[str, Any]],
    role_index: Mapping[str, Mapping[str, Any]],
    requested: Mapping[str, Iterable[str]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, values in requested.items():
        for value in values:
            uid = parse_canonical_uid(str(value))
            key = uid.canonical.lower()
            if key in seen:
                raise ConfigError(f"Requested profile appears more than once: {uid.canonical}")
            if key not in split_index:
                raise ConfigError(f"Requested UID not found in split manifest: {uid.canonical}")
            if key not in role_index:
                raise ConfigError(f"Requested UID not found in G2 role manifest: {uid.canonical}")
            observed_role = str(role_index[key].get("role", ""))
            if observed_role != role:
                raise ConfigError(
                    f"Role mismatch for {uid.canonical}: requested={role}, G2 manifest={observed_role}"
                )
            record = dict(split_index[key])
            record["d18_s2_role"] = role
            record["g2_stage"] = role_index[key].get("stage", "")
            record["semantic_branch"] = role_index[key].get("semantic_branch", "")
            record["branch_family"] = uid.branch_family
            selected.append(record)
            seen.add(key)
    return selected


def _scalar_text(npz: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key not in npz:
            continue
        arr = np.asarray(npz[key])
        if arr.size == 0:
            continue
        value = arr.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            return text
    return ""


def npz_identity(path: str | Path) -> dict[str, str]:
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        canonical = _scalar_text(data, ["canonical_cell_uid", "profile_uid", "profile_key"])
        cell_uid = _scalar_text(data, ["cell_uid"])
        batch = _scalar_text(data, ["batch"])
        protocol = _scalar_text(data, ["protocol"])
    return {
        "canonical_cell_uid": canonical,
        "cell_uid": cell_uid,
        "batch": batch,
        "protocol": protocol,
    }


def path_mentions_exact_uid(path: str | Path, uid: CanonicalUID) -> bool:
    text = normalize_path_text(path)
    canonical = uid.canonical.lower()
    cell = uid.cell_uid.lower()
    canonical_ok = re.search(
        rf"(?<![a-z0-9]){re.escape(canonical)}(?!\d)", text, flags=re.IGNORECASE
    ) is not None
    if canonical_ok:
        return True
    # For paths that omit Batch in an indexed folder name, require exact battery token,
    # exact protocol token and exact Batch token somewhere in the full path.
    batch_ok = re.search(rf"(?<!\d)batch-{uid.batch}(?!\d)", text, flags=re.IGNORECASE) is not None
    battery_ok = re.search(rf"(?<!\d)battery-{uid.battery}(?!\d)", text, flags=re.IGNORECASE) is not None
    protocol_ok = uid.protocol.lower() in text
    cell_ok = re.search(
        rf"(?<![a-z0-9]){re.escape(cell)}(?!\d)", text, flags=re.IGNORECASE
    ) is not None
    return bool(batch_ok and battery_ok and protocol_ok) or bool(cell_ok and protocol_ok)


def build_replay_index(replay_roots: Iterable[str | Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root_value in replay_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("solution_replay_profile.npz"):
            candidates: set[str] = set()
            try:
                identity = npz_identity(path)
                canonical = identity.get("canonical_cell_uid", "")
                if canonical:
                    candidates.add(parse_canonical_uid(canonical).canonical.lower())
            except Exception:
                pass
            # Path fallback is strict and tests complete numeric tokens.
            for match in re.finditer(
                r"Batch-(\d+)_([^\\/]+?)_battery-(\d+)", str(path), flags=re.IGNORECASE
            ):
                try:
                    candidates.add(
                        CanonicalUID(int(match.group(1)), match.group(2), int(match.group(3))).canonical.lower()
                    )
                except Exception:
                    continue
            for key in candidates:
                index.setdefault(key, []).append(path)
    for paths in index.values():
        paths.sort(key=lambda p: (len(str(p)), normalize_path_text(p)))
    return index


def resolve_replay_path(record: Mapping[str, Any], replay_index: Mapping[str, list[Path]]) -> tuple[Path | None, str]:
    uid = canonical_from_record(record)
    declared_text = str(record.get("replay_npz", "")).strip()
    if declared_text:
        declared = Path(declared_text)
        if declared.exists():
            try:
                identity = npz_identity(declared)
                canonical = identity.get("canonical_cell_uid", "")
                if canonical and parse_canonical_uid(canonical) == uid:
                    return declared, "declared_npz_metadata_exact"
            except Exception:
                pass
            if path_mentions_exact_uid(declared, uid):
                return declared, "declared_path_exact"
    matches = list(replay_index.get(uid.canonical.lower(), []))
    if len(matches) == 1:
        return matches[0], "reindexed_exact_uid"
    if len(matches) > 1:
        # Prefer the shortest path, but expose duplicate count to the caller.
        return matches[0], f"reindexed_exact_uid_ambiguous_{len(matches)}"
    return None, "not_found"


def audit_record_identity(record: Mapping[str, Any], replay_path: Path | None) -> dict[str, Any]:
    uid = canonical_from_record(record)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        cbatch, cbattery = parse_cell_uid(str(record.get("cell_uid", "")))
        if (cbatch, cbattery) != (uid.batch, uid.battery):
            errors.append("cell_uid_mismatch")
    except ConfigError:
        errors.append("invalid_cell_uid")
    expected_batch = f"Batch-{uid.batch}"
    expected_battery = f"battery-{uid.battery}"
    if str(record.get("batch", "")) != expected_batch:
        errors.append("batch_field_mismatch")
    if str(record.get("battery", "")) != expected_battery:
        errors.append("battery_field_mismatch")
    if str(record.get("protocol", "")) != uid.protocol:
        errors.append("protocol_field_mismatch")

    softlabel = Path(str(record.get("softlabel_npz", "")))
    expected_parent = uid.cell_uid.lower()
    if not softlabel.exists():
        errors.append("softlabel_npz_missing")
    elif softlabel.parent.name.lower() != expected_parent:
        errors.append("softlabel_parent_not_exact_cell_uid")

    if replay_path is None:
        warnings.append("replay_npz_not_resolved")
    else:
        try:
            identity = npz_identity(replay_path)
            canonical = identity.get("canonical_cell_uid", "")
            if canonical and parse_canonical_uid(canonical) != uid:
                errors.append("replay_metadata_uid_mismatch")
            elif not canonical and not path_mentions_exact_uid(replay_path, uid):
                warnings.append("replay_identity_not_embedded")
        except Exception as exc:
            warnings.append(f"replay_identity_read_failed:{type(exc).__name__}")

    return {
        "canonical_cell_uid": uid.canonical,
        "cell_uid": uid.cell_uid,
        "role": str(record.get("d18_s2_role", "")),
        "split": str(record.get("split", "")),
        "protocol": uid.protocol,
        "branch_family": uid.branch_family,
        "softlabel_npz": str(softlabel),
        "replay_npz": str(replay_path) if replay_path else "",
        "identity_status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
    }
