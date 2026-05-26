"""Protocol-label parsing for GV1 dataset discovery.

This is intentionally lightweight.  It labels observed protocol metadata from
filenames or optional manifest mappings; it does not decide the physics model.
The measured-current replay solver should still use recorded I(t).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Optional

from .cell_id_parser import sanitize_identifier


@dataclass(frozen=True)
class ProtocolInfo:
    protocol_id: str
    protocol_hint: Optional[str]
    observed_control_mode: str
    charge_rate_C_hint: Optional[float]
    discharge_rate_C_hint: Optional[float]
    is_partial_discharge_hint: Optional[bool]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_c_rate_token(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    m = re.search(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)\s*C\b", token, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def infer_control_mode_from_text(text: str) -> str:
    """Infer a coarse observed-control-mode label from free text."""
    t = text.lower()
    if any(x in t for x in ["random", "rw", "walk"]):
        return "dynamic_or_random_profile"
    if "cccv" in t or "cc-cv" in t or ("cc" in t and "cv" in t):
        return "cccv_record"
    if "constant_power" in t or "const_power" in t or "cp" in t:
        return "constant_power_record"
    if "constant_voltage" in t or "const_voltage" in t or "cv" in t:
        return "constant_voltage_record"
    if "pulse" in t or "dynamic" in t or "drive" in t:
        return "dynamic_current_record"
    return "measured_current_profile"


def protocol_from_hint(
    protocol_hint: Optional[str],
    *,
    batch_id: Optional[str] = None,
    protocol_mapping: Optional[Mapping[str, Any]] = None,
) -> ProtocolInfo:
    """Create a protocol info object from filename hints and optional mapping.

    ``protocol_mapping`` may contain entries keyed by batch_id or protocol_hint,
    for example:

    {
      "Batch-1": {"protocol_id": "xjtu_batch1_fixed_full", "observed_control_mode": "cccv_record"}
    }
    """
    mapping_entry: Mapping[str, Any] = {}
    if protocol_mapping:
        for key in [batch_id, protocol_hint]:
            if key and key in protocol_mapping and isinstance(protocol_mapping[key], Mapping):
                mapping_entry = protocol_mapping[key]
                break

    raw_id = mapping_entry.get("protocol_id") if mapping_entry else None
    if raw_id is None:
        raw_id = protocol_hint or batch_id or "measured_current_profile"
    protocol_id = sanitize_identifier(raw_id, default="measured_current_profile")

    observed_control_mode = str(
        mapping_entry.get("observed_control_mode")
        if mapping_entry
        else infer_control_mode_from_text(str(protocol_hint or batch_id or ""))
    )

    charge_rate = mapping_entry.get("charge_rate_C_hint") if mapping_entry else None
    discharge_rate = mapping_entry.get("discharge_rate_C_hint") if mapping_entry else None
    if charge_rate is None:
        charge_rate = _parse_c_rate_token(protocol_hint)
    if discharge_rate is None:
        discharge_rate = _parse_c_rate_token(protocol_hint)

    partial = mapping_entry.get("is_partial_discharge_hint") if mapping_entry else None
    if partial is not None:
        partial = bool(partial)

    notes = str(mapping_entry.get("notes", "")) if mapping_entry else ""
    return ProtocolInfo(
        protocol_id=protocol_id,
        protocol_hint=protocol_hint,
        observed_control_mode=observed_control_mode,
        charge_rate_C_hint=float(charge_rate) if charge_rate is not None else None,
        discharge_rate_C_hint=float(discharge_rate) if discharge_rate is not None else None,
        is_partial_discharge_hint=partial,
        notes=notes,
    )
