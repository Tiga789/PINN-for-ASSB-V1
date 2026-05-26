"""Protocol definitions for the XJTU 18650 NCM battery dataset.

The values here describe the *observed experimental strategy*.  GV1 still uses
``measured_current_profile`` as the actual solver input whenever current is
recorded in the data file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _norm_batch(batch_id: str | int | None) -> str:
    if batch_id is None:
        return "unknown"
    s = str(batch_id).strip()
    if s.lower().startswith("batch-"):
        try:
            return f"Batch-{int(s.split('-')[-1])}"
        except Exception:
            return s
    if s.lower().startswith("batch"):
        digits = "".join(ch for ch in s if ch.isdigit())
        return f"Batch-{int(digits)}" if digits else s
    if s.isdigit():
        return f"Batch-{int(s)}"
    return s


@dataclass(frozen=True)
class XJTUProtocol:
    batch_id: str
    protocol_id: str
    observed_control_mode: str
    charge_strategy: str
    discharge_strategy: str
    first_cycle_strategy: str
    nominal_capacity_Ah: float = 2.0
    voltage_upper_V: float = 4.2
    voltage_lower_full_V: float = 2.5
    voltage_lower_partial_V: float | None = None
    default_temperature_C: float = 25.0
    current_input_mode: str = "measured_current_profile"
    measured_current_replay_ok: bool = True
    needs_capacity_test_cycles_for_soh: bool = False
    high_rate_warning: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The wording follows the Chinese data-introduction file.  Batch-2/5/6 are
# included for future compatibility even if the first GV1 experiments use only
# Batch-1, Batch-3 and Batch-4.
XJTU_BATCH_PROTOCOLS: dict[str, XJTUProtocol] = {
    "Batch-1": XJTUProtocol(
        batch_id="Batch-1",
        protocol_id="xjtu_batch1_fixed_full_cccv_discharge1c",
        observed_control_mode="cccv_record",
        charge_strategy="Cycle 1: 0.5C CC to 4.2V then CV to 0.02C. Other cycles: 2C CC to 4.2V then CV to 0.05C.",
        discharge_strategy="Cycle 1: 0.2C discharge to 2.5V. Other cycles: 1C discharge to 2.5V. Full discharge.",
        first_cycle_strategy="Initial capacity test: 0.5C CC-CV charge, rest 5 min, 0.2C discharge to 2.5V.",
        notes="Fixed charge/discharge strategy; full charge and full discharge.",
    ),
    "Batch-2": XJTUProtocol(
        batch_id="Batch-2",
        protocol_id="xjtu_batch2_fixed_full_cccv_charge3c",
        observed_control_mode="cccv_record",
        charge_strategy="Cycle 1: 0.5C CC-CV. Other cycles: 3C CC to 4.2V then CV to 0.05C.",
        discharge_strategy="Cycle 1: 0.2C discharge to 2.5V. Other cycles: 1C discharge to 2.5V. Full discharge.",
        first_cycle_strategy="Initial capacity test: 0.5C CC-CV charge, rest 5 min, 0.2C discharge to 2.5V.",
        high_rate_warning=True,
        notes="High-rate charge may stress the effective SPM assumption.",
    ),
    "Batch-3": XJTUProtocol(
        batch_id="Batch-3",
        protocol_id="xjtu_batch3_variable_discharge_full",
        observed_control_mode="cccv_record_variable_discharge",
        charge_strategy="Cycle 1: 0.5C CC-CV. Other cycles: 2C CC-CV to 4.2V.",
        discharge_strategy="Other cycles: xC discharge to 2.5V, x cycles through {0.5,1,2,3,5}. Full discharge.",
        first_cycle_strategy="Initial capacity test: 0.5C CC-CV charge, rest 5 min, 0.2C discharge to 2.5V.",
        high_rate_warning=True,
        notes="Use measured I(t); 3C/5C discharge segments should be audited for effective SPM validity.",
    ),
    "Batch-4": XJTUProtocol(
        batch_id="Batch-4",
        protocol_id="xjtu_batch4_variable_discharge_partial_with_capacity_tests",
        observed_control_mode="cccv_record_variable_partial_discharge",
        charge_strategy="Cycle 1: 0.5C CC-CV. Other cycles: 2C CC-CV to 4.2V. Capacity-test cycles use 2C CC-CV charge.",
        discharge_strategy="Other cycles: xC discharge to 3.0V, x cycles through {0.5,1,2,3,5}; periodic capacity-test cycles discharge 1C to 2.5V.",
        first_cycle_strategy="Initial capacity test: 0.5C CC-CV charge, rest 5 min, 0.2C discharge to 2.5V.",
        voltage_lower_partial_V=3.0,
        needs_capacity_test_cycles_for_soh=True,
        high_rate_warning=True,
        notes="Partial-discharge operating cycles should not be used as full-capacity SOH labels; use capacity-test cycles.",
    ),
    "Batch-5": XJTUProtocol(
        batch_id="Batch-5",
        protocol_id="xjtu_batch5_random_walk_partial",
        observed_control_mode="dynamic_or_random_profile",
        charge_strategy="Early cycles 0.5C CC-CV; later cycles include 3C CC-CV charge before random discharge blocks.",
        discharge_strategy="Random-current partial discharge with safety cutoff at 3.0V; periodic capacity tests.",
        first_cycle_strategy="Random walk protocol starts with low-rate CC-CV and partial discharge; see data introduction.",
        voltage_lower_partial_V=3.0,
        needs_capacity_test_cycles_for_soh=True,
        high_rate_warning=True,
        notes="Useful as a later challenge set, not recommended as the first GV1 training batch.",
    ),
    "Batch-6": XJTUProtocol(
        batch_id="Batch-6",
        protocol_id="xjtu_batch6_geo_satellite_profile",
        observed_control_mode="mission_profile_record",
        charge_strategy="Geosynchronous Earth Orbit satellite simulation profile; use measured I(t).",
        discharge_strategy="GEO satellite simulation profile; use measured I(t) and capacity-test cycles for SOH when present.",
        first_cycle_strategy="Initial capacity test followed by mission-profile cycling.",
        needs_capacity_test_cycles_for_soh=True,
        notes="Valuable non-standard protocol validation set after Batch-1/3/4 are stable.",
    ),
}


def get_xjtu_protocol(batch_id: str | int | None) -> XJTUProtocol:
    key = _norm_batch(batch_id)
    if key in XJTU_BATCH_PROTOCOLS:
        return XJTU_BATCH_PROTOCOLS[key]
    return XJTUProtocol(
        batch_id=key,
        protocol_id="xjtu_unknown_batch_measured_replay",
        observed_control_mode="measured_current_profile",
        charge_strategy="Unknown; use measured current profile.",
        discharge_strategy="Unknown; use measured current profile.",
        first_cycle_strategy="Unknown.",
        notes="Batch not listed in built-in XJTU protocol map.",
    )


def build_xjtu_protocol_mapping(include_batches: list[str] | None = None) -> dict[str, Mapping[str, Any]]:
    """Return a mapping compatible with ``gv1.dataset_index.build_dataset_index``."""
    batches = include_batches or list(XJTU_BATCH_PROTOCOLS)
    out: dict[str, Mapping[str, Any]] = {}
    for b in batches:
        p = get_xjtu_protocol(b)
        d = p.to_dict()
        out[p.batch_id] = {
            "protocol_id": d["protocol_id"],
            "observed_control_mode": d["observed_control_mode"],
            "charge_rate_C_hint": None,
            "discharge_rate_C_hint": None,
            "is_partial_discharge_hint": bool(p.voltage_lower_partial_V is not None),
            "notes": p.notes,
        }
    return out
