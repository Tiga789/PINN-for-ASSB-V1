"""Default CellSpec anchors for the XJTU 18650 NCM dataset.

Only use facts that are stated in the data-introduction file as hard anchors.
Items such as graphite negative electrode and geometry are marked as assumptions
or effective parameters because they cannot be uniquely inferred from t/I/V.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


XJTU_DEFAULT_CELL_SPEC: dict[str, Any] = {
    "schema_version": "gv1.0",
    "dataset_id": "XJTU",
    "cell": {
        "cell_instance_id": "xjtu_18650_ncm523_graphite_assumed",
        "manufacturer": "Lishen",
        "form_factor": "18650",
        "model_family": "effective_spm",
        "electrolyte_system": "liquid_lumped_effective_spm",
        "operating_temperature_C_default": 25.0,
        "operating_temperature_source": "dataset_room_temperature_default",
    },
    "chemistry": {
        "positive_electrode_material": "NCM523",
        "positive_formula": "LiNi0.5Co0.2Mn0.3O2",
        "negative_electrode_material": "graphite_assumed",
        "negative_material_source": "engineering_assumption_not_explicit_in_dataset_intro",
        "ocp_configuration_mode": "library_or_low_rate_ocv_fit",
    },
    "ratings": {
        "nominal_capacity_Ah": 2.0,
        "nominal_voltage_V": 3.6,
        "voltage_upper_V": 4.2,
        "voltage_lower_full_discharge_V": 2.5,
        "voltage_lower_partial_discharge_V": 3.0,
    },
    "capacity_anchor": {
        "mode": "initial_capacity_cycle_or_discharge_integral",
        "q_nominal_Ah": 2.0,
        "q_ref_strategy": "first_low_rate_capacity_or_early_stable_mean",
        "soh_definition": "Q_discharge_cycle / Q_ref",
    },
    "geometry_anchor": {
        "mode": "capacity_normalized_effective_spm",
        "requires_true_internal_geometry": False,
        "notes": "18650 internal electrode area/thickness/loading are not in the data-introduction file; use effective parameters or adapter calibration.",
    },
    "parameter_priors": {
        "positive_particle_radius_m": "material_library_or_calibrated_effective",
        "negative_particle_radius_m": "material_library_or_calibrated_effective",
        "eps_s_positive": "material_default_or_calibrated_effective",
        "eps_s_negative": "material_default_or_calibrated_effective",
        "csmax_positive": "material_library_ncm523_or_capacity_scaled",
        "csmax_negative": "material_library_graphite_or_capacity_scaled",
        "Ds_positive": "broad_prior_then_calibrate",
        "Ds_negative": "broad_prior_then_calibrate",
        "i0_positive": "broad_prior_then_calibrate",
        "i0_negative": "broad_prior_then_calibrate",
        "R_ohm_eff": "estimate_from_current_steps_then_calibrate",
    },
    "warnings": [
        "negative electrode material is assumed as graphite; verify from dataset paper/manufacturer if possible",
        "internal geometry is unavailable; first GV1 implementation should use capacity-normalized effective SPM",
        "Batch-3/4 include high-rate 3C/5C discharge segments; audit effective SPM validity before using as held-out proof",
    ],
}


def build_xjtu_default_cell_spec(cell_instance_id: str | None = None, *, temperature_C: float = 25.0) -> dict[str, Any]:
    spec = deepcopy(XJTU_DEFAULT_CELL_SPEC)
    if cell_instance_id:
        spec["cell"]["cell_instance_id"] = cell_instance_id
    spec["cell"]["operating_temperature_C_default"] = float(temperature_C)
    return spec
