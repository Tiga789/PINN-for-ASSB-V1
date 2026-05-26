from pathlib import Path

from gv1.adapters.xjtu_adapter import parse_xjtu_file_metadata
from gv1.adapters.xjtu_protocols import get_xjtu_protocol
from gv1.adapters.xjtu_cell_spec_defaults import build_xjtu_default_cell_spec


def test_xjtu_metadata_from_path():
    p = Path(r"E:/XJTU battery dataset/Batch-1/2C_battery-1.mat")
    meta = parse_xjtu_file_metadata(p, dataset_root=r"E:/XJTU battery dataset")
    assert meta.batch_id == "Batch-1"
    assert meta.battery_id == "battery-1"
    assert meta.observed_control_mode == "cccv_record"


def test_batch4_protocol_requires_capacity_tests():
    proto = get_xjtu_protocol("Batch-4")
    assert proto.needs_capacity_test_cycles_for_soh is True
    assert proto.voltage_lower_partial_V == 3.0


def test_default_cell_spec_temperature_and_capacity():
    spec = build_xjtu_default_cell_spec("xjtu_test_cell", temperature_C=25)
    assert spec["ratings"]["nominal_capacity_Ah"] == 2.0
    assert spec["cell"]["operating_temperature_C_default"] == 25.0
    assert spec["chemistry"]["positive_electrode_material"] == "NCM523"
