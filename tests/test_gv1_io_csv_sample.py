"""Optional smoke test for the uploaded XJTU-like CSV sample.

Run from repository root after placing a sample file:
python -m pytest tests/test_gv1_io_csv_sample.py
"""
from pathlib import Path

from gv1.io import ReadOptions, read_battery_file
from gv1.io.field_mapper import validate_standard_table


def test_xjtu_like_csv_sample(tmp_path):
    sample = Path("R2.5_battery-1_cycle_0001.csv")
    if not sample.exists():
        return
    result = read_battery_file(
        sample,
        ReadOptions(dataset_id="XJTU", batch_id="Batch-3", battery_id="battery-1", cycle_id=1),
    )
    ok, problems = validate_standard_table(result.dataframe)
    assert ok, problems
    assert {"time_s", "current_A", "voltage_V", "temperature_C"}.issubset(result.dataframe.columns)
