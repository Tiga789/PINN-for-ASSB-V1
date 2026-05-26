import pandas as pd
from gv1.io.field_mapper import standardize_dataframe
from gv1.io.reader_base import ReadOptions


def test_xjtu_time_axis_uses_relative_time_and_subrecord():
    df = pd.DataFrame({
        "system_time": ["2022-01-01,00:00:00", "2022-01-01,00:00:01", "2021-12-31,23:59:00", "2021-12-31,23:59:01"],
        "relative_time_min": [0.0, 1/60, 0.0, 1/60],
        "voltage_V": [3.7, 3.7, 3.8, 3.8],
        "current_A": [0.0, 1.0, 0.0, -1.0],
        "temperature_C": [25, 25, 25, 25],
        "mat_subrecord_index": [0, 0, 1, 1],
    })
    out, warnings = standardize_dataframe(df, ReadOptions(dataset_id="XJTU"), source_format="mat")
    assert out["time_s"].is_monotonic_increasing or out["time_s"].is_monotonic_non_decreasing
    assert "raw__system_time" in out.columns
    assert any("GV1 time fix v3" in w for w in warnings)
