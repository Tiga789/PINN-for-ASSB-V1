from __future__ import annotations

import pandas as pd

from gv1.io.field_mapper import standardize_dataframe, validate_standard_table
from gv1.io.reader_base import ReadOptions


def test_repair_nonmonotonic_mat_subrecord_time_axis():
    raw = pd.DataFrame(
        {
            "system_time": ["2024-01-02,00:00:00", "2024-01-02,00:00:01", "2024-01-01,00:00:00", "2024-01-01,00:00:01"],
            "relative_time_min": [0.0, 1.0 / 60.0, 0.0, 1.0 / 60.0],
            "voltage_V": [3.7, 3.6, 3.8, 3.7],
            "current_A": [0.0, -1.0, 0.0, -1.0],
            "temperature_C": [25.0, 25.0, 25.0, 25.0],
            "mat_subrecord_index": [0, 0, 1, 1],
        }
    )
    out, warnings = standardize_dataframe(raw, ReadOptions(dataset_id="TEST", batch_id="B1", battery_id="C1"))
    valid, problems = validate_standard_table(out)
    assert valid, problems
    assert "time_s" in out.columns
    assert out["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert any("time_s rebuilt" in w for w in warnings)
