from pathlib import Path
import tempfile

from gv1.data_discovery import discover_files
from gv1.dataset_index import build_dataset_index


def test_discover_and_index_xjtu_like_paths():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "Batch-1").mkdir()
        (root / "Batch-3").mkdir()
        (root / "Batch-1" / "2C_battery-1.mat").write_bytes(b"fake")
        (root / "Batch-3" / "R2.5_battery-2_cycle_0001.csv").write_text("t,I,V\n0,0,3.7\n", encoding="utf-8")
        (root / "Batch-9" ).mkdir()
        (root / "Batch-9" / "ignore.txt").write_text("x", encoding="utf-8")

        files = discover_files(root, file_patterns=["*.mat", "*.csv"], include_batches=["Batch-1", "Batch-3"])
        assert len(files) == 2
        rows, summary = build_dataset_index(dataset_root=root, dataset_id="XJTU", include_batches=["Batch-1", "Batch-3"])
        assert len(rows) == 2
        assert summary["index"]["cell_count"] == 2
        assert any(r.batch_id == "Batch-1" and r.battery_id == "battery-1" for r in rows)
