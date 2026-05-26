from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.pipeline.metrics import regression_metrics


def test_metrics_perfect():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m['mae'] == 0
    assert m['r2'] == 1
