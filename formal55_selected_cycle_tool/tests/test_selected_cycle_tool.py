from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
path = SCRIPTS / "formal55_selected_cycle_infer_plot.py"
spec = importlib.util.spec_from_file_location("selected_tool", path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
result = module.self_test()
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result.get("self_test") == "PASS" else 1)
