from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d18_s2.common import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify installed D18-S2 package files")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest", default="D18_S2_PACKAGE_MANIFEST.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = 0
    for item in data.get("files", []):
        rel = str(item["relative_path"]).replace("\\", "/")
        path = root / Path(rel)
        if not path.exists():
            errors.append(f"MISSING {rel}")
            continue
        size = path.stat().st_size
        if size != int(item["size_bytes"]):
            errors.append(f"SIZE {rel}: {size} != {item['size_bytes']}")
            continue
        digest = sha256_file(path)
        if digest.lower() != str(item["sha256"]).lower():
            errors.append(f"SHA256 {rel}: {digest} != {item['sha256']}")
            continue
        checked += 1
    if errors:
        print("FAIL: D18-S2 package verification errors", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(f"PASS: verified {checked} installed D18-S2 package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
