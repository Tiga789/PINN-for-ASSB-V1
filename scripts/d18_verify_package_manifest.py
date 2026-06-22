from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from gv1.d18_cycleaware.common import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify installed D18-S0/S1-FIX package files against its manifest")
    parser.add_argument("--root", default=str(PACKAGE_ROOT))
    parser.add_argument("--manifest", default="D18_S0_S1_FIX_PACKAGE_MANIFEST.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest.get("files", []):
        rel = str(entry["path"])
        path = root / rel
        if not path.is_file():
            failures.append(f"MISSING {rel}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(entry["size_bytes"]):
            failures.append(f"SIZE {rel}: {size} != {entry['size_bytes']}")
        if digest.lower() != str(entry["sha256"]).lower():
            failures.append(f"SHA256 {rel}: {digest} != {entry['sha256']}")
    if failures:
        print("FAIL: package verification errors")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print(f"PASS: verified {len(manifest.get('files', []))} D18-S0/S1-FIX package files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
