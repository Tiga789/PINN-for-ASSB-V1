from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "D18_S2_HOTFIX_PACKAGE_MANIFEST.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    errors = []
    for item in data.get('files', []):
        rel = item['path']
        path = ROOT / rel
        if not path.exists():
            errors.append(f'MISSING {rel}')
            continue
        size = path.stat().st_size
        digest = sha256(path)
        if size != item['size']:
            errors.append(f'SIZE {rel}: {size} != {item["size"]}')
        if digest != item['sha256']:
            errors.append(f'SHA256 {rel}: {digest} != {item["sha256"]}')
    if errors:
        print('FAIL: D18-S2 hotfix manifest verification errors')
        for e in errors:
            print('  - ' + e)
        return 2
    print(f'PASS: verified {len(data.get("files", []))} installed D18-S2 hotfix files')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
