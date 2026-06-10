from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='D15-P4D CUDA compute smoke. This does not generate soft labels.')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--seconds', type=float, default=25.0)
    ap.add_argument('--matrix-size', type=int, default=2048)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    res: Dict[str, Any] = {
        'stage': 'D15-P4D CUDA smoke',
        'purpose': 'Verify that CUDA compute can be utilized; current NumPy P2Dlite-RG label generator does not automatically use GPU.',
        'torch_import_ok': False,
        'cuda_available': False,
        'status': 'REVIEW',
    }
    try:
        import torch
        res['torch_import_ok'] = True
        res['cuda_available'] = bool(torch.cuda.is_available())
        if not res['cuda_available']:
            res['status'] = 'REVIEW'
            res['note'] = 'torch imported but CUDA is not available'
            write_json(res, args.out_json)
            print('[D15-P4D CUDA smoke] CUDA unavailable')
            return 0
        dev = torch.device('cuda')
        res['device_name'] = torch.cuda.get_device_name(0)
        n = int(args.matrix_size)
        torch.cuda.empty_cache()
        a = torch.randn((n, n), device=dev, dtype=torch.float32)
        b = torch.randn((n, n), device=dev, dtype=torch.float32)
        c = torch.empty((n, n), device=dev, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 0
        # Run at least 3 iterations and approximately args.seconds seconds.
        while True:
            c = a @ b
            a = c.mul(1e-5).add_(a, alpha=0.99999)
            iters += 1
            if iters >= 3:
                torch.cuda.synchronize()
                if time.perf_counter() - t0 >= float(args.seconds):
                    break
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # GEMM flop estimate: 2*n^3 per matmul.
        tflops = (2.0 * (n ** 3) * iters) / max(dt, 1e-9) / 1e12
        res.update({
            'status': 'PASS',
            'matrix_size': n,
            'iterations': iters,
            'elapsed_seconds': dt,
            'estimated_tflops': tflops,
            'memory_allocated_mb': float(torch.cuda.max_memory_allocated(0) / (1024 * 1024)),
            'note': 'CUDA compute smoke passed. This does not mean NumPy generator uses GPU.'
        })
        write_json(res, args.out_json)
        print('[D15-P4D CUDA smoke] PASS device=', res['device_name'], 'iters=', iters, 'seconds=', round(dt, 2))
        return 0
    except Exception as exc:
        res['status'] = 'REVIEW'
        res['error'] = repr(exc)
        write_json(res, args.out_json)
        print('[D15-P4D CUDA smoke] REVIEW', repr(exc))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
