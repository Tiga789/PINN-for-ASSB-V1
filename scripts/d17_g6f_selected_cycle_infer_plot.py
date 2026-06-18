from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g6f_selected_cycle_infer import run_selected_cycle_infer_plot


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6F selected-cycle on-demand inference, metrics and 3D plotting.")
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--candidate_dir", required=True)
    ap.add_argument("--candidate_summary", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", required=True, help="Example: 2 or Batch-2")
    ap.add_argument("--battery", required=True, help="Example: 3 or battery-3")
    ap.add_argument("--cycles", required=True, help="Example: 13-15, 1,3,5, or all")
    ap.add_argument("--metric_targets", nargs="+", default=["cs_a", "cs_c", "phie", "phis_c"], help="Targets for metrics. Use 'all' for all checkpoint targets.")
    ap.add_argument("--plot_targets", nargs="+", default=["both"], help="Targets for 3D plots. Use both for cs_a and cs_c.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--predict_batch_size", type=int, default=8192)
    ap.add_argument("--max_points_per_cycle", type=int, default=0, help="0 means use all points in selected cycles.")
    ap.add_argument("--prefer_softlabel_observed", action="store_true", help="Use observed I/V/T copies inside softlabel NPZ instead of replay interpolation.")
    ap.add_argument("--plot_3d", action="store_true", help="Open interactive Matplotlib 3D windows.")
    ap.add_argument("--save_png", action="store_true", help="Save 3D figures to out_dir/figures.")
    ap.add_argument("--backend", default="", help="Optional Matplotlib backend, e.g. QtAgg or TkAgg.")
    ap.add_argument("--plot_max_time_points", type=int, default=1200)
    ap.add_argument("--time_axis", choices=["relative", "global"], default="relative")
    ap.add_argument("--pred_cmap", default="coolwarm")
    ap.add_argument("--true_cmap", default="viridis")
    ap.add_argument("--save_temp_npz", action="store_true", help="Save temporary selected-cycle prediction npz.")
    ap.add_argument("--keep_temp_npz", action="store_true", help="Keep temporary npz even if delete_temp_predictions is set.")
    ap.add_argument("--delete_temp_predictions", action="store_true", help="Delete temporary prediction npz after plotting/metrics.")
    ap.add_argument("--r2_mean_gate", type=float, default=0.98)
    ap.add_argument("--r2_min_gate", type=float, default=0.95)
    args = ap.parse_args()
    summary = run_selected_cycle_infer_plot(args)
    print(json.dumps({
        "status": summary.get("status"),
        "full_training_recommendation": summary.get("full_training_recommendation"),
        "aggregate_metrics": summary.get("aggregate_metrics"),
        "n_time_points": summary.get("n_time_points"),
        "evaluated_cycles": summary.get("evaluated_cycles"),
        "summary_json": summary.get("files", {}).get("summary_json"),
        "metrics_csv": summary.get("files", {}).get("metrics_csv"),
        "plot_files": summary.get("files", {}).get("plot_files"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
