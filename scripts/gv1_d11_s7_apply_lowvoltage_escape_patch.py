#!/usr/bin/env python
"""Apply D11-S7 low-voltage escape-head patch to GV1 mainline files.

This patch is intentionally narrow and reversible:
- backs up gv1/output_transform.py and scripts/gv1_train_conditioned_pinn.py
- adds optional low-voltage escape config fields to output_transform.py
- inserts a deterministic low-voltage escape correction before hard-clamp logic
- adds CLI arguments to gv1_train_conditioned_pinn.py and passes them into the
  existing transform config key list.

The escape correction is designed for diagnostic smoke runs only.  It is NOT a
mainline promotion.  It keeps hard clamp disabled and is activated only by CLI.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path


def backup(path: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    b = path.with_suffix(path.suffix + f".D11S7_backup_{stamp}")
    b.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return b


def insert_after(text: str, marker: str, insertion: str) -> tuple[str, bool]:
    if insertion.strip() in text:
        return text, False
    idx = text.find(marker)
    if idx < 0:
        return text, False
    pos = idx + len(marker)
    return text[:pos] + insertion + text[pos:], True


def patch_output_transform(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists(), "changed": False, "notes": []}
    if not path.exists():
        out["notes"].append("missing output_transform.py")
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    original = text
    b = backup(path)
    out["backup"] = str(b)

    # 1) Add dataclass/config fields near low-voltage gate fields.
    if "enable_low_voltage_escape" not in text:
        marker_candidates = [
            "low_voltage_gate_width_V: float = 0.18\n",
            "low_voltage_gate_width_V",
        ]
        inserted = False
        field_block = (
            "    # D11-S7 diagnostic low-voltage escape head. Disabled by default.\n"
            "    enable_low_voltage_escape: bool = False\n"
            "    low_voltage_escape_scale_V: float = 0.0\n"
            "    low_voltage_escape_gate_center_V: float = 3.08\n"
            "    low_voltage_escape_gate_width_V: float = 0.18\n"
            "    low_voltage_escape_pred_center_V: float = 3.55\n"
            "    low_voltage_escape_pred_width_V: float = 0.18\n"
        )
        for m in marker_candidates:
            if m in text:
                if m.endswith("\n"):
                    text = text.replace(m, m + field_block, 1)
                else:
                    # Fallback: insert after line containing marker.
                    lines = text.splitlines(True)
                    for i, line in enumerate(lines):
                        if m in line:
                            lines.insert(i + 1, field_block)
                            text = "".join(lines)
                            break
                inserted = True
                break
        out["notes"].append("added escape config fields" if inserted else "could not add escape config fields automatically")

    # 2) Insert correction block before hard-clamp if present.
    if "D11-S7 low-voltage escape diagnostic block" not in text:
        escape_block = r'''
        # D11-S7 low-voltage escape diagnostic block.
        # Motivation: D11-S6 showed a structural low-voltage floor barrier;
        # low-target points had true V≈2.5-2.9 V while predictions stayed near
        # 3.4 V.  This diagnostic branch allows the output map to escape
        # downward when the low-voltage gate is active.  It is disabled by
        # default and should only be activated in D11-S7 smoke runs.
        voltage_escape = torch.zeros_like(phis_c)
        if bool(getattr(self.config, "enable_low_voltage_escape", False)):
            esc_center = float(getattr(self.config, "low_voltage_escape_gate_center_V", getattr(self.config, "low_voltage_gate_center_V", 3.08)))
            esc_width = max(float(getattr(self.config, "low_voltage_escape_gate_width_V", getattr(self.config, "low_voltage_gate_width_V", 0.18))), 1e-3)
            pred_center = float(getattr(self.config, "low_voltage_escape_pred_center_V", 3.55))
            pred_width = max(float(getattr(self.config, "low_voltage_escape_pred_width_V", 0.18)), 1e-3)
            esc_scale = float(getattr(self.config, "low_voltage_escape_scale_V", 0.0))
            escape_gate_from_ocv = torch.sigmoid((esc_center - v_ocv) / esc_width)
            escape_gate_from_pred = torch.sigmoid((pred_center - phis_c) / pred_width)
            voltage_escape = esc_scale * escape_gate_from_ocv * escape_gate_from_pred
            phis_c = phis_c - voltage_escape
'''
        # Try inserting immediately before hard clamp block.
        hard_markers = [
            "        if bool(self.config.enable_voltage_hard_clamp):",
            "        if bool(getattr(self.config, \"enable_voltage_hard_clamp\", False)):",
            "        if bool(self.config.enable_voltage_hard_clamp)",
        ]
        inserted = False
        for m in hard_markers:
            if m in text:
                text = text.replace(m, escape_block + "\n" + m, 1)
                inserted = True
                break
        if not inserted:
            # Fallback: insert before return dictionary line containing voltage_exp_pred.
            m = '"voltage_exp_pred"'
            idx = text.find(m)
            if idx >= 0:
                line_start = text.rfind("\n", 0, idx)
                text = text[:line_start+1] + escape_block + text[line_start+1:]
                inserted = True
        out["notes"].append("inserted escape block" if inserted else "could not insert escape block automatically")

    # 3) Add output dictionary entries if the return dict exists.
    if '"voltage_low_escape_correction"' not in text and '"voltage_exp_pred"' in text:
        text = text.replace(
            '"voltage_exp_pred": phis_c,',
            '"voltage_exp_pred": phis_c,\n            "voltage_low_escape_correction": voltage_escape,',
            1,
        )
        out["notes"].append("added voltage_low_escape_correction output")

    if text != original:
        path.write_text(text, encoding="utf-8")
        out["changed"] = True
    else:
        out["notes"].append("no changes made to output_transform.py")
    return out


def patch_train_script(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists(), "changed": False, "notes": []}
    if not path.exists():
        out["notes"].append("missing train script")
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    original = text
    b = backup(path)
    out["backup"] = str(b)

    # Add CLI args after enable_voltage_hard_clamp parser arg.
    if "--enable_low_voltage_escape" not in text:
        arg_block = (
            "    # D11-S7 diagnostic low-voltage escape head arguments.\n"
            "    ap.add_argument(\"--enable_low_voltage_escape\", type=_bool_arg, default=None)\n"
            "    ap.add_argument(\"--low_voltage_escape_scale_V\", type=float, default=None)\n"
            "    ap.add_argument(\"--low_voltage_escape_gate_center_V\", type=float, default=None)\n"
            "    ap.add_argument(\"--low_voltage_escape_gate_width_V\", type=float, default=None)\n"
            "    ap.add_argument(\"--low_voltage_escape_pred_center_V\", type=float, default=None)\n"
            "    ap.add_argument(\"--low_voltage_escape_pred_width_V\", type=float, default=None)\n"
        )
        # Insert after hard clamp arg if possible.
        pat = re.compile(r"^\s*ap\.add_argument\(\"--enable_voltage_hard_clamp\".*\)\s*$", re.M)
        m = pat.search(text)
        if m:
            line_end = text.find("\n", m.end())
            if line_end < 0:
                line_end = m.end()
            text = text[:line_end+1] + arg_block + text[line_end+1:]
            out["notes"].append("added escape CLI args")
        else:
            out["notes"].append("could not locate enable_voltage_hard_clamp arg; escape CLI args not inserted")

    # Add keys to config pass-through list if present.
    keys = [
        "enable_low_voltage_escape",
        "low_voltage_escape_scale_V",
        "low_voltage_escape_gate_center_V",
        "low_voltage_escape_gate_width_V",
        "low_voltage_escape_pred_center_V",
        "low_voltage_escape_pred_width_V",
    ]
    for key in keys:
        if f'"{key}"' not in text:
            # Insert after enable_voltage_hard_clamp string in the config key list.
            marker = '"enable_voltage_hard_clamp",'
            if marker in text:
                text = text.replace(marker, marker + f'\n        "{key}",', 1)
                out["notes"].append(f"added transform pass-through key {key}")
            else:
                out["notes"].append(f"could not add pass-through key {key}; marker missing")

    if text != original:
        path.write_text(text, encoding="utf-8")
        out["changed"] = True
    else:
        out["notes"].append("no changes made to train script")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    root = Path(args.project_root)
    report = {
        "ok": True,
        "stage": "D11-S7 apply low-voltage escape diagnostic patch",
        "project_root": str(root),
        "output_transform": patch_output_transform(root / "gv1" / "output_transform.py"),
        "train_script": patch_train_script(root / "scripts" / "gv1_train_conditioned_pinn.py"),
    }
    report["ok"] = bool(report["output_transform"].get("exists")) and bool(report["train_script"].get("exists"))
    out_path = Path(args.report) if args.report else root / "D11_S7_patch_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
