"""Dataset-specific adapters for GV1.

This package contains thin adapters that convert public battery datasets into
GV1's standard measured-current replay representation.  The adapters should not
contain model training logic and should not modify the original ASSB pipeline.
"""

from .xjtu_adapter import (
    XJTUAdapter,
    parse_xjtu_file_metadata,
    read_xjtu_file,
    standardize_xjtu_dataframe,
)
from .xjtu_protocols import XJTU_BATCH_PROTOCOLS, get_xjtu_protocol, build_xjtu_protocol_mapping
from .xjtu_cell_spec_defaults import build_xjtu_default_cell_spec
from .xjtu_soh_targets import build_xjtu_cycle_capacity_table, build_xjtu_soh_targets

__all__ = [
    "XJTUAdapter",
    "parse_xjtu_file_metadata",
    "read_xjtu_file",
    "standardize_xjtu_dataframe",
    "XJTU_BATCH_PROTOCOLS",
    "get_xjtu_protocol",
    "build_xjtu_protocol_mapping",
    "build_xjtu_default_cell_spec",
    "build_xjtu_cycle_capacity_table",
    "build_xjtu_soh_targets",
]
