"""Generic battery data readers for GV1.

This layer intentionally does not depend on the legacy ASSB training files.
It reads raw .mat/.csv/.parquet files and normalizes them to a standard
DataFrame used by later GV1 dataset builders.
"""

from .reader_base import ReadOptions, ReadResult, STANDARD_COLUMNS
from .auto_reader import read_battery_file, read_battery_files
from .field_mapper import standardize_dataframe, validate_standard_table
from .cache_writer import write_standard_cache

__all__ = [
    "ReadOptions",
    "ReadResult",
    "STANDARD_COLUMNS",
    "read_battery_file",
    "read_battery_files",
    "standardize_dataframe",
    "validate_standard_table",
    "write_standard_cache",
]
