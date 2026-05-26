# GV1 streaming replay profile patch

This patch fixes ArrowMemoryError during `scripts/gv1_generate_softlabels.py` on large XJTU Parquet caches.

## Files

- `scripts/gv1_generate_softlabels.py`
- `gv1/pipeline/data_loader.py`
- `gv1/io/parquet_reader.py`

## What changed

1. Reads one indexed source file at a time instead of concatenating all 24 files into one DataFrame.
2. Reads only replay-required Parquet columns instead of raw debug columns.
3. Releases each large DataFrame after writing its replay profile.
4. Keeps the old single-file smoke output behavior when `--max_files 1` is used.

No ASSB mainline files are modified.
