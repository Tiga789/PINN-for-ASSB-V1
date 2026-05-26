# GV1 time axis repair patch v2

This patch replaces only:

```text
gv1/io/field_mapper.py
```

Purpose:

- Some XJTU `.mat` files concatenate many MATLAB subrecords under `root.data[*]`.
- `system_time` can jump backwards across subrecords, causing `time_s is not monotonic nondecreasing`.
- For measured-current replay, this patch rebuilds a monotonic global `time_s` by preserving row order and using contiguous `raw__mat_subrecord_index` segments.
- It does not modify old ASSB mainline files.

Expected warning after successful repair:

```text
time_s rebuilt as monotonic global time from contiguous raw__mat_subrecord_index segments
```

Then `valid_standard_table` should become `true` for Batch-3 / Batch-4 XJTU `.mat` files.
