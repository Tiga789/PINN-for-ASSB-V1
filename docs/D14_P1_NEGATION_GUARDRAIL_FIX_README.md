# D14-P1 negation-aware guardrail fix

This patch replaces only:

```text
scripts/gv1_d14_p1_generate_evidence_boundary_report.py
```

Purpose:

- Fix false-positive risky wording hits when the document is explicitly saying `not`, `do not`, `must not`, `不是`, `不能`, `不要`, `不应`, `不纳入`, `继续 flagged`, etc.
- Keep true risky claims detectable.
- Do not modify D9.6/D9.5.1 GV1 mainline files.
- Do not modify ASSB legacy files.
- Do not change D14-P1 run command.

Expected rerun result:

```text
overall_status = WARN
```

The remaining WARN is expected if D14-P0 status is WARN due to the harmless hard-clamp entry being present but default-off. A FAIL after this patch should be treated as a real wording issue and should be inspected manually.
