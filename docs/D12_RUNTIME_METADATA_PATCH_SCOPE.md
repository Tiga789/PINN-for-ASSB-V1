# D12 Runtime Metadata Model Patch Scope

This package implements the runtime backend requested by the D12 plan.

It is **separate and opt-in**. It does not overwrite D9.6/D9.5.1 source files. The new wrapper registers a process-local patch and then delegates to the existing D9 trainer.

## Modes

```text
off   strict D9.6/D9.5.1 reference; no metadata dimensions appended
zero  architecture-control; append same metadata schema but all values are zero
on    append D11C2/D12 metadata values
```

## Guardrails

- Battery-8 remains target-probe only unless explicitly overridden.
- No 24-profile 200ks mainline command is generated.
- The package is for D12 ablation, not mainline replacement.

## First command

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_runtime_patch_guardrail.ps1"
```
