$ErrorActionPreference = "Stop"
Write-Host "Verifying installed D18-S2 hotfix files..."
python scripts\d18_s2_hotfix_verify_manifest.py
if ($LASTEXITCODE -ne 0) { throw "D18-S2 hotfix manifest verification failed." }
Write-Host "Running D18-S2 hotfix fast resume. This does not start formal S2 training."
python scripts\d18_run_s2_hotfix_fast_resume.py
if ($LASTEXITCODE -ne 0) {
    throw "D18-S2 hotfix fast resume stopped. Inspect D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json."
}
