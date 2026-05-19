param(
  [int[]]$Seeds = @(7,42,2026,3407,7890),
  [int]$MaxParallel = 4,
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$Device = "cuda",
  [string]$DType = "float32",
  [int]$Epochs = 2500,
  [int]$EvalEvery = 10,
  [int]$PrintEvery = 100,
  [int]$MonitorEverySeconds = 15,
  [switch]$Clean
)

# V7 compatibility wrapper.  The old parallel script used Start-Job/Receive-Job
# and could swallow Python tracebacks.  This wrapper intentionally delegates to
# the Start-Process implementation.
$script = Join-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) "run_assb112_guarded_seed_sweep_startprocess.ps1"
& $script -Seeds $Seeds -MaxParallel $MaxParallel -Root $Root -Python $Python -Device $Device -DType $DType -Epochs $Epochs -EvalEvery $EvalEvery -PrintEvery $PrintEvery -MonitorEverySeconds $MonitorEverySeconds -Clean:$Clean
exit $LASTEXITCODE
