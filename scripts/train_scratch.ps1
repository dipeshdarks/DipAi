# DIPAI - train the from-scratch model (Mode A)
# Usage:
#   .\scripts\train_scratch.ps1                      # config: configs\scratch.yaml
#   .\scripts\train_scratch.ps1 -Config configs\finetune.yaml
#   .\scripts\train_scratch.ps1 -NoResume -MaxIters 500
param(
    [string]$Config = "configs\scratch.yaml",
    [switch]$NoResume,
    [int]$MaxIters = -1
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("-m", "dipai.train", "--config", $Config)
    if ($NoResume) { $a += "--no-resume" }
    if ($MaxIters -gt 0) { $a += @("--max-iters", $MaxIters) }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
