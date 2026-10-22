# DIPAI - Mode B fine-tune (LoRA / QLoRA) with VRAM pre-flight
# Usage:
#   .\scripts\finetune.ps1                      # dry run: prints VRAM estimate
#   .\scripts\finetune.ps1 -Train               # actually train
#   .\scripts\finetune.ps1 -Train -MaxSteps 10  # short smoke run
param(
    [string]$Config = "configs\finetune.yaml",
    [switch]$Train,
    [int]$MaxSteps = -1
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("-m", "dipai.finetune", "--config", $Config)
    if ($Train) { $a += "--train" }
    if ($MaxSteps -ge 0) { $a += @("--max-steps", $MaxSteps) }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
