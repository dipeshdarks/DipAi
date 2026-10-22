# DIPAI - terminal chat with a trained scratch checkpoint
# Usage:
#   .\scripts\chat.ps1                                  # checkpoints\scratch
#   .\scripts\chat.ps1 -Ckpt checkpoints\smoke
#   .\scripts\chat.ps1 -Once "your prompt here"
param(
    [string]$Ckpt = "checkpoints\scratch",
    [string]$Once = "",
    [string]$Device = "auto"
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("-m", "dipai.chat", "--ckpt", $Ckpt, "--device", $Device)
    if ($Once -ne "") { $a += @("--once", $Once) }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
