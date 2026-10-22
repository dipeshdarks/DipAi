# DIPAI - hardware / environment check (no GPU work, safe to run anytime)
# Usage:  .\scripts\check_hardware.ps1
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try { & $Py scripts\check_hardware.py $args; exit $LASTEXITCODE } finally { Pop-Location }
