# DIPAI - run the pytest suite (CPU-only, no GPU required)
# Usage:  .\scripts\test.ps1
#         .\scripts\test.ps1 -Filter tokenizer -Verbose
param(
    [string]$Filter = "",
    [switch]$Verbose
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("-m", "pytest", "tests", "-q")
    if ($Filter -ne "") { $a += @("-k", $Filter) }
    if ($Verbose) { $a += "-v" }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
