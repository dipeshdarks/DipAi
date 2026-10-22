# DIPAI - download training corpora (streaming, low RAM)
#   1) TinyStories 50k stories -> data/raw/tinystories_50k.txt
#   2) OASST1 English messages -> data/raw/oasst1_english.jsonl
# Usage:  .\scripts\download_data.ps1
#         .\scripts\download_data.ps1 -SkipOasst
param([switch]$SkipTinyStories, [switch]$SkipOasst)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    if (-not $SkipTinyStories) {
        & $Py scripts\download_training_data.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (-not $SkipOasst) {
        & $Py scripts\download_oasst.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    exit 0
} finally { Pop-Location }
