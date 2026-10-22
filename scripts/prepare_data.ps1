# DIPAI - prepare raw data into packed train/val bins + tokenizer
# Usage:
#   .\scripts\prepare_data.ps1                                      # whole data/raw dir
#   .\scripts\prepare_data.ps1 -InputPaths data\raw\tinystories_50k.txt  # specific file
#   .\scripts\prepare_data.ps1 -VocabSize 4096 -ReuseTokenizer
param(
    [string[]]$InputPaths = @("data\raw"),
    [int]$VocabSize = 4096,
    [int]$BlockSize = 256,
    [switch]$ReuseTokenizer,
    [string]$OutputDir = "data\processed"
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("scripts\prepare_data.py", "--input") + $InputPaths +
         @("--vocab-size", $VocabSize, "--block-size", $BlockSize, "--output-dir", $OutputDir)
    if ($ReuseTokenizer) { $a += "--reuse-tokenizer" }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
