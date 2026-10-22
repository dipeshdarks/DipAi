# DIPAI - build data/processed/finetune.jsonl for Mode B (LoRA/QLoRA)
# Usage:  .\scripts\prepare_finetune_data.ps1 -Limit 2000
param(
    [string[]]$InputPaths = @("data\raw\tinystories_50k.txt"),
    [string]$Output = "data\processed\finetune.jsonl",
    [int]$Limit = 2000
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    & $Py scripts\prepare_finetune_data.py --input @InputPaths --output $Output --limit $Limit
    exit $LASTEXITCODE
} finally { Pop-Location }
