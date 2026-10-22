# DIPAI - local Gradio web chat (dark theme, electric red accent)
# Usage:
#   .\scripts\webui.ps1                    # http://127.0.0.1:7860
#   .\scripts\webui.ps1 -Port 7861 -Ckpt checkpoints\smoke
param(
    [string]$Ckpt = "checkpoints\scratch",
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$Share
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found: $Py (see WINDOWS-SETUP.md)"; exit 1 }
Push-Location $Root
try {
    $a = @("webui\app.py", "--ckpt", $Ckpt, "--host", $HostAddr, "--port", $Port)
    if ($Share) { $a += "--share" }
    & $Py @a
    exit $LASTEXITCODE
} finally { Pop-Location }
