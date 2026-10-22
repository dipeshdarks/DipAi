$log = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\logs\memory.log'
while ($true) {
    $t = Get-Date -Format 'HH:mm:ss'
    $g = nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
    $os = Get-CimInstance Win32_OperatingSystem
    $r = [math]::Round($os.FreePhysicalMemory / 1KB)
    $line = "$t | $g | RAMfree $r MB"
    Add-Content -LiteralPath $log -Value $line
    Start-Sleep 30
}
