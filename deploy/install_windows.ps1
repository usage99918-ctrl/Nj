# Claude Crypto Research Lab - Windows installer (paper trading only, no real orders).
# Run in PowerShell from the project folder:
#   powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
# What it does: creates .venv, installs dependencies, audits the records, registers a
# Scheduled Task "CryptoLab" that starts the supervisor at logon and restarts it if it
# stops (every minute, indefinitely), then starts it now.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($py) { $base = "py -3" } else { $base = "python" }
Write-Host "Creating virtual environment in $Root\.venv ..."
Invoke-Expression "$base -m venv .venv"
& "$Root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "Checking records (rebuilds the database from the ledger if needed) ..."
& "$Root\.venv\Scripts\python.exe" -m cryptolab audit | Out-Null
$action = New-ScheduledTaskAction -Execute "$Root\.venv\Scripts\pythonw.exe" -Argument "-m cryptolab run" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -RestartCount 9999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
  -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "CryptoLab" -Action $action -Trigger $trigger -Settings $settings `
  -Description "Claude Crypto Research Lab supervisor (paper trading only)" -Force | Out-Null
Start-ScheduledTask -TaskName "CryptoLab"
Start-Sleep -Seconds 60
& "$Root\.venv\Scripts\python.exe" -m cryptolab health
Write-Host ""
Write-Host "Installed. The lab starts automatically when you log in and restarts after crashes."
Write-Host "Keep the PC awake: Settings > System > Power > Sleep = Never (when plugged in)."
Write-Host "Status:    .venv\Scripts\python.exe -m cryptolab status"
Write-Host "Dashboard: .venv\Scripts\python.exe -m cryptolab serve   (then open http://127.0.0.1:8787)"
Write-Host "Stop:      .venv\Scripts\python.exe -m cryptolab stop ; Disable-ScheduledTask -TaskName CryptoLab"
