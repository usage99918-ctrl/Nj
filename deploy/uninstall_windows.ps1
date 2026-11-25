# Stops the lab and removes the Scheduled Task. Records in data\ are NOT deleted.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& "$Root\.venv\Scripts\python.exe" -m cryptolab stop
Unregister-ScheduledTask -TaskName "CryptoLab" -Confirm:$false
Write-Host "Removed the CryptoLab task. Your records in data\ are untouched."
