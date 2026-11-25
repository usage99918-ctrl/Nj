<#
CryptoLab ROLLBACK for Windows: put the PREVIOUS CODE back from a backup made by update_windows.ps1.

    powershell -ExecutionPolicy Bypass -File deploy\rollback_windows.ps1 -Backup "C:\CryptoLab-backups\before-update-20261125-153000"

Restores CODE ONLY (cryptolab\, deploy\, docs\, tests\, scripts\, config\default.toml, requirements*.txt, ...).
It never restores or deletes data\ (ledger, database, models, paper portfolios): records created
after the backup are kept. (Restoring data from a backup would erase newer records, so this script
refuses to do it; the backup folder still holds a full verified copy if you ever need one.)
Uses the EXISTING "CryptoLab" task only. Run it from any folder EXCEPT the backup itself.
#>
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Backup, [string]$InstallDir = "", [switch]$Yes)
$ErrorActionPreference = "Stop"
$TaskName = "CryptoLab"
$CodeDirs = @("cryptolab", "deploy", "docs", "tests", "scripts")
$CodeFiles = @("requirements.txt", "requirements-llm.txt", "pyproject.toml", "README.md", "Dockerfile",
               "docker-compose.yml", "config\default.toml")

function Get-BotProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match '-m\s+cryptolab\s+(run|worker|start|daemon)(\s|$)' })
}
function Invoke-Robocopy([string]$From, [string]$To, [string[]]$Extra) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & robocopy $From $To @Extra /R:2 /W:2 /NFL /NDL /NP /NJH /NJS | Out-Null; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ge 8) { throw "robocopy failed ($code) copying '$From' -> '$To'" }
}

$Backup = (Resolve-Path -LiteralPath $Backup).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $Backup "BACKUP_MANIFEST.sha256"))) { throw "'$Backup' is not a backup made by update_windows.ps1 (no BACKUP_MANIFEST.sha256)." }
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) { throw "No scheduled task named '$TaskName'. This script never creates tasks." }
if (-not $InstallDir) { $InstallDir = ($task.Actions | Select-Object -First 1).WorkingDirectory }
$InstallDir = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd('\')
$Py = Join-Path $InstallDir ".venv\Scripts\python.exe"
Write-Host "Restore CODE from : $Backup"
Write-Host "Into installation : $InstallDir   (data\, reports\, logs\, config\local.toml are NOT touched)"
if (-not $Yes) { if ((Read-Host "Type YES to continue") -ne "YES") { Write-Host "Cancelled."; exit 1 } }

$wasEnabled = ($task.State -ne "Disabled")
Disable-ScheduledTask -TaskName $TaskName | Out-Null
Push-Location $InstallDir
try {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Py -m cryptolab stop | Out-Host
    $ErrorActionPreference = $old
    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-BotProcesses).Count -gt 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 3 }
    if ((Get-BotProcesses).Count -gt 0) { throw "The bot did not stop; nothing was restored." }
    foreach ($d in $CodeDirs) { if (Test-Path -LiteralPath (Join-Path $Backup $d)) { Invoke-Robocopy (Join-Path $Backup $d) (Join-Path $InstallDir $d) @("/E") } }
    foreach ($f in $CodeFiles) { $s = Join-Path $Backup $f; if (Test-Path -LiteralPath $s) { Copy-Item -LiteralPath $s -Destination (Join-Path $InstallDir $f) -Force } }
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Py -m pip install -q -r requirements.txt | Out-Host
    & $Py -m cryptolab audit | Out-Null
    $auditOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $old
    Write-Host ("Previous code restored. Records audit: " + $(if ($auditOk) { "OK" } else { "PROBLEM - keep the backup and ask for help" }))
} finally {
    Pop-Location
    if ($wasEnabled) { Enable-ScheduledTask -TaskName $TaskName | Out-Null; Start-ScheduledTask -TaskName $TaskName; Write-Host "Task '$TaskName' re-enabled and started." }
}
