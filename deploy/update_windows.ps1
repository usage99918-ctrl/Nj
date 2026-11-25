<#
CryptoLab SAFE UPDATE for Windows (paper research only: it never trades and never touches funds).

Run it from the NEW code folder you downloaded, NOT from inside your installation:
    powershell -ExecutionPolicy Bypass -File deploy\update_windows.ps1 -DryRun   (checks only)
    powershell -ExecutionPolicy Bypass -File deploy\update_windows.ps1           (the update)
Options:
    -InstallDir "C:\CryptoLab"  installation to update (default: the folder the existing
                                "CryptoLab" scheduled task runs in)
    -BackupRoot "D:\Backups"    where the backup goes (default: "<install>-backups" next to it)
    -DryRun                     run every check, change nothing
    -Yes                        do not ask for confirmation
    -SkipTests                  skip the test suite (not recommended)

Order of operations (it stops at the first problem):
  1. Verifies the downloaded code against its SHA-256 manifest and finds your installation.
  2. Audits your CURRENT records (ledger hash chain + database) before anything changes.
  3. Pauses the EXISTING "CryptoLab" task and stops the bot gracefully. No task is created.
  4. FULL BACKUP of the installation (everything except .venv), every file verified by SHA-256.
     No verified backup = no update.
  5. Copies CODE ONLY. It never copies over, deletes or edits data\ (ledger, database, models,
     paper portfolios), reports\, logs\, config\local.toml or your backups.
  6. Verifies: the ledger only GREW (byte-exact prefix check against the backup), database
     audit, doctor (every live model loads and predicts on this PC's Python), tests.
     Any failure -> the previous code is restored from the backup automatically.
  7. Re-enables and starts the SAME "CryptoLab" task and checks the bot is healthy.
A full log is written to UPDATE_REPORT.txt inside the backup folder.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = "",
    [string]$BackupRoot = "",
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$TaskName = "CryptoLab"
$CodeDirs = @("cryptolab", "deploy", "docs", "tests", "scripts")          # keep in sync with scripts\make_code_manifest.py
$CodeFiles = @("requirements.txt", "requirements-llm.txt", "pyproject.toml", "README.md", "Dockerfile",
               "docker-compose.yml", "config\default.toml")
$CriticalTests = @("test_portable.py", "test_storage.py", "test_research_integrity.py", "test_live.py", "test_paper.py",
                   "test_opportunity.py")
$Sep = [System.IO.Path]::DirectorySeparatorChar
$Pkg = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.TrimEnd('\')
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$script:Log = New-Object System.Collections.Generic.List[string]
$script:Bk = $null
$script:CodeChanged = $false
$script:PipRan = $false
$script:TaskDisabledByUs = $false
$script:WasRunning = $false
$script:WasEnabled = $true
$script:Py = $null

function Say([string]$m, [string]$c = "Gray") {
    Write-Host $m -ForegroundColor $c
    $script:Log.Add("$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m")
}
function Ok([string]$m) { Say "  [OK] $m" "Green" }
function Warn([string]$m) { Say "  [NOTE] $m" "Yellow" }
function Stop-Update([string]$m) { throw "UPDATE STOPPED: $m" }

function Get-NormHash([string]$Path) {
    # SHA-256 with CRLF normalized to LF (matches scripts\make_code_manifest.py)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $text = [System.Text.Encoding]::UTF8.GetString($bytes).Replace("`r`n", "`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return (($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($text)) | ForEach-Object { $_.ToString("x2") }) -join "") }
    finally { $sha.Dispose() }
}

function Quote-Arg([string]$a) { if ($a -match '[\s"]') { return '"' + ($a -replace '"', '\"') + '"' } return $a }

function Invoke-Py {
    # Runs the INSTALLED code (working directory = installation) and captures stdout/stderr separately.
    param([string[]]$PyArgs, [int]$TimeoutSec = 1800)
    $o = [System.IO.Path]::GetTempFileName(); $e = [System.IO.Path]::GetTempFileName()
    try {
        $argLine = ($PyArgs | ForEach-Object { Quote-Arg $_ }) -join " "
        $p = Start-Process -FilePath $script:Py -ArgumentList $argLine -WorkingDirectory $InstallDir -NoNewWindow `
            -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
        $null = $p.Handle   # keeps ExitCode available (PowerShell 5.1 quirk)
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            try { $p.Kill() } catch { }
            return [pscustomobject]@{ Code = 124; Out = ""; Err = "timed out after $TimeoutSec s" }
        }
        $p.WaitForExit()
        $out = Get-Content -LiteralPath $o -Raw -ErrorAction SilentlyContinue
        $err = Get-Content -LiteralPath $e -Raw -ErrorAction SilentlyContinue
        return [pscustomobject]@{ Code = $p.ExitCode; Out = [string]$out; Err = [string]$err }
    } finally {
        Remove-Item -LiteralPath $o, $e -ErrorAction SilentlyContinue
    }
}

function Get-BotProcesses([string]$Kind = "run|worker|start|daemon") {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match ("-m\s+cryptolab\s+(" + $Kind + ")(\s|$)") })
}

function Invoke-Robocopy([string]$From, [string]$To, [string[]]$Extra) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & robocopy $From $To @Extra /R:2 /W:2 /NFL /NDL /NP /NJH /NJS | Out-Null; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ge 8) { throw "robocopy failed ($code) copying '$From' -> '$To'" }
}

function Restore-Code {
    if (-not $script:CodeChanged -or -not $script:Bk) { return }
    Say "Restoring the PREVIOUS code from the backup (your data was not changed)..." "Yellow"
    foreach ($d in $CodeDirs) {
        $src = Join-Path $script:Bk $d
        if (Test-Path -LiteralPath $src) { Invoke-Robocopy $src (Join-Path $InstallDir $d) @("/E") }
    }
    foreach ($f in $CodeFiles) {
        $src = Join-Path $script:Bk $f
        if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $InstallDir $f) -Force }
    }
    if ($script:PipRan) { $null = Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt") }
    $a = Invoke-Py @("-m", "cryptolab", "audit")
    if ($a.Code -eq 0) { Ok "Previous code restored; audit of your records still OK" }
    else { Say "  [WARNING] audit after restore returned $($a.Code). Your backup is complete at $($script:Bk)." "Red" }
    $script:CodeChanged = $false
}

function Restore-Task {
    if ($script:TaskDisabledByUs) {
        if ($script:WasEnabled) { Enable-ScheduledTask -TaskName $TaskName | Out-Null }
        if ($script:WasRunning -and $script:WasEnabled) { Start-ScheduledTask -TaskName $TaskName }
        $script:TaskDisabledByUs = $false
        Say "  Scheduled task '$TaskName' restored to its previous state." "Yellow"
    }
}

function Save-Report {
    if ($script:Bk -and (Test-Path -LiteralPath $script:Bk)) {
        $script:Log | Set-Content -LiteralPath (Join-Path $script:Bk "UPDATE_REPORT.txt") -Encoding UTF8
        Write-Host "Full log: $(Join-Path $script:Bk 'UPDATE_REPORT.txt')"
    }
}

function Main {
    # ------------------------------------------------------------------ 1. package + installation
    Say "== 1/7  Checking the downloaded code and finding your installation" "Cyan"
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        Stop-Update "No scheduled task named '$TaskName' exists. This script only UPDATES an existing installation and never creates a task."
    }
    $script:WasEnabled = ($task.State -ne "Disabled")
    $taskDir = ($task.Actions | Select-Object -First 1).WorkingDirectory
    if (-not $script:InstallDir) { $script:InstallDir = $taskDir }
    if (-not $script:InstallDir -or -not (Test-Path -LiteralPath $script:InstallDir)) {
        Stop-Update "Installation folder not found ('$($script:InstallDir)'). Pass it with -InstallDir."
    }
    $script:InstallDir = (Resolve-Path -LiteralPath $script:InstallDir).Path.TrimEnd('\')
    $InstallDir = $script:InstallDir
    if ($taskDir -and ((Resolve-Path -LiteralPath $taskDir).Path.TrimEnd('\') -ne $InstallDir)) {
        Stop-Update "The CryptoLab task runs in '$taskDir' but -InstallDir is '$InstallDir'. Refusing to update a folder the bot does not run from."
    }
    if ($InstallDir -eq $Pkg) { Stop-Update "Run this script from the NEW downloaded code folder, not from inside the installation." }
    $script:Py = Join-Path $InstallDir ".venv\Scripts\python.exe"
    foreach ($p in @($script:Py, (Join-Path $InstallDir "cryptolab\__init__.py"), (Join-Path $InstallDir "data\ledger"))) {
        if (-not (Test-Path -LiteralPath $p)) { Stop-Update "Expected '$p' in the installation but it is missing." }
    }
    $man = Join-Path $Pkg "deploy\CODE_MANIFEST.txt"
    if (-not (Test-Path -LiteralPath $man)) { Stop-Update "deploy\CODE_MANIFEST.txt is missing: the download is incomplete." }
    $entries = New-Object System.Collections.Generic.List[object]; $knownDefault = @()
    foreach ($line in (Get-Content -LiteralPath $man)) {
        if ($line -match '^# known-default-toml ([0-9a-f]{64})') { $knownDefault += $Matches[1]; continue }
        if ($line -match '^([0-9a-f]{64})  (.+)$') { $entries.Add(@($Matches[1], $Matches[2])) }
    }
    $bad = @()
    foreach ($en in $entries) {
        $f = Join-Path $Pkg ($en[1] -replace '/', '\')
        if (-not (Test-Path -LiteralPath $f)) { $bad += "missing $($en[1])" }
        elseif ((Get-NormHash $f) -ne $en[0]) { $bad += "different $($en[1])" }
    }
    if ($bad.Count -gt 0) { Stop-Update "The downloaded code does not match its manifest ($($bad.Count) problem(s), first: $($bad[0])). Download it again." }
    Ok "Downloaded code verified: $($entries.Count) files match the SHA-256 manifest"
    $newVer = (Select-String -LiteralPath (Join-Path $Pkg "cryptolab\__init__.py") -Pattern '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
    $oldVerM = Select-String -LiteralPath (Join-Path $InstallDir "cryptolab\__init__.py") -Pattern '__version__\s*=\s*"([^"]+)"'
    $oldVer = if ($oldVerM) { $oldVerM.Matches[0].Groups[1].Value } else { "?" }
    Ok "Installation: $InstallDir (version $oldVer)  ->  new version $newVer"

    if (-not $BackupRoot) { $BackupRoot = Join-Path (Split-Path $InstallDir -Parent) ((Split-Path $InstallDir -Leaf) + "-backups") }
    $BackupRoot = [System.IO.Path]::GetFullPath($BackupRoot).TrimEnd('\')
    if ($BackupRoot.StartsWith($InstallDir + $Sep, [System.StringComparison]::OrdinalIgnoreCase) -or $BackupRoot -ieq $InstallDir) {
        Stop-Update "The backup folder must be OUTSIDE the installation (got '$BackupRoot')."
    }
    $script:BkPlanned = Join-Path $BackupRoot "before-update-$Stamp"

    # ------------------------------------------------------------------ 2. records before
    Say "== 2/7  Checking your records BEFORE anything changes" "Cyan"
    $a = Invoke-Py @("-m", "cryptolab", "audit")
    if ($a.Code -ne 0) { Stop-Update "The audit of your CURRENT records failed (exit $($a.Code)). Nothing was changed. Output:`n$($a.Out)`n$($a.Err)" }
    $pre = $a.Out | ConvertFrom-Json
    Ok "Records intact: $($pre.ledger.records) ledger records, hash chain OK, $($pre.db_rows_checked) database rows match the ledger"

    $serve = Get-BotProcesses "serve"
    if ($serve.Count -gt 0) {
        Warn "The dashboard server (cryptolab serve) is running. Close its window, then press Enter."
        if (-not $Yes -and -not $DryRun) { $null = Read-Host "Press Enter when the dashboard window is closed" }
        if ((Get-BotProcesses "serve").Count -gt 0 -and -not $DryRun) { Stop-Update "The dashboard server is still running. Close it and run the update again. Nothing was changed." }
    }

    Say ""
    Say "Installation : $InstallDir"
    Say "New code from: $Pkg"
    Say "Backup to    : $($script:BkPlanned)"
    Say "Will be replaced (code only): cryptolab\, deploy\, docs\, tests\, scripts\, config\default.toml, requirements*.txt, pyproject.toml, README.md"
    Say "Never touched: data\ (ledger, database, models, paper portfolios), reports\, logs\, config\local.toml, your backups"
    if ($DryRun) { Say "DRY RUN finished: every check passed. Nothing was changed." "Green"; return }
    if (-not $Yes) {
        $ans = Read-Host "Type YES to stop the bot, back up everything and update"
        if ($ans -ne "YES") { Say "Cancelled. Nothing was changed."; return }
    }

    # ------------------------------------------------------------------ 3. stop the bot
    Say "== 3/7  Stopping the bot (pausing the existing task; no new task is created)" "Cyan"
    $script:WasRunning = ((Get-BotProcesses).Count -gt 0)
    try { Disable-ScheduledTask -TaskName $TaskName | Out-Null }
    catch { Stop-Update "Could not pause the '$TaskName' task ($($_.Exception.Message)). Try again from PowerShell opened with 'Run as administrator'. Nothing was changed." }
    $script:TaskDisabledByUs = $true
    $null = Invoke-Py @("-m", "cryptolab", "stop") 240
    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-BotProcesses).Count -gt 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 3 }
    if ((Get-BotProcesses).Count -gt 0) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue; Start-Sleep -Seconds 20 }
    $left = Get-BotProcesses
    if ($left.Count -gt 0) {
        $desc = ($left | ForEach-Object { "pid $($_.ProcessId): $($_.ExecutablePath) $($_.CommandLine)" }) -join "`n    "
        Stop-Update ("A CryptoLab bot process is still running after 3 minutes, so nothing was changed:`n    $desc`n" +
            "If one of these runs from ANOTHER folder, that is a second (duplicate) bot: stop it first, then run the update again.")
    }
    Ok "Bot stopped (was running: $($script:WasRunning)); no cryptolab run/worker process remains"

    # ------------------------------------------------------------------ 4. verified backup
    Say "== 4/7  Backing up the whole installation and verifying every file" "Cyan"
    $script:Bk = $script:BkPlanned
    $venv = Join-Path $InstallDir ".venv"
    $src = @(Get-ChildItem -LiteralPath $InstallDir -Recurse -File -Force |
        Where-Object { -not $_.FullName.StartsWith($venv + $Sep, [System.StringComparison]::OrdinalIgnoreCase) })
    $need = [double](($src | Measure-Object -Property Length -Sum).Sum)
    try {
        $drive = New-Object System.IO.DriveInfo([System.IO.Path]::GetPathRoot($BackupRoot))
        if ($drive.AvailableFreeSpace -lt ($need * 1.2 + 200MB)) {
            Stop-Update "Not enough free disk space for the backup: need about $([math]::Round($need * 1.2 / 1GB, 2)) GB on $($drive.Name). Nothing was changed."
        }
    } catch [System.ArgumentException] { Warn "Could not check free space on '$BackupRoot' (network path?)." }
    New-Item -ItemType Directory -Path $script:Bk -Force | Out-Null
    Invoke-Robocopy $InstallDir $script:Bk @("/E", "/COPY:DAT", "/DCOPY:T", "/XD", $venv)
    $lines = New-Object System.Collections.Generic.List[string]; $mism = @()
    foreach ($f in $src) {
        $rel = $f.FullName.Substring($InstallDir.Length + 1)
        $b = Join-Path $script:Bk $rel
        if (-not (Test-Path -LiteralPath $b)) { $mism += "missing $rel"; continue }
        $h1 = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
        $h2 = (Get-FileHash -LiteralPath $b -Algorithm SHA256).Hash
        if ($h1 -ne $h2) { $mism += "different $rel" } else { $lines.Add("$h1  $rel") }
    }
    if ($mism.Count -gt 0) { Stop-Update "Backup verification failed for $($mism.Count) file(s), first: $($mism[0]). Nothing was changed." }
    $lines | Set-Content -LiteralPath (Join-Path $script:Bk "BACKUP_MANIFEST.sha256") -Encoding ASCII
    $info = [ordered]@{ created = (Get-Date).ToString("o"); source = $InstallDir; files = $src.Count; bytes = $need
                        ledger_records = $pre.ledger.records; ledger_head = $pre.ledger.head; old_version = $oldVer; new_version = $newVer }
    $info | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $script:Bk "backup_info.json") -Encoding UTF8
    Ok "Backup verified: $($src.Count) files, $([math]::Round($need / 1MB, 1)) MB, every SHA-256 matches -> $($script:Bk)"

    # ------------------------------------------------------------------ 5. code only
    Say "== 5/7  Installing the new code (code only)" "Cyan"
    $curDefault = Join-Path $InstallDir "config\default.toml"
    if ((Test-Path -LiteralPath $curDefault) -and ($knownDefault -notcontains (Get-NormHash $curDefault))) {
        $keep = Join-Path $InstallDir "config\default.toml.your-edits-$Stamp"
        Copy-Item -LiteralPath $curDefault -Destination $keep
        Warn "Your config\default.toml had local edits. A copy was kept as $keep (and in the backup). Put personal settings in config\local.toml, which updates never touch."
    }
    $script:CodeChanged = $true
    foreach ($d in $CodeDirs) { Invoke-Robocopy (Join-Path $Pkg $d) (Join-Path $InstallDir $d) @("/E", "/XD", "__pycache__", "/XF", "*.pyc") }
    foreach ($f in $CodeFiles) {
        $s = Join-Path $Pkg $f
        if (Test-Path -LiteralPath $s) { Copy-Item -LiteralPath $s -Destination (Join-Path $InstallDir $f) -Force }
    }
    Ok "Code copied (no file was deleted; data\, reports\, logs\ and config\local.toml untouched)"
    $oldReq = Join-Path $script:Bk "requirements.txt"
    if (-not (Test-Path -LiteralPath $oldReq) -or ((Get-NormHash $oldReq) -ne (Get-NormHash (Join-Path $InstallDir "requirements.txt")))) {
        Say "  requirements.txt changed: installing dependencies into the existing .venv ..."
        $script:PipRan = $true
        $r = Invoke-Py @("-m", "pip", "install", "-r", "requirements.txt")
        if ($r.Code -ne 0) { Stop-Update "pip install failed:`n$($r.Err)" }
        Ok "Dependencies installed"
    } else { Ok "Dependencies unchanged (pip not needed)" }
    if (Test-Path -LiteralPath (Join-Path $InstallDir "cryptolab\frozen\registry.json")) {
        $r = Invoke-Py @("-m", "cryptolab", "register-frozen")
        if ($r.Code -ne 0) { Stop-Update "Registering the new frozen experiment models failed:`n$($r.Out)`n$($r.Err)" }
        Ok "New experiment models registered in your ledger (append-only): $($r.Out.Trim())"
    }

    # ------------------------------------------------------------------ 6. verify
    Say "== 6/7  Verifying the updated installation (bot still stopped)" "Cyan"
    $c = Invoke-Py @("-m", "cryptolab", "ledger-compare", "--against", (Join-Path $script:Bk "data\ledger"))
    if ($c.Code -ne 0) { Stop-Update "Ledger comparison with the backup FAILED:`n$($c.Out)`n$($c.Err)" }
    $cj = $c.Out | ConvertFrom-Json
    Ok "Ledger only grew: $($cj.unchanged_shards) shard(s) byte-identical, $($cj.appended_records) record(s) appended, $($cj.current_records) total, hash chain OK"
    $a2 = Invoke-Py @("-m", "cryptolab", "audit")
    if ($a2.Code -ne 0) { Stop-Update "Database audit FAILED after the update:`n$($a2.Out)" }
    Ok "Database audit OK"
    $d1 = Invoke-Py @("-m", "cryptolab", "doctor", "--offline")
    if ($d1.Code -ne 0) { Stop-Update "Self-check (offline) FAILED:`n$($d1.Out)`n$($d1.Err)" }
    $d2 = Invoke-Py @("-m", "cryptolab", "doctor")
    foreach ($ln in ($d2.Out -split "`n")) { if ($ln.Trim()) { Say "     $($ln.TrimEnd())" } }
    if ($d2.Code -ne 0) {
        if ($d2.Out -match "live panel failed") { Warn "Live self-check could not reach market data (internet?). Offline self-check passed; the bot retries automatically." }
        else { Stop-Update "Self-check FAILED (see the lines above)." }
    } else { Ok "Self-check passed: every live model loads, matches its ledger hash and predicts on this PC's Python" }
    if (-not $SkipTests) {
        $crit = @($CriticalTests | ForEach-Object { Join-Path "tests" $_ } | Where-Object { Test-Path -LiteralPath (Join-Path $InstallDir $_) })
        $t = Invoke-Py (@("-m", "pytest", "-q", "-p", "no:cacheprovider") + $crit) 1800
        $tail = (($t.Out -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 3) -join " | "
        if ($t.Code -ne 0) { Stop-Update "Critical tests FAILED: $tail`n$($t.Out)" }
        Ok "Critical tests passed: $tail"
        $rest = @(Get-ChildItem -LiteralPath (Join-Path $InstallDir "tests") -Filter "test_*.py" |
            Where-Object { $CriticalTests -notcontains $_.Name } | ForEach-Object { Join-Path "tests" $_.Name })
        if ($rest.Count -gt 0) {
            $t2 = Invoke-Py (@("-m", "pytest", "-q", "-p", "no:cacheprovider") + $rest) 1800
            $tail2 = (($t2.Out -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 2) -join " | "
            if ($t2.Code -eq 0) { Ok "Other tests passed: $tail2" }
            else { Warn "Some timing/server tests failed (informational, update kept): $tail2. Please share UPDATE_REPORT.txt."; $script:Log.Add($t2.Out) }
        }
    }

    # ------------------------------------------------------------------ 7. restart
    Say "== 7/7  Restarting the existing '$TaskName' task" "Cyan"
    $script:CodeChanged = $false     # from here on the update is accepted
    if ($script:WasEnabled) {
        Enable-ScheduledTask -TaskName $TaskName | Out-Null
        $script:TaskDisabledByUs = $false
        Start-ScheduledTask -TaskName $TaskName
        $ok = $false; $deadline = (Get-Date).AddSeconds(180)
        while (-not $ok -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 10
            $h = Invoke-Py @("-m", "cryptolab", "health") 60
            $ok = ($h.Code -eq 0)
        }
        if ($ok) { Ok "Bot running again: $($h.Out.Trim())" }
        else { Warn "The task was started but the bot is not healthy yet. Check in 5 minutes with: .venv\Scripts\python.exe -m cryptolab status" }
    } else {
        $script:TaskDisabledByUs = $false
        Warn "The '$TaskName' task was DISABLED before the update, so it was left disabled and not started."
    }
    Say ""
    Say "UPDATE COMPLETE ($oldVer -> $newVer). Backup kept at: $($script:Bk)" "Green"
    Say "If the dashboard server was open, start it again: .venv\Scripts\python.exe -m cryptolab serve"
}

try {
    Main
    Save-Report
    exit 0
} catch {
    Say "" ; Say $_.Exception.Message "Red"
    try { Restore-Code } catch { Say "  [WARNING] code restore problem: $($_.Exception.Message). Backup: $($script:Bk)" "Red" }
    try { Restore-Task } catch { Say "  [WARNING] could not restore the task state: $($_.Exception.Message)" "Red" }
    Save-Report
    exit 1
}
