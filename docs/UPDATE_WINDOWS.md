# Updating the bot on your Windows PC

This update fixes two errors on your PC:
- the 60-minute experiments (B_60min, G2_60min) failed with "all keys need to be the same shape";
- the 4-hour experiment (G2_240min) failed with "no locals found when setting up annotations".

The 24-hour experiment (G2_1440min) would fail the same way, because it uses the same kind of
model file. All the update does to your existing data is add new records.

## What the update changes

| Changed (code only) | Never touched |
|---|---|
| `cryptolab\` program code, `deploy\`, `docs\`, `tests\`, `scripts\`, `config\default.toml`, `requirements*.txt`, `README.md`, `pyproject.toml` | `data\`: ledger, database, model files, paper portfolios. Also `reports\`, `logs\`, `config\local.toml`, and your backups |

The script never creates a second scheduled task and never starts a second bot. It pauses your
existing **CryptoLab** task, updates, then starts that same task again.

## Why the errors happened

Three live models are simple formula rules: 60-minute, 4-hour and 24-hour. Their model files
stored each rule as Python 3.13 bytecode, and your PC runs Python 3.12. Bytecode only works on the
Python version that created it.

The fixed code rebuilds each rule from the bot's own source code instead of running the stored
bytecode. Model files and their fingerprints are unchanged, so your history stays valid. On the
version that created the files, the rebuilt rules are byte-identical to the stored ones. On 3,000
historical rows, Python 3.12 and 3.13 now give bit-identical scores and the same coin choices.

## Step by step

**Before you start:** keep the PC plugged in. If the dashboard window (`cryptolab serve`) is
open, close it.

**1. Download the new code.** Put it in your Downloads folder, **not** your bot folder. Open
**Windows PowerShell** (Start menu → type `PowerShell` → open it), then paste these lines and press
Enter:

```powershell
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$dl = "$env:USERPROFILE\Downloads\CryptoLab-update"
Invoke-WebRequest -UseBasicParsing "https://github.com/usage99918-ctrl/Nj/archive/refs/heads/claude/crypto-forecasting-system-58d48l.zip" -OutFile "$dl.zip"
Expand-Archive "$dl.zip" -DestinationPath $dl -Force
Set-Location (Get-ChildItem $dl -Directory | Select-Object -First 1).FullName
```

You can also download the ZIP in a browser from the GitHub page (green **Code** button →
**Download ZIP**), extract it, and open PowerShell in the extracted folder.

**2. Do a dry run.** It checks everything and changes nothing:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\update_windows.ps1 -DryRun
```

It finds your installation through the CryptoLab task and checks that the download is complete
(SHA-256 of every file). It also checks your current records. It should end with
`DRY RUN finished: every check passed. Nothing was changed.`

**3. Run the update.**

```powershell
powershell -ExecutionPolicy Bypass -File deploy\update_windows.ps1
```

Type `YES` when asked. The script then:

1. stops the bot;
2. makes a **full backup** next to your bot folder, for example
   `C:\CryptoLab-backups\before-update-20261125-153000`, and verifies every file in it by
   SHA-256. **No verified backup means no update.**
3. installs the code;
4. checks that the ledger only grew, with every old record byte-identical to the backup;
5. audits the database;
6. runs `doctor`, which loads every live model and makes a test prediction **on your Python**
   without recording anything;
7. runs the tests;
8. restarts the CryptoLab task and checks that the bot is healthy.

If any check fails after the code was copied, it **restores the previous code automatically**
and restarts the old version.

**4. Check the result yourself.** In your bot folder:

```powershell
cd C:\CryptoLab                                   # your bot folder
.venv\Scripts\python.exe -m cryptolab doctor      # every check should say PASS
.venv\Scripts\python.exe -m cryptolab status
```

What you should see in the dashboard. The times are UTC; Toronto = UTC − 5 h in winter, Vancouver
= UTC − 8 h.
- **60-minute experiments:** a forecast at the next full hour.
- **4-hour experiment:** a forecast at the next 00:00, 04:00, 08:00, 12:00, 16:00 or 20:00 UTC.
- **24-hour experiment:** a forecast at the next 00:00–03:00 UTC window. On about 3 days in 100
  the 24-hour rule finds no signal. That day it records a gap that says "model abstained". This
  was a rule fixed before testing.
- Slots missed while the old code was failing stay recorded as gaps with their error. They are
  never deleted or back-filled.

## If something goes wrong

- The update script restores the previous code automatically and tells you the backup folder.
  The full log is in `UPDATE_REPORT.txt` inside that folder.
- Manual rollback of the code (your data is kept):
  ```powershell
  powershell -ExecutionPolicy Bypass -File deploy\rollback_windows.ps1 -Backup "C:\CryptoLab-backups\before-update-..."
  ```
- If PowerShell says it cannot pause the task, open PowerShell with **Run as administrator** and
  run the same command again.

## How this was tested and what was not tested

**Tested on Linux with PowerShell 7.4, using stand-ins for Task Scheduler, robocopy and the
Windows process list.** The test installation ran your PC's old code on **Python 3.12**. It first
reproduced the 4-hour error live. After the update, the same bot made the 4-hour and 60-minute
forecasts that had failed.

The failure path was also tested. A deliberately broken release was caught. The previous code was
restored, the records audit was still OK, and the old bot was restarted.

**Not tested:** a real Windows PC. Windows PowerShell 5.1, the real Task Scheduler and robocopy
have not run this script yet. That is why the dry run comes first: it reports a problem before
anything changes.

## Letting Claude work on your PC directly (optional)

A Claude Code session in the cloud cannot see your PC by itself. There are two ways:

- **Remote Control:** in the Claude desktop app open **Settings → Claude Code → Remote Control**
  and share your bot folder. A shared folder then appears to the cloud session, which can ask to
  use it. You approve each request. That page may not offer a way to add folders. Some folders
  are also not usable from a cloud session (your home folder itself, hidden folders, `~\Library`).
- **A session on your PC:** start a Claude Code session in the desktop app that runs on your
  computer, in your bot folder. It starts without this conversation's history, so tell it:
  *"Read docs\UPDATE_WINDOWS.md from the downloaded update folder and follow it. Make the backup
  first. Never delete or overwrite anything in data\."*

You can also run the steps above yourself. The bot never needs Claude to stay open.
