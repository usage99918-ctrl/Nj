# Install and run the bot independently

The application runs entirely on your own machine or server. It needs no AI service, no exchange
account, no API keys, and no paid services. It uses public market data only and **simulates**
all trading.

**What "running continuously" requires.** A computer that stays on and connected. This includes
a Windows PC that doesn't sleep, a Linux server, or any host that runs Docker. The cloud coding
session where this was built **cannot** do it: processes there are stopped when the session
goes idle, as recorded in `docs/RESULTS.md`.

## Requirements

- Python 3.10 or newer (3.12 recommended). On Windows, install it from python.org and tick
  "Add python.exe to PATH".
- About 1 GB free disk space for live operation. Re-running the full research needs about 3 GB
  more.
- Outbound HTTPS to these hosts:
  - `data-api.binance.vision` (required)
  - `api.exchange.coinbase.com`, `api.kraken.com`, `www.okx.com`, `api.kucoin.com`,
    `api.gateio.ws`, `api.mexc.com`, `api-pub.bitfinex.com`, `www.bitstamp.net`
    (market scan and context)
  - `api.elections.kalshi.com` (Kalshi data)
  - `cointelegraph.com`, `decrypt.co`, `www.theblock.co` (news)

  If one host is unreachable, that part is skipped and the gap is recorded.

## Windows (recommended for your PC)

1. Copy or unzip the project folder, for example to `C:\CryptoLab`.
2. Open PowerShell in that folder and run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
   ```
   The script:
   - creates `.venv` and installs the dependencies;
   - checks the records, rebuilding the database from the ledger;
   - registers a Scheduled Task named **CryptoLab** that starts the supervisor when you log in
     and restarts it every minute if it stops;
   - starts the task and prints a health check.
3. Stop the PC from sleeping: **Settings → System → Power → Screen and sleep → "When plugged in,
   put my device to sleep after: Never"**.

   Restarts are handled like this:
   - **The worker crashes:** the supervisor restarts it within seconds.
   - **The supervisor stops:** Task Scheduler restarts it within a minute.
   - **The PC restarts:** the bot starts again at your next login. To start before anyone logs
     in, open Task Scheduler, edit the CryptoLab task, and choose "Run whether user is logged on
     or not". This needs your Windows password.
4. Daily use, from the project folder:
   ```powershell
   .venv\Scripts\python.exe -m cryptolab status          # experiments, continuity, worker health
   .venv\Scripts\python.exe -m cryptolab serve           # dashboard at http://127.0.0.1:8787/?token=...
   .venv\Scripts\python.exe -m cryptolab kill-switch on  # emergency stop of paper trading
   .venv\Scripts\python.exe -m cryptolab stop            # stop forecasting (the task restarts it at next login)
   ```
   To remove everything, run `deploy\uninstall_windows.ps1`. Your records are kept.

This installer was written for Windows 10 and 11 but **has not been executed on Windows** in
this build environment, which is Linux. The Python code avoids every Unix-only call:
file locks use `msvcrt`, liveness checks use the Windows API, and processes are started
detached. If a step fails, `deploy\start_windows.bat` starts the bot without Task Scheduler.

## Linux server or Raspberry Pi

```bash
bash deploy/install_linux.sh       # venv + systemd user service with Restart=always, starts at boot
journalctl --user -u cryptolab -f  # live logs
.venv/bin/python -m cryptolab status
```

## Docker (any cloud VM or NAS)

```bash
docker compose up -d     # restarts automatically after crashes and host reboots
docker compose logs -f cryptolab
```

The `dashboard` service listens on `127.0.0.1:8787` of the host.

Free hosting options exist: for example, the always-free compute tiers of the large cloud
providers. They require you to sign up, and some ask for a card for identity checks. **Nothing
is set up or paid for without your decision.**

## Updating an existing installation

Never reinstall over a running bot. Use the safe update script, which backs up and verifies
before changing anything and copies code only. See **[UPDATE_WINDOWS.md](UPDATE_WINDOWS.md)**
for the step-by-step guide. Self-check at any time: `python -m cryptolab doctor`.

## Moving your existing records

Copy these from the old machine: `data/ledger/`, `data/research/`, `reports/`, and `config/`,
if you changed it. The database rebuilds itself from the ledger, and `python -m cryptolab audit`
verifies the hash chain. **Never run two copies at once.** The ledger must have exactly one
writer.

## Phone access to the dashboard (secure)

`python -m cryptolab serve` listens on this computer only and requires a random access token.
It prints the token, which is also saved in `data/run/dashboard_token`. To reach it from your
phone **without exposing it to the internet**:

- **Tailscale (free personal plan):**
  1. Install Tailscale on the PC and on your phone, and sign in to both with the same account.
  2. Run `python -m cryptolab serve --host 0.0.0.0`.
  3. On the phone, open `http://<pc-tailscale-name>:8787/?token=<token>`.

  Traffic is end-to-end encrypted and is only visible to your own devices.
- **Cloudflare Tunnel (free):** run `cloudflared tunnel --url http://127.0.0.1:8787`. Keep the
  token secret; anyone with the link and token can view the dashboard.

Never forward the port on your router.

The dashboard is read-only, with one deliberate exception: an **Emergency stop paper trading**
button engages the kill switch. Releasing it is only possible on the host, with
`cryptolab kill-switch off`.

## Optional: AI summaries (paid, off by default)

1. Install the extra package: `pip install -r requirements-llm.txt`.
2. Provide Anthropic credentials, for example `ANTHROPIC_API_KEY`.
3. Set `enabled = true` under `[llm]` in a `config/local.toml`.

The model gets text only and has no tools, so it cannot trade. Leave it off to keep the bot
completely free.

## Verifying that it really runs

`python -m cryptolab status` shows the following, all computed from the tamper-evident ledger
rather than claimed:
- slot coverage per experiment;
- missed slots, with the reason for each;
- crashes that were restarted;
- network outages;
- the share of hours that have an hourly checkpoint.
