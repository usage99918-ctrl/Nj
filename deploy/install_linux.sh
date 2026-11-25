#!/usr/bin/env bash
# Claude Crypto Research Lab - Linux installer (systemd user service, restarts on crash and reboot).
# Usage: bash deploy/install_linux.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m cryptolab audit > /dev/null
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/cryptolab.service" <<UNIT
[Unit]
Description=Claude Crypto Research Lab (paper trading only)
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python -m cryptolab run
ExecStop=$ROOT/.venv/bin/python -m cryptolab stop
Restart=always
RestartSec=10
TimeoutStopSec=120

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now cryptolab.service
loginctl enable-linger "$USER" || echo "note: enable-linger failed; the service then runs only while you are logged in"
sleep 60
.venv/bin/python -m cryptolab health || true
echo "Installed. Logs: journalctl --user -u cryptolab -f   |   status: .venv/bin/python -m cryptolab status"
