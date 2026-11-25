#!/usr/bin/env bash
# Package the project (code, ledger, research results, reports, full git history) into one zip.
# Excludes regenerable data (raw history, caches, logs, working DB).
# Restore:  unzip cryptolab-bundle.zip && cd Nj && git log && python -m cryptolab audit
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-cryptolab-bundle.zip}"
rm -f "$out"
name="$(basename "$PWD")"
cd ..
zip -qr "$name/$out" "$name" \
  -x "$name/data/raw/*" "$name/data/cache/*" "$name/data/run/*" "$name/logs/*" \
     "$name/data/*.sqlite*" "$name/*__pycache__*" "$name/.pytest_cache/*" "$name/$out" "$name/*.zip"
cd "$name"
ls -la "$out"
