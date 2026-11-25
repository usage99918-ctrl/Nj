#!/usr/bin/env bash
# Package the project (code, ledger, research results, reports, full git history) into one zip.
# Excludes regenerable data (raw history, caches, logs, working DB).
# Restore:  unzip cryptolab-bundle.zip && cd Nj && git log && python -m cryptolab audit
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
name="$(basename "$root")"
out="$(realpath -m "${1:-$root/cryptolab-bundle.zip}")"
rm -f "$out"
(cd .. && zip -qr "$out" "$name" \
  -x "$name/data/raw/*" "$name/data/cache/*" "$name/data/run/*" "$name/logs/*" \
     "$name/data/*.sqlite*" "*__pycache__*" "$name/.pytest_cache/*" "$name/*.zip")
ls -la "$out"
