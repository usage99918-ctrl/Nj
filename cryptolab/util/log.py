"""Logging setup: console + rotating file under logs/."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    import time as _t
    fmt.converter = _t.gmtime
    root = logging.getLogger()
    root.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / "cryptolab.log", maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True
