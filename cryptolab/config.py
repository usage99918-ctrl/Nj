"""Configuration loading (TOML) with an optional local override file."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.toml"
LOCAL_CONFIG = ROOT / "config" / "local.toml"  # optional, git-ignored overrides


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with attribute-free helpers for resolved paths."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def root(self) -> Path:
        return Path(self.get("_root", ROOT))


def load_config(path: str | os.PathLike | None = None, root: str | os.PathLike | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    if path is None and LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "rb") as fh:
            cfg = _deep_merge(cfg, tomllib.load(fh))
    env_root = os.environ.get("CRYPTOLAB_ROOT")
    c = Config(cfg)
    c["_root"] = str(root or env_root or ROOT)
    return c


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
