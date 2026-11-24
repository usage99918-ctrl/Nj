"""HTTP client with retries, exponential backoff, per-host pacing and failure accounting.

Only GET requests to public, unauthenticated endpoints are made by this project.
No API keys are used or accepted anywhere.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int | None, body: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.url = url
        self.status = status
        self.body = body


@dataclass
class HttpStats:
    requests: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    retries: Counter = field(default_factory=Counter)
    last_error: dict = field(default_factory=dict)


class HttpClient:
    # Status codes that are worth retrying. 451/403 are geo/policy blocks: never retried.
    RETRY_STATUS = {408, 418, 425, 429, 500, 502, 503, 504}

    def __init__(self, timeout_s: float = 15, max_retries: int = 4, backoff_base_s: float = 1.0,
                 min_interval_s: float = 0.06, user_agent: str = "cryptolab-research/1.0") -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.min_interval_s = min_interval_s
        self._local = threading.local()
        self._host_lock: dict[str, threading.Lock] = {}
        self._host_next: dict[str, float] = {}
        self._global_lock = threading.Lock()
        self.user_agent = user_agent
        self.stats = HttpStats()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})
            self._local.session = s
        return s

    def _pace(self, host: str) -> None:
        with self._global_lock:
            lock = self._host_lock.setdefault(host, threading.Lock())
        with lock:
            now = time.monotonic()
            nxt = self._host_next.get(host, 0.0)
            if nxt > now:
                time.sleep(nxt - now)
            self._host_next[host] = time.monotonic() + self.min_interval_s

    def get(self, url: str, params: dict | None = None, raw: bool = False, timeout_s: float | None = None) -> Any:
        host = urlparse(url).netloc
        attempt = 0
        while True:
            self._pace(host)
            self.stats.requests[host] += 1
            try:
                r = self.session.get(url, params=params, timeout=timeout_s or self.timeout_s)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError) as e:
                status, body, retry_after = None, str(e), None
            else:
                if r.status_code == 200:
                    return r.content if raw else r.json()
                status, body = r.status_code, r.text
                retry_after = r.headers.get("Retry-After")
                if status not in self.RETRY_STATUS:
                    if status != 404:  # 404 = "no such file/market" (expected for bulk-file probing)
                        self.stats.failures[host] += 1
                    self.stats.last_error[host] = f"{status}: {body[:160]}"
                    raise HttpError(url, status, body)
            attempt += 1
            if attempt > self.max_retries:
                self.stats.failures[host] += 1
                self.stats.last_error[host] = f"{status}: {body[:160]}"
                raise HttpError(url, status, body)
            self.stats.retries[host] += 1
            delay = self.backoff_base_s * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            log.warning("retry %d/%d %s (%s) in %.1fs", attempt, self.max_retries, url, status, delay)
            time.sleep(min(delay, 60))
