"""A small, polite HTTP client for SEC EDGAR.

SEC's rules for automated downloads (https://www.sec.gov/os/accessing-edgar-data):
  1. Identify yourself: send a User-Agent header with your name and email.
  2. Make at most 10 requests per second.
Break them and SEC blocks your IP address for a while.

This client follows both rules, retries temporary server errors, and saves every
response to disk (a cache), so re-running the pipeline skips files it already has.
"""

import json
import logging
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class SecClient:
    # 0.15 s between requests = at most ~7 per second, safely under SEC's limit of 10.
    MIN_SECONDS_BETWEEN_REQUESTS = 0.15

    def __init__(self, user_agent: str, cache_dir: Path, refresh: bool = False):
        if "@" not in user_agent or "example.com" in user_agent:
            raise ValueError(
                "SEC_USER_AGENT is not set. Copy .env.example to .env and put your "
                'name and email in it, e.g. SEC_USER_AGENT="Jane Doe jane@gmail.com"'
            )
        self.cache_dir = cache_dir
        # JSON indexes (filing lists, financial facts) grow whenever a company files
        # a new report, so refresh=True re-downloads them. Filed documents never change.
        self.refresh = refresh
        self._last_request_at = 0.0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        # Retry "too many requests" and server errors automatically, waiting
        # longer after each failed attempt (exponential backoff).
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get(self, url: str) -> requests.Response:
        wait = self.MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()
        log.debug("GET %s", url)
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response

    def get_json(self, url: str, cache_name: str) -> dict:
        """Return parsed JSON from `url`, cached at data/raw/sec/<cache_name>."""
        path = self.cache_dir / cache_name
        if path.exists() and not self.refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._get(url).json()
        _write_atomically(path, json.dumps(data).encode("utf-8"))
        return data

    def download_file(self, url: str, dest: Path) -> Path:
        """Download `url` to `dest`, unless it was already downloaded."""
        if not dest.exists():
            _write_atomically(dest, self._get(url).content)
        return dest


def _write_atomically(path: Path, content: bytes) -> None:
    # Write to a temporary ".part" file, then rename it. If the program is stopped
    # mid-download, no half-written file is left behind looking complete.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)
