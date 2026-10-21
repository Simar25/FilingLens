"""Download annual reports of Indian listed companies from NSE (National Stock Exchange).

Why NSE and not MCA (the Ministry of Corporate Affairs registry)? MCA's documents
need a login and a fee per company, have no bulk access, and its copy is filed
after the shareholders' meeting, later than the exchange's. Under SEBI's listing
rules (LODR Regulation 34) every listed company sends its annual report to the
exchanges, and NSE publishes the list on its website with the PDF links and,
for recent years, the exact second each report was made public.

That list comes from the JSON service NSE's own web page uses. It isn't an
official API, so:
  - it only answers requests that look like a browser and carry the cookie
    NSE sets when its page is opened, so we open the page first;
  - we wait 1.5 s between requests and cache every answer and file on disk,
    so each file is downloaded once;
  - it may change without notice. If it breaks, this module is the only place
    to fix.
"""

import io
import json
import logging
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from filinglens.config import RAW_DIR

log = logging.getLogger(__name__)

PAGE_URL = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
LIST_URL = "https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
NSE_DIR = RAW_DIR / "nse"


class NseClient:
    SECONDS_BETWEEN_REQUESTS = 1.5

    def __init__(self, cache_dir: Path = NSE_DIR, refresh: bool = False):
        self.cache_dir = cache_dir
        self.refresh = refresh  # re-download the report lists (new reports appear each year)
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self._has_cookie = False
        self._last_request_at = 0.0

    def _get(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(3):
            if not self._has_cookie:
                self._wait()
                self.session.get(PAGE_URL, timeout=30)
                self._has_cookie = True
            self._wait()
            response = self.session.get(url, timeout=120, **kwargs)
            if response.status_code in (401, 403):  # cookie expired: open the page again
                self._has_cookie = False
                continue
            if response.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        response.raise_for_status()
        raise requests.HTTPError(f"NSE kept refusing {url} (HTTP {response.status_code})")

    def _wait(self) -> None:
        pause = self.SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request_at)
        if pause > 0:
            time.sleep(pause)
        self._last_request_at = time.monotonic()

    def annual_report_list(self, symbol: str) -> list[dict]:
        path = self.cache_dir / "annual_reports" / f"{symbol}.json"
        if path.exists() and not self.refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        rows = self._get(LIST_URL.format(symbol=symbol), headers={"Accept": "application/json",
                                                                  "Referer": PAGE_URL}).json().get("data", [])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return rows

    def download(self, url: str, symbol: str) -> list[Path]:
        """Download a report file; returns its PDF(s). A .zip is unpacked (some years are zipped).

        NSE's file server is slow (~0.2 MB/s), sometimes stops sending halfway without
        an error, and drops long transfers (some reports are 250 MB). So the file is
        written to a ".part" file as it arrives; after a break, the download RESUMES
        from where it stopped (an HTTP Range request: "send bytes 13,803,520 onwards"),
        like a browser does. A file is only used once it's complete (is_complete).
        """
        name = url.rsplit("/", 1)[-1]
        path = self.cache_dir / "reports" / symbol / name
        if not (path.exists() and is_complete(path.read_bytes(), path.suffix)):
            self._download_resumable(url, path)
            log.info("%s: downloaded %s (%.1f MB)", symbol, name, path.stat().st_size / 1e6)
        if path.suffix.lower() == ".pdf":
            return [path]
        if path.suffix.lower() == ".zip":
            return unzip_pdfs(path, path.with_suffix(""))
        raise ValueError(f"{name}: expected a .pdf or .zip file")

    def _download_resumable(self, url: str, path: Path, attempts: int = 8) -> None:
        part = path.with_name(path.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(attempts):
            have = part.stat().st_size if part.exists() else 0
            try:
                response = self._get(url, stream=True, headers={"Range": f"bytes={have}-"} if have else {})
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 416:
                    raise
                # 416 = "nothing after byte {have}". Either the file really ends here, or NSE's
                # servers hiccuped: once, a 416 at 5.4 MB was followed by more data on the next try.
                if is_complete(part.read_bytes(), path.suffix):
                    part.replace(path)
                    return
                log.warning("%s: server said nothing after %.1f MB, but the file isn't complete; retrying",
                            path.name, have / 1e6)
                time.sleep(30)
                continue
            try:
                with response, open(part, "ab" if have and response.status_code == 206 else "wb") as out:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                log.warning("%s: broke off at %.1f MB (%s), resuming", path.name, part.stat().st_size / 1e6,
                            type(exc).__name__)
                continue
            if is_complete(part.read_bytes(), path.suffix):
                part.replace(path)
                return
            log.warning("%s: arrived incomplete (%.1f MB), resuming", path.name, part.stat().st_size / 1e6)
        raise IOError(f"{path.name}: still incomplete after {attempts} attempts ({part.stat().st_size / 1e6:.1f} MB "
                      "so far, kept for the next run); NSE's server may be busy, or try BSE's copy")


def is_complete(content: bytes, suffix: str) -> bool:
    """A whole PDF starts with '%PDF-' and ends with an '%%EOF' marker; a whole zip opens as one."""
    if suffix.lower() == ".pdf":
        return content.startswith(b"%PDF-") and b"%%EOF" in content[-2048:]
    if suffix.lower() == ".zip":
        return zipfile.is_zipfile(io.BytesIO(content))
    return bool(content)


def unzip_pdfs(zip_path: Path, folder: Path) -> list[Path]:
    """Extract only the PDFs, each under its bare file name (a name like '../x.pdf' can't escape the folder)."""
    paths = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".pdf"):
                continue
            target = folder / name
            if not target.exists():
                folder.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source:
                    target.write_bytes(source.read())
            paths.append(target)
    return sorted(paths)


def _report_id(file_url: str) -> int:
    """NSE numbers its annual-report files in order of arrival: AR_26481_INFY_2024_2025_..."""
    match = re.search(r"/AR_(\d+)_", file_url)
    return int(match.group(1)) if match else 10**9


def _published_at(value: str | None) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(value, "%d-%b-%Y %H:%M:%S"))
    except (TypeError, ValueError):  # "-" or missing: older entries often have no time stamp
        return pd.NaT


def choose_reports(rows: list[dict]) -> pd.DataFrame:
    """One report per fiscal year: the FIRST submission, not a later "Revised" one.

    The first version is what investors saw first, so it's the one whose date
    counts. Indian fiscal years run April to March; like the US data, a year is
    labelled by the year it ends: April 2024 - March 2025 = fiscal 2025.
    """
    df = pd.DataFrame([{
        "fiscal_year": int(r["toYr"]),
        "revised": str(r.get("submission_type", "")).lower().startswith("rev"),
        "report_id": _report_id(r["fileName"]),
        "published_at": _published_at(r.get("disseminationDateTime")),
        "url": r["fileName"],
        "company": r.get("companyName", ""),
    } for r in rows if r.get("fileName") and str(r.get("toYr", "")).isdigit()])
    if df.empty:
        return df
    return (df.sort_values(["fiscal_year", "revised", "report_id"])
            .drop_duplicates("fiscal_year").reset_index(drop=True))
