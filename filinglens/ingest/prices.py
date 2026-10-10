"""Step 4 - download daily stock prices from Yahoo Finance (via the yfinance library).

We store *adjusted* prices, which are corrected for stock splits and dividends.
Example: NVIDIA split each share into 10 in June 2024, so the raw price fell
from about $1,200 to $120 overnight. Unadjusted, that looks like a 90% crash.
"""

import logging
from datetime import date

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def download_prices(tickers: list[str], start: date) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(start=start.isoformat(), auto_adjust=True)
        except Exception as exc:  # Yahoo sometimes rate-limits or times out
            log.warning("%s: price download failed (%s), skipping", ticker, exc)
            continue
        if history.empty:
            log.warning("%s: Yahoo returned no prices, skipping", ticker)
            continue

        df = history.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        # Dates arrive with a New York timezone attached; we only need the trading day.
        df["date"] = pd.to_datetime(df["date"].dt.date)
        frames.append(df.assign(ticker=ticker))
        log.info("%s: %d trading days", ticker, len(df))

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
