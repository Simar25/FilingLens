"""Step 4.3 - what happened to the stock after each report came out?

For every 10-K we measure the stock's return over the next 3, 6 and 12 months,
minus the market's return (SPY) over exactly the same days. That difference is
the "abnormal" return (in finance: a market-adjusted buy-and-hold return). It
removes what moved all stocks together, so a report filed just before a market
crash isn't blamed for the crash.

Avoiding look-ahead bias, the most common mistake in backtests:
  - The holding period starts at the close of the first trading day AFTER the
    filing date. Many 10-Ks are published after the market closes, so the
    filing day's own close may not know the report yet. Starting a day later
    means an investor really could have read the report first.
  - A window is only used when it is complete. A report filed 4 months ago has
    a 3-month return, but no 6- or 12-month return yet (not a partial one).

Months are counted in trading days: 3 months = 63, 6 = 126, 12 = 252.

Also measured: the abnormal return over the 12 months BEFORE the filing
("momentum"). Stocks that rose recently tend to keep rising for a while, so it
is a standard control: we want to know if the text says something beyond that.
"""

import numpy as np
import pandas as pd

HORIZONS = {"3m": 63, "6m": 126, "12m": 252}  # trading days
PAST_DAYS = 252


def future_returns(filings: pd.DataFrame, prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """One row per filing with abnormal returns after it.

    filings: ticker, fiscal_year, accession_number, filing_date
    prices:  ticker, date, close (adjusted), including the benchmark
    """
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    calendar = closes.index[closes[benchmark].notna()]  # the market's trading days
    closes = closes.loc[calendar].ffill()  # a missing day for one stock (e.g. a halt) uses its last close
    market = closes[benchmark].to_numpy()

    rows = []
    for f in filings.itertuples():
        # Position of the first trading day strictly after the filing date.
        start = int(calendar.searchsorted(pd.Timestamp(f.filing_date), side="right"))
        row = {"ticker": f.ticker, "fiscal_year": f.fiscal_year, "accession_number": f.accession_number,
               "filing_date": f.filing_date,
               "start_date": calendar[start] if start < len(calendar) else pd.NaT}
        stock = closes[f.ticker].to_numpy() if f.ticker in closes else np.full(len(calendar), np.nan)

        for name, days in HORIZONS.items():
            end = start + days
            if end < len(calendar):  # complete windows only
                ret, mkt = stock[end] / stock[start] - 1, market[end] / market[start] - 1
            else:
                ret = mkt = np.nan
            row |= {f"ret_{name}": ret, f"market_{name}": mkt, f"abnormal_{name}": ret - mkt}

        # Momentum: the 12 months up to the filing date (the last close the investor knew).
        last_known, first = start - 1, start - 1 - PAST_DAYS
        if first >= 0 and start < len(calendar):
            row["past_abnormal_12m"] = ((stock[last_known] / stock[first] - 1)
                                        - (market[last_known] / market[first] - 1))
        else:
            row["past_abnormal_12m"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
