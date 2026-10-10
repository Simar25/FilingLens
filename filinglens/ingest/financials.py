"""Step 2 - download each company's reported financial numbers into a tidy table.

SEC's "companyfacts" API returns every number a company has tagged in its
filings using XBRL, a standard vocabulary for financial data. For example the
tag "NetIncomeLoss" means net income, whichever company reports it.

The raw data has three traps, and this module handles each one:

1. Same idea, different tags. Revenue may be tagged "Revenues" by one company
   and "RevenueFromContractWithCustomerExcludingAssessedTax" by another, or by
   the same company after an accounting-rule change.
   -> METRICS gives each friendly metric name a list of tags, in order of preference.

2. The "fy" field is misleading. A 10-K repeats the previous two years for
   comparison, and every number in it carries the *filing's* fiscal year, not
   the year that number describes. It's also occasionally just wrong.
   -> We work out a number's year from its period END date instead
      (see fiscal_year_labels).

3. Restatements. Companies sometimes correct old numbers, so one year can show
   up in several filings with different values.
   -> We keep the most recently filed value.
"""

from dataclasses import dataclass

import pandas as pd

from filinglens.ingest.companies import cik10
from filinglens.ingest.sec_client import SecClient

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

ANNUAL_FORMS = ("10-K", "10-K/A")
# A fiscal year is 52 or 53 weeks long (364 or 371 days). This range allows for
# that while filtering out quarterly (~91 day) and half-year numbers.
MIN_YEAR_DAYS, MAX_YEAR_DAYS = 350, 380

FINANCIALS_COLUMNS = [
    "ticker", "fiscal_year", "metric", "value", "unit",
    "period_start", "period_end", "xbrl_tag", "accession_number", "filed_date",
]


@dataclass(frozen=True)
class Metric:
    tags: tuple[str, ...]  # XBRL tags to try, most preferred first
    unit: str              # "USD" for money, "USD/shares" for per-share amounts
    kind: str              # "duration": measured over the year (income statement, cash flow)
                           # "instant": measured on the last day of the year (balance sheet)


METRICS: dict[str, Metric] = {
    "revenue": Metric(
        ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
         "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"),
        "USD", "duration"),
    "cost_of_revenue": Metric(("CostOfRevenue", "CostOfGoodsAndServicesSold"), "USD", "duration"),
    "gross_profit": Metric(("GrossProfit",), "USD", "duration"),
    "rnd_expense": Metric(
        ("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
        "USD", "duration"),
    "operating_income": Metric(("OperatingIncomeLoss",), "USD", "duration"),
    "net_income": Metric(("NetIncomeLoss",), "USD", "duration"),
    "eps_diluted": Metric(("EarningsPerShareDiluted",), "USD/shares", "duration"),
    "operating_cash_flow": Metric(("NetCashProvidedByUsedInOperatingActivities",), "USD", "duration"),
    "capex": Metric(
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        "USD", "duration"),
    "total_assets": Metric(("Assets",), "USD", "instant"),
    "total_liabilities": Metric(("Liabilities",), "USD", "instant"),
    "stockholders_equity": Metric(("StockholdersEquity",), "USD", "instant"),
    "cash": Metric(("CashAndCashEquivalentsAtCarryingValue",), "USD", "instant"),
    "dividends_paid": Metric(("PaymentsOfDividends","PaymentsOfDividendsCommonStock"), "USD", "duration"),
}


def fetch_company_facts(client: SecClient, cik: int) -> dict:
    return client.get_json(COMPANY_FACTS_URL.format(cik=cik10(cik)), f"companyfacts/CIK{cik10(cik)}.json")


def _annual_report_facts(facts_json: dict) -> pd.DataFrame:
    """Collect every value of the tags in METRICS, from annual reports only, as rows."""
    # The JSON is nested: facts -> us-gaap -> <tag> -> units -> <unit> -> [list of values]
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    rows = []
    for metric in METRICS.values():
        for tag in metric.tags:
            for fact in us_gaap.get(tag, {}).get("units", {}).get(metric.unit, []):
                if fact.get("form") in ANNUAL_FORMS:
                    rows.append({
                        "tag": tag, "start": fact.get("start"), "end": fact["end"], "value": fact["val"],
                        "accession_number": fact["accn"], "fy": fact.get("fy"), "filed": fact["filed"],
                    })
    df = pd.DataFrame(rows, columns=["tag", "start", "end", "value", "accession_number", "fy", "filed"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["days"] = (df["end"] - df["start"]).dt.days  # empty for instant values (no start date)
    return df


def _calendar_year(period_end: pd.Series) -> pd.Series:
    # Some companies' 52/53-week years end on the Saturday closest to Dec 31, which
    # can fall on Jan 1-3. Shifting back a week counts those as the year just ended.
    return (period_end - pd.Timedelta(days=7)).dt.year


def fiscal_year_labels(facts: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], dict[str, int]]:
    """Work out what the company calls each fiscal year.

    In every annual report, the LATEST period end among its full-year numbers is
    the year the report is about. But companies name that year differently:
    NVIDIA calls the year ending January 2026 "fiscal 2026", while Target calls
    the year ending February 2026 "fiscal 2025".

    Each report's "fy" field shows the company's convention, but it's occasionally
    wrong (SEC has Salesforce's report for the year ending January 2026 as
    fy=2025). So instead of trusting any single report, we learn the convention by
    majority vote across all of the company's reports:

        offset      = most common value of (fy - calendar year of period end)
        fiscal year = calendar year of period end + offset

    Returns:
        year_by_period_end:  {Timestamp('2024-01-28'): 2024, ...}
        year_by_accession:   {'0001045810-24-000029': 2024, ...}
    """
    full_year = facts[facts["days"].between(MIN_YEAR_DAYS, MAX_YEAR_DAYS)]
    per_report = full_year.groupby("accession_number").agg(
        period_end=("end", "max"), fy=("fy", "first"), filed=("filed", "first")
    )
    if per_report.empty:
        return {}, {}

    calendar_year = _calendar_year(per_report["period_end"])
    votes = (per_report["fy"] - calendar_year).dropna()
    offset = int(votes.mode().iloc[0]) if not votes.empty else 0
    per_report["fiscal_year"] = (calendar_year + offset).astype(int)

    # If an amendment (10-K/A) covers the same year as the original, the original wins.
    year_by_period_end = (
        per_report.sort_values("filed").drop_duplicates("period_end").set_index("period_end")["fiscal_year"].to_dict()
    )
    year_by_accession = per_report["fiscal_year"].to_dict()
    return year_by_period_end, year_by_accession


def extract_annual_financials(facts_json: dict, ticker: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn a companyfacts JSON into one row per (fiscal year, metric).

    Also returns {accession_number: fiscal_year}, used to label the 10-K documents.
    """
    facts = _annual_report_facts(facts_json)
    if facts.empty:
        return pd.DataFrame(columns=FINANCIALS_COLUMNS), {}
    year_by_period_end, year_by_accession = fiscal_year_labels(facts)

    picked = []
    for name, metric in METRICS.items():
        candidates = facts[facts["tag"].isin(metric.tags)]
        if metric.kind == "duration":
            candidates = candidates[candidates["days"].between(MIN_YEAR_DAYS, MAX_YEAR_DAYS)]
        else:
            candidates = candidates[candidates["start"].isna()]

        # Keep only values dated exactly on a fiscal year end. Anything else (e.g. a
        # share count dated on the report's cover page) gets no label and is dropped.
        candidates = candidates.assign(fiscal_year=candidates["end"].map(year_by_period_end))
        candidates = candidates.dropna(subset=["fiscal_year"])
        if candidates.empty:
            continue

        preference = {tag: rank for rank, tag in enumerate(metric.tags)}
        best = (
            candidates.assign(tag_rank=candidates["tag"].map(preference))
            # For each year: most preferred tag first, then the most recently filed value.
            .sort_values(["fiscal_year", "tag_rank", "filed"], ascending=[True, True, False])
            .drop_duplicates("fiscal_year")
        )
        picked.append(best.assign(metric=name, unit=metric.unit))

    if not picked:
        return pd.DataFrame(columns=FINANCIALS_COLUMNS), year_by_accession

    out = pd.concat(picked, ignore_index=True).rename(
        columns={"start": "period_start", "end": "period_end", "tag": "xbrl_tag", "filed": "filed_date"}
    )
    out = out.assign(ticker=ticker, fiscal_year=out["fiscal_year"].astype(int))
    return out[FINANCIALS_COLUMNS], year_by_accession
