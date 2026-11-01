import pandas as pd

from filinglens.ingest.financials import extract_annual_financials


def fact(start, end, val, accn, fy, filed, form="10-K", fp="FY"):
    """One value in the same shape SEC's companyfacts API returns."""
    out = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        out["start"] = start
    return out


# Modelled on NVIDIA, whose fiscal year ends in late January.
FY24 = "0001045810-24-000029"  # annual report for fiscal 2024 (year ending 2024-01-28)
FY23 = "0001045810-23-000017"  # annual report for fiscal 2023 (year ending 2023-01-29)

FACTS = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        # The fiscal 2024 report: current year plus two comparison years, ALL tagged fy=2024.
        fact("2023-01-30", "2024-01-28", 60_922, FY24, 2024, "2024-02-21"),
        fact("2022-01-31", "2023-01-29", 26_974, FY24, 2024, "2024-02-21"),
        fact("2021-02-01", "2022-01-30", 26_914, FY24, 2024, "2024-02-21"),
        # A quarterly number inside the annual report: must be ignored.
        fact("2023-10-30", "2024-01-28", 22_103, FY24, 2024, "2024-02-21"),
        # The fiscal 2023 report originally said 26_900; the 2024 report restated it.
        fact("2022-01-31", "2023-01-29", 26_900, FY23, 2023, "2023-02-24"),
        # A quarterly report (10-Q): must be ignored.
        fact("2023-07-31", "2023-10-29", 18_120, "q3", 2024, "2023-11-21", form="10-Q", fp="Q3"),
    ]}},
    # A less preferred revenue tag: ignored in years where "Revenues" exists.
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        fact("2023-01-30", "2024-01-28", 99_999, FY24, 2024, "2024-02-21"),
    ]}},
    "Assets": {"units": {"USD": [
        fact(None, "2024-01-28", 65_728, FY24, 2024, "2024-02-21"),
        fact(None, "2023-01-29", 41_182, FY24, 2024, "2024-02-21"),
        fact(None, "2024-02-16", 1, FY24, 2024, "2024-02-21"),  # not a year-end date: dropped
    ]}},
}}}


def metric_by_year(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df[df["metric"] == metric].set_index("fiscal_year")


def test_year_comes_from_period_end_not_fy_field():
    df, _ = extract_annual_financials(FACTS, "NVDA")
    revenue = metric_by_year(df, "revenue")
    assert revenue.loc[2024, "value"] == 60_922
    assert revenue.loc[2024, "period_end"] == pd.Timestamp("2024-01-28")
    # The year ending 2022-01-30 has no annual report of its own in this data, so no label.
    assert sorted(revenue.index) == [2023, 2024]


def test_restated_value_wins():
    df, _ = extract_annual_financials(FACTS, "NVDA")
    fy2023 = metric_by_year(df, "revenue").loc[2023]
    assert fy2023["value"] == 26_974
    assert fy2023["accession_number"] == FY24


def test_quarterly_numbers_ignored():
    df, _ = extract_annual_financials(FACTS, "NVDA")
    assert not df["value"].isin([22_103, 18_120]).any()


def test_preferred_tag_wins():
    df, _ = extract_annual_financials(FACTS, "NVDA")
    assert metric_by_year(df, "revenue").loc[2024, "xbrl_tag"] == "Revenues"


def test_balance_sheet_values_use_year_end_dates():
    df, _ = extract_annual_financials(FACTS, "NVDA")
    assets = metric_by_year(df, "total_assets")
    assert assets["value"].to_dict() == {2023: 41_182, 2024: 65_728}


def test_fiscal_year_for_each_report():
    _, year_by_accession = extract_annual_financials(FACTS, "NVDA")
    assert year_by_accession == {FY23: 2023, FY24: 2024}


def revenue_reports(*reports):
    """Facts with one revenue value per annual report: (accession, fy, start, end, filed)."""
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        fact(start, end, 100, accn, fy, filed) for accn, fy, start, end, filed in reports
    ]}}}}}


def test_one_wrong_fy_label_is_outvoted():
    # Real case: SEC tags Salesforce's report for the year ending 2026-01-31 as fy=2025.
    facts = revenue_reports(
        ("r24", 2024, "2023-02-01", "2024-01-31", "2024-03-06"),
        ("r25", 2025, "2024-02-01", "2025-01-31", "2025-03-05"),
        ("r26", 2025, "2025-02-01", "2026-01-31", "2026-03-02"),  # wrong: should be 2026
    )
    df, year_by_accession = extract_annual_financials(facts, "CRM")
    assert year_by_accession == {"r24": 2024, "r25": 2025, "r26": 2026}
    assert sorted(df["fiscal_year"]) == [2024, 2025, 2026]


def test_company_naming_years_after_the_previous_calendar_year():
    # Target calls the year ending 2024-02-03 "fiscal 2023".
    facts = revenue_reports(
        ("t22", 2022, "2022-01-30", "2023-01-28", "2023-03-08"),
        ("t23", 2023, "2023-01-29", "2024-02-03", "2024-03-13"),
    )
    _, year_by_accession = extract_annual_financials(facts, "TGT")
    assert year_by_accession == {"t22": 2022, "t23": 2023}


def test_year_ending_in_first_days_of_january_belongs_to_previous_year():
    # 52/53-week calendars ending on the Saturday closest to Dec 31.
    facts = revenue_reports(
        ("a20", 2020, "2019-12-29", "2020-12-26", "2021-02-20"),
        ("a21", 2021, "2020-12-27", "2022-01-01", "2022-02-19"),
    )
    _, year_by_accession = extract_annual_financials(facts, "X")
    assert year_by_accession == {"a20": 2020, "a21": 2021}


def test_falls_back_to_next_tag_when_preferred_is_missing():
    # e.g. a company that switched tags after the 2018 revenue accounting change
    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [
            fact("2017-01-01", "2017-12-31", 100, "A17", 2017, "2018-02-01")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            fact("2018-01-01", "2018-12-31", 120, "A18", 2018, "2019-02-01")]}},
    }}}
    df, _ = extract_annual_financials(facts, "X")
    tags = metric_by_year(df, "revenue")["xbrl_tag"].to_dict()
    assert tags == {2017: "SalesRevenueNet", 2018: "RevenueFromContractWithCustomerExcludingAssessedTax"}


def test_income_statement_total_beats_contract_revenue():
    # Like American Express: contract revenue leaves out interest income, so it's far below the total.
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            fact("2021-01-01", "2021-12-31", 27, "A21", 2021, "2022-02-01")]}},
        "RevenuesNetOfInterestExpense": {"units": {"USD": [
            fact("2021-01-01", "2021-12-31", 42, "A21", 2021, "2022-02-01")]}},
    }}}
    df, _ = extract_annual_financials(facts, "X")
    assert metric_by_year(df, "revenue").loc[2021, "value"] == 42


def test_net_income_fallback_only_fills_missing_years():
    # Like Caterpillar: "NetIncomeLoss" until 2024, then only "ProfitLoss" (which includes
    # minority interests). The fallback must fill 2025 without replacing 2024.
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            fact("2024-01-01", "2024-12-31", 10, "A24", 2024, "2025-02-01")]}},
        "ProfitLoss": {"units": {"USD": [
            fact("2024-01-01", "2024-12-31", 11, "A25", 2025, "2026-02-01"),
            fact("2025-01-01", "2025-12-31", 12, "A25", 2025, "2026-02-01")]}},
    }}}
    df, _ = extract_annual_financials(facts, "X")
    net_income = metric_by_year(df, "net_income")
    assert net_income["value"].to_dict() == {2024: 10, 2025: 12}
    assert net_income["xbrl_tag"].to_dict() == {2024: "NetIncomeLoss", 2025: "ProfitLoss"}


def test_company_without_data():
    df, year_by_accession = extract_annual_financials({"facts": {}}, "X")
    assert df.empty and year_by_accession == {}


def per_share(reports):
    """Earnings per share as annual reports show it: {year of the report: {year of the period: value}}."""
    return {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        fact(f"{period}-01-01", f"{period}-12-31", value, f"r{year}", year, f"{year + 1}-02-01")
        for year, values in reports.items() for period, value in values.items()]}}}}}


def eps_by_year(reports):
    df, _ = extract_annual_financials(per_share(reports), "X")
    return dict(zip(df["fiscal_year"], df["value"]))


def test_split_puts_every_year_on_todays_share_basis():
    eps = eps_by_year({2020: {2020: 10.0}, 2021: {2020: 10.0, 2021: 20.0}, 2022: {2020: 10.0, 2021: 20.0, 2022: 30.0},
                       2023: {2021: 20.0, 2022: 30.0, 2023: 40.0}, 2024: {2022: 3.0, 2023: 4.0, 2024: 5.0}})
    assert eps == {2020: 1.0, 2021: 2.0, 2022: 3.0, 2023: 4.0, 2024: 5.0}


def test_reverse_split_multiplies_older_years():
    eps = eps_by_year({2019: {2019: 0.5}, 2020: {2019: 0.5, 2020: 1.0}, 2021: {2019: 0.5, 2020: 1.0, 2021: 2.0},
                       2022: {2020: 8.0, 2021: 16.0, 2022: 24.0}})
    assert eps == {2019: 4.0, 2020: 8.0, 2021: 16.0, 2022: 24.0}


def test_a_single_restated_year_is_not_a_split():
    eps = eps_by_year({2021: {2021: 4.0}, 2022: {2021: 4.0, 2022: 6.0}, 2023: {2021: 2.0, 2022: 6.0, 2023: 7.0}})
    assert eps == {2021: 2.0, 2022: 6.0, 2023: 7.0}


def test_two_years_restated_by_different_amounts_are_not_a_split():
    eps = eps_by_year({2021: {2021: 0.26}, 2022: {2021: 0.26, 2022: 0.17}, 2023: {2021: 0.46, 2022: 0.49, 2023: 1.0}})
    assert eps == {2021: 0.46, 2022: 0.49, 2023: 1.0}
