import pandas as pd

from filinglens import database


def financial_rows(revenue_2023: float) -> pd.DataFrame:
    rows = [
        ("META", 2022, "revenue", 100.0),
        ("META", 2022, "cost_of_revenue", 20.0),
        ("META", 2023, "revenue", revenue_2023),
        ("META", 2023, "cost_of_revenue", 30.0),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "fiscal_year", "metric", "value"])
    return df.assign(
        unit="USD", period_start=pd.Timestamp("2023-01-01"), period_end=pd.Timestamp("2023-12-31"),
        xbrl_tag="Revenues", accession_number="x", filed_date=pd.Timestamp("2024-02-01"),
    )


def test_rerunning_replaces_rows_instead_of_duplicating(tmp_path):
    con = database.connect(tmp_path / "test.duckdb")
    database.replace_rows(con, "financials", financial_rows(revenue_2023=120.0))
    database.replace_rows(con, "financials", financial_rows(revenue_2023=150.0))

    assert con.execute("SELECT COUNT(*) FROM financials").fetchone()[0] == 4
    value = con.execute(
        "SELECT value FROM financials WHERE fiscal_year = 2023 AND metric = 'revenue'"
    ).fetchone()[0]
    assert value == 150.0


def test_financials_wide_view(tmp_path):
    con = database.connect(tmp_path / "test.duckdb")
    database.replace_rows(con, "financials", financial_rows(revenue_2023=150.0))
    database.create_views(con)

    row = con.execute("""
        SELECT gross_profit, gross_margin, revenue_growth
        FROM financials_wide WHERE fiscal_year = 2023
    """).fetchone()
    # No gross profit was reported, so it's derived: 150 - 30 = 120, margin 120/150 = 0.8.
    assert row[0] == 120.0
    assert abs(row[1] - 0.8) < 1e-9
    assert abs(row[2] - 0.5) < 1e-9  # 150 vs 100 the year before
