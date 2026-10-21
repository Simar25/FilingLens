"""Step 4.4 - the statistics: is a bigger change in the report followed by lower returns?

The hypothesis, from "Lazy Prices": companies whose Risk Factors change a lot
from last year do worse afterwards than companies that barely change them.
If so, the change score and the later abnormal return are negatively related.

Three ways to look at it, from simplest to most careful:
  1. Correlation: do the two numbers move together? Pearson measures a
     straight-line relation and is sensitive to a few extreme values; Spearman
     uses ranks only, so one huge return can't dominate it.
  2. Groups (as in the paper): each year, sort reports into five equal groups
     (quintiles) by change score. Compare the average return of the most-changed
     fifth with the least-changed fifth (Welch's t-test).
  3. Regression: the effect of change while holding other things fixed: the
     MD&A's tone, past returns (momentum), company size, plus the year and the
     sector (fixed effects: each year and sector gets its own baseline).
     Standard errors are clustered by company, because the same company's
     results in different years aren't independent observations.

The main test is fixed in advance (PRIMARY), before looking at any results.
Trying many measures and horizons and reporting whichever "works" would find
something by chance: with 20 tests at the 5% level, about one is expected to
look significant even if there is no effect at all.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PRIMARY = {"x": "rf_change_cosine", "y": "abnormal_6m"}
HORIZONS = ["abnormal_3m", "abnormal_6m", "abnormal_12m"]
CHANGE_MEASURES = ["rf_change_cosine", "rf_change_jaccard", "rf_change_embedding",
                   "mdna_change_cosine", "mdna_change_jaccard", "mdna_change_embedding"]
CONTROLS = ["tone", "past_abnormal_12m", "log_assets"]

STUDY_SQL = """
SELECT r.ticker, r.fiscal_year, ci.sector, ci.industry, r.filing_date,
       rf.change_cosine     AS rf_change_cosine,
       rf.change_jaccard    AS rf_change_jaccard,
       rf.change_embedding  AS rf_change_embedding,
       LN(rf.n_words / rf.prev_n_words) AS rf_length_change,   -- 0.10 = about 10% longer
       rf.n_words           AS rf_words,
       md.change_cosine     AS mdna_change_cosine,
       md.change_jaccard    AS mdna_change_jaccard,
       md.change_embedding  AS mdna_change_embedding,
       t.tone, t.negative_share,
       t.tone - tp.tone     AS tone_change,
       r.abnormal_3m, r.abnormal_6m, r.abnormal_12m, r.past_abnormal_12m,
       LN(fw.total_assets)  AS log_assets
FROM future_returns r
JOIN text_change rf       ON rf.accession_number = r.accession_number AND rf.item = '1A'
LEFT JOIN text_change md  ON md.accession_number = r.accession_number AND md.item = '7'
LEFT JOIN tone t          ON t.accession_number = r.accession_number
LEFT JOIN tone tp         ON tp.ticker = r.ticker AND tp.fiscal_year = r.fiscal_year - 1
LEFT JOIN company_info ci ON ci.ticker = r.ticker
LEFT JOIN financials_wide fw ON fw.ticker = r.ticker AND fw.fiscal_year = r.fiscal_year
ORDER BY r.ticker, r.fiscal_year
"""


def study_table(con) -> pd.DataFrame:
    """One row per company and fiscal year with a Risk Factors change score: the data for every test."""
    return con.execute(STUDY_SQL).df()


def correlations(df: pd.DataFrame, xs: list[str], ys: list[str]) -> pd.DataFrame:
    rows = []
    for x in xs:
        for y in ys:
            d = df[[x, y]].dropna()
            pearson, spearman = stats.pearsonr(d[x], d[y]), stats.spearmanr(d[x], d[y])
            rows.append({"x": x, "y": y, "n": len(d),
                         "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
                         "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue})
    return pd.DataFrame(rows)


def assign_groups(df: pd.DataFrame, x: str, n_groups: int = 5) -> pd.Series:
    """Group 1 (changed least) to n_groups (changed most), ranked within each fiscal year.

    Within each year, because some years changed every report (2020-21: COVID
    added new risks everywhere). Ranking across all years at once would put
    those whole years in the top group and compare years, not companies.
    """
    ranks = df.groupby("fiscal_year")[x].rank(method="first")
    sizes = df.groupby("fiscal_year")[x].transform("count")
    return (np.ceil(ranks / sizes * n_groups)).astype(int)


def group_returns(df: pd.DataFrame, x: str, y: str, n_groups: int = 5) -> pd.DataFrame:
    """Average return per change group, with a 95% confidence interval."""
    d = df.dropna(subset=[x, y])
    d = d.assign(group=assign_groups(d, x, n_groups))
    out = d.groupby("group")[y].agg(n="size", mean="mean", std="std").reset_index()
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["n"])  # +/- this much around the mean
    return out


def high_minus_low(df: pd.DataFrame, x: str, y: str, n_groups: int = 5) -> dict:
    """Most-changed group minus least-changed group, and whether the difference could be chance."""
    d = df.dropna(subset=[x, y])
    group = assign_groups(d, x, n_groups)
    high, low = d.loc[group == n_groups, y], d.loc[group == 1, y]
    test = stats.ttest_ind(high, low, equal_var=False)  # Welch: doesn't assume equal spreads
    return {"x": x, "y": y, "n_high": len(high), "n_low": len(low), "mean_high": high.mean(),
            "mean_low": low.mean(), "difference": high.mean() - low.mean(), "t": test.statistic,
            "p_value": test.pvalue}


def regression(df: pd.DataFrame, x: str, y: str, controls: list[str] = CONTROLS,
               fixed_effects: tuple[str, ...] = ("fiscal_year", "sector")):
    """OLS of the return (in percent) on the change score and controls, all standardized.

    Standardized = rescaled to mean 0 and standard deviation 1, so each coefficient
    reads the same way: "a one-standard-deviation bigger X goes with a return that
    is this many percentage points higher (or lower, if negative)".
    Returns the fitted statsmodels result.
    """
    cols = [x, y, *controls, *fixed_effects, "ticker"]
    d = df[cols].dropna().copy()
    d[y] = d[y] * 100  # fraction -> percentage points
    for col in [x, *controls]:
        d[col] = (d[col] - d[col].mean()) / d[col].std()
    terms = [x, *controls, *[f"C({fe})" for fe in fixed_effects]]
    model = smf.ols(f"{y} ~ {' + '.join(terms)}", data=d)
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})


def coefficient(result, name: str) -> dict:
    ci_low, ci_high = result.conf_int().loc[name]
    return {"coef_pp": result.params[name], "std_err": result.bse[name], "p_value": result.pvalues[name],
            "ci_low": ci_low, "ci_high": ci_high, "n": int(result.nobs), "r2": result.rsquared}


def robustness_grid(df: pd.DataFrame, xs: list[str] = CHANGE_MEASURES, ys: list[str] = HORIZONS) -> pd.DataFrame:
    """The regression coefficient of every change measure on every horizon (the primary test is one cell)."""
    rows = []
    for x in xs:
        for y in ys:
            rows.append({"x": x, "y": y, "primary": (x, y) == (PRIMARY["x"], PRIMARY["y"]),
                         **coefficient(regression(df, x, y), x)})
    return pd.DataFrame(rows)
