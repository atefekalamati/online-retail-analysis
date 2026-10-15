"""Analysis: RFM, segments, customer concentration, cohort retention, Pareto and LTV.

Hand-made tests check the logic on data small enough to verify by hand. Data tests check
that the full data reproduces the figures in the README and notebook 03.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import analysis as an
from src import clean_data as cd

from conftest import make_rows

# Notebook 03 outputs; the README quotes them rounded
RFM_SCORED, RFM_DROPPED = 5_832, 43
SEGMENT_CUSTOMERS = {"Champions": 1280, "Cannot Lose Them": 236, "Loyal": 693, "At Risk": 590,
                     "New": 168, "Potential Loyalists": 715, "Hibernating": 643, "Lost": 1507}
CANNOT_LOSE_REVENUE = 901_055            # README: "worth £901K"
TOP20_SHARE = 76.7                       # README: "top 20% of customers bring 77%"
MONTH1_RETENTION = 21.3                  # README: "about 21% buy again in month 1"
MONTH3_RETENTION = 22.0
PARETO_PRODUCTS, PARETO_PCT = 4_739, 21.9    # README: "22% of products make 80%"
CUM_LTV = {                                  # README: £634 in 3 months, £1,373 in 12 (excl. Dec 2009)
    "all cohorts": {3: 766.1, 6: 1176.0, 12: 2119.7},
    "excl. Dec 2009": {3: 634.5, 6: 895.2, 12: 1373.0},
}
HISTORICAL_LTV = {"all customers": 2784.4, "scored customers": 2805.5, "median scored": 843.7}
SEGMENT_LTV = {"Champions": 8849.3, "Cannot Lose Them": 3818.0, "Loyal": 2555.1, "At Risk": 909.0,
               "New": 325.1, "Potential Loyalists": 879.8, "Hibernating": 828.5, "Lost": 404.8}
DEC_2009 = pd.Timestamp("2009-12-01")


# ---------- RFM on hand-made data ----------

@pytest.fixture
def every_score_combination() -> pd.DataFrame:
    """125 customers whose R, F and M quintiles cover every combination from 1 to 5 once.

    Customer (a, b, c) ranks a*25+... on recency, b*25+... on frequency and c*25+... on
    monetary, so its quintiles are R = 5 - a, F = b + 1, M = c + 1.
    """
    rows = []
    for a, b, c in itertools.product(range(5), repeat=3):
        rows.append({"recency_days": a * 25 + b * 5 + c,
                     "frequency": b * 25 + a * 5 + c + 1,
                     "monetary": float(c * 25 + a * 5 + b + 1),
                     "expected": (5 - a, b + 1, c + 1)})
    return pd.DataFrame(rows).rename_axis("Customer ID")


def test_rfm_scores_are_1_to_5(every_score_combination):
    out = an.score_rfm(every_score_combination.drop(columns="expected"))
    for col in ["R", "F", "M"]:
        assert set(out[col]) == {1, 2, 3, 4, 5}
        assert (out[col].value_counts() == 25).all()               # quintiles: 20% each
    actual = list(zip(out["R"], out["F"], out["M"]))
    assert actual == list(every_score_combination["expected"])
    assert (out["RFM_score"] == out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)).all()


def test_every_score_combination_gets_exactly_one_segment(every_score_combination):
    out = an.score_rfm(every_score_combination.drop(columns="expected"))
    assert out["segment"].notna().all()
    assert set(out["segment"]) == set(an.SEGMENT_ORDER)
    assert out.index.is_unique and len(out) == 125


@pytest.mark.parametrize("r, f, m, segment", [
    (5, 5, 5, "Champions"), (4, 4, 4, "Champions"),
    (1, 5, 5, "Cannot Lose Them"), (2, 4, 4, "Cannot Lose Them"),
    (3, 5, 5, "Loyal"), (5, 4, 1, "Loyal"),
    (2, 3, 5, "At Risk"), (1, 4, 1, "At Risk"),
    (5, 1, 5, "New"), (4, 1, 1, "New"),
    (4, 2, 3, "Potential Loyalists"), (5, 3, 5, "Potential Loyalists"),
    (3, 1, 1, "Hibernating"), (3, 3, 5, "Hibernating"),
    (2, 2, 5, "Lost"), (1, 1, 1, "Lost"),
])
def test_segment_rules(every_score_combination, r, f, m, segment):
    out = an.score_rfm(every_score_combination.drop(columns="expected"))
    hit = out[(out["R"] == r) & (out["F"] == f) & (out["M"] == m)]
    assert list(hit["segment"]) == [segment]


def test_rfm_table_drops_customers_with_no_net_revenue():
    cust = cd.add_derived_columns(cd.add_flags(make_rows([
        {"Customer ID": 1, "Invoice": "1", "Quantity": 2, "Price": 5.0, "InvoiceDate": "2011-11-01"},
        {"Customer ID": 1, "Invoice": "2", "Quantity": 1, "Price": 5.0, "InvoiceDate": "2011-12-01"},
        {"Customer ID": 2, "Invoice": "3", "Quantity": 1, "Price": 5.0, "InvoiceDate": "2011-12-05"},
        {"Customer ID": 2, "Invoice": "C4", "Quantity": -1, "Price": 5.0, "InvoiceDate": "2011-12-09 10:00"},
    ])))
    rfm, dropped, snapshot = an.rfm_table(cust)
    assert snapshot == pd.Timestamp("2011-12-10")
    assert dropped == 1                                        # customer 2 returned everything
    assert rfm.loc[1].to_dict() == {"recency_days": 9, "frequency": 2, "monetary": 15.0}


def test_rfm_table_treats_float_residue_as_zero():
    cust = cd.add_derived_columns(cd.add_flags(make_rows([
        # customer 1 returns everything: 0.1 + 0.2 - 0.3 leaves a float residue, not 0
        {"Customer ID": 1, "Invoice": "1", "Quantity": 1, "Price": 0.1, "InvoiceDate": "2011-12-01"},
        {"Customer ID": 1, "Invoice": "2", "Quantity": 1, "Price": 0.2, "InvoiceDate": "2011-12-02"},
        {"Customer ID": 1, "Invoice": "C3", "Quantity": -1, "Price": 0.3, "InvoiceDate": "2011-12-03"},
        # customer 2 keeps one penny of net revenue
        {"Customer ID": 2, "Invoice": "4", "Quantity": 1, "Price": 0.01, "InvoiceDate": "2011-12-04"},
    ])))
    residue = cust.loc[cust["Customer ID"] == 1, "line_revenue"].sum()
    assert 0 < residue < 1e-12
    rfm, dropped, _ = an.rfm_table(cust)
    assert dropped == 1
    assert list(rfm.index) == [2]


def test_segment_shares_sum_to_100_on_hand_made_data(every_score_combination):
    seg = an.segment_summary(an.score_rfm(every_score_combination.drop(columns="expected")))
    assert seg["customers"].sum() == 125
    assert seg["customers_pct"].sum() == pytest.approx(100)
    assert seg["revenue_pct"].sum() == pytest.approx(100)


# ---------- concentration, Pareto, cohorts on hand-made data ----------

def test_top_share():
    assert an.top_share(pd.Series([10.0, 1, 1, 1, 1]), 0.2) == pytest.approx(10 / 14 * 100)
    assert an.top_share(pd.Series([1.0, 1, 1, 1, 1, 1]), 0.2) == pytest.approx(2 / 6 * 100)  # rounds up


def test_pareto_share_of_items_for_80_percent():
    df = pd.DataFrame({"StockCode": ["A", "B", "C", "D", "A"], "line_revenue": [40.0, 30, 10, 10, 10]})
    curve, pct = an.pareto(df)
    assert list(curve.index) == ["A", "B", "C", "D"]
    assert list(curve["cum_revenue_pct"]) == pytest.approx([50, 80, 90, 100])
    assert pct == 50                                            # A and B reach 80%


@pytest.fixture
def tiny_customers() -> pd.DataFrame:
    rows = [
        {"Customer ID": 1, "Invoice": "1", "InvoiceDate": "2010-01-05", "Quantity": 1, "Price": 100.0},
        {"Customer ID": 1, "Invoice": "2", "InvoiceDate": "2010-02-05", "Quantity": 1, "Price": 50.0},
        {"Customer ID": 2, "Invoice": "3", "InvoiceDate": "2010-01-20", "Quantity": 1, "Price": 10.0},
        {"Customer ID": 2, "Invoice": "C4", "InvoiceDate": "2010-02-20", "Quantity": -1, "Price": 10.0},
        {"Customer ID": 3, "Invoice": "5", "InvoiceDate": "2010-02-10", "Quantity": 1, "Price": 30.0},
        {"Customer ID": 3, "Invoice": "6", "InvoiceDate": "2011-12-02", "Quantity": 1, "Price": 30.0},  # partial month
    ]
    return cd.add_derived_columns(cd.add_flags(make_rows(rows)))


def test_cohort_retention_by_hand(tiny_customers):
    m = an.cohort_retention(tiny_customers)
    jan, feb = pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01")
    assert list(m.index) == [jan, feb]
    assert (m[0] == 100).all()
    assert m.loc[jan, 1] == 50          # customer 1 bought again; customer 2 only cancelled
    assert pd.isna(m.loc[feb, 1])       # December 2011 is excluded
    assert an.cohort_sizes(tiny_customers).to_dict() == {jan: 2, feb: 1}


def test_cumulative_ltv_by_hand(tiny_customers):
    ltv = an.cumulative_ltv(tiny_customers, horizons=(1, 2))
    # month 1: customers 1, 2 (Jan) and 3 (Feb): 100, 10, 30
    assert ltv.loc[1, "avg_cum_revenue"] == pytest.approx((100 + 10 + 30) / 3)
    # 2 months needs a complete second month, so only the January cohort: 150 and 0
    assert ltv.loc[2, "avg_cum_revenue"] == pytest.approx((150 + 0) / 2)
    assert ltv.loc[2, "customers"] == 2


# ---------- full data: RFM and segments ----------

def test_rfm_scores_on_full_data(cust, rfm):
    _, dropped, snapshot = an.rfm_table(cust)
    assert snapshot == pd.Timestamp("2011-12-10")
    assert dropped == RFM_DROPPED
    assert len(rfm) == RFM_SCORED
    for col in ["R", "F", "M"]:
        assert rfm[col].between(1, 5).all()
        assert set(rfm[col]) == {1, 2, 3, 4, 5}


def test_segments_cover_every_customer_once(cust, rfm):
    positive = cust.groupby("Customer ID")["line_revenue"].sum().round(2).gt(0)   # rule: M in pence > 0
    assert rfm.index.is_unique
    assert set(rfm.index) == set(positive[positive].index)
    assert rfm["segment"].notna().all()
    assert rfm["segment"].value_counts().to_dict() == SEGMENT_CUSTOMERS
    assert sum(SEGMENT_CUSTOMERS.values()) == RFM_SCORED


def test_segment_revenue_shares_sum_to_100(rfm):
    seg = an.segment_summary(rfm)
    assert seg["revenue_pct"].sum() == pytest.approx(100)
    assert seg["customers_pct"].sum() == pytest.approx(100)
    assert seg["revenue"].sum() == pytest.approx(rfm["monetary"].sum())
    assert round(seg.loc["Cannot Lose Them", "revenue"]) == CANNOT_LOSE_REVENUE


def test_top_20_percent_customer_share(rfm):
    share = an.top_share(rfm["monetary"], 0.2)
    assert round(share, 1) == TOP20_SHARE
    assert round(share) == 77


# ---------- full data: cohorts ----------

def test_cohort_month0_is_100_and_month1_about_21(cust):
    m = an.cohort_retention(cust)
    assert (m[0] == 100).all()
    avg = an.mean_retention(m, [1, 3])
    assert round(avg[1], 1) == MONTH1_RETENTION
    assert round(avg[1]) == 21
    assert round(avg[3], 1) == MONTH3_RETENTION


def test_cohort_matrix_matches_report(cust, cohort_report):
    m = an.cohort_retention(cust)
    m.index.name, m.columns.name = "cohort", None
    pd.testing.assert_frame_equal(m.round(2), cohort_report, check_freq=False, atol=0.005,
                                  check_column_type=False, check_index_type=False)


def test_cohort_report_month0_is_100(cohort_report):
    assert (cohort_report[0] == 100).all()
    assert round(cohort_report[1].mean()) == 21


# ---------- full data: Pareto and LTV ----------

def test_pareto_products(products):
    curve, pct = an.pareto(products)
    assert len(curve) == PARETO_PRODUCTS
    assert round(pct, 1) == PARETO_PCT
    assert round(pct) == 22
    assert curve["cum_revenue_pct"].iloc[-1] == pytest.approx(100)


@pytest.mark.parametrize("label, exclude", [("all cohorts", []), ("excl. Dec 2009", [DEC_2009])])
def test_cumulative_ltv(cust, label, exclude):
    ltv = an.cumulative_ltv(cust, exclude_cohorts=exclude)
    for months, value in CUM_LTV[label].items():
        assert round(ltv.loc[months, "avg_cum_revenue"], 1) == value


def test_readme_ltv_figures(cust):
    ltv = an.cumulative_ltv(cust, exclude_cohorts=[DEC_2009])
    assert f"{ltv.loc[3, 'avg_cum_revenue']:,.0f}" == "634"
    assert f"{ltv.loc[12, 'avg_cum_revenue']:,.0f}" == "1,373"


def test_historical_ltv(cust, rfm):
    assert round(cust.groupby("Customer ID")["line_revenue"].sum().mean(), 1) == HISTORICAL_LTV["all customers"]
    assert round(rfm["monetary"].mean(), 1) == HISTORICAL_LTV["scored customers"]
    assert round(rfm["monetary"].median(), 1) == HISTORICAL_LTV["median scored"]
    seg = an.segment_summary(rfm)["avg_revenue_per_customer"].round(1)
    assert seg.to_dict() == SEGMENT_LTV


def test_uk_share(products):
    # README: "The UK accounts for 85%"
    countries = an.top_by_revenue(products, "Country", 1)
    assert countries.index[0] == "United Kingdom"
    assert round(countries["share_pct"].iloc[0], 1) == 85.4
    assert np.isclose(countries["net_revenue"].iloc[0] / products["line_revenue"].sum() * 100,
                      countries["share_pct"].iloc[0])
