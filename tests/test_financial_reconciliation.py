"""Financial reconciliation: revenue, orders, customers, AOV and cancellation rates.

Cancellation rows carry negative revenue, so "net = gross - cancellations" is computed as
gross + cancellations, where cancellations <= 0.

The report tests read the committed tables and always run. The data tests recompute the
same figures from the raw data and check that they match the reports.
"""
import pandas as pd
import pytest

from src import analysis as an
from src import clean_data as cd

from conftest import to_number

# All time, products only: README "All-time KPIs" and reports/tables/kpis.csv
NET_REVENUE = 18_927_399
ORDERS = 39_246
CUSTOMERS = 5_852
AOV = 489
FULLY_CANCELLED_ORDERS = 270
CANCEL_RATE_ALL_CODES = 7.1        # % of gross revenue, cleaning_log.md
CANCEL_RATE_PRODUCTS = 3.6         # % of gross revenue, README


def cancellation_rate(df: pd.DataFrame) -> float:
    gross = df.loc[~df["is_cancelled"], "line_revenue"].sum()
    cancelled = df.loc[df["is_cancelled"], "line_revenue"].sum()
    return -cancelled / gross * 100


# ---------- kpis() on a tiny frame (always runs) ----------

def test_kpis_on_hand_made_orders():
    df = cd.add_derived_columns(cd.add_flags(pd.DataFrame({
        "Invoice": pd.array(["1", "1", "2", "C3", "4"], dtype="string"),
        "StockCode": pd.array(["A", "B", "A", "A", "A"], dtype="string"),
        "Quantity": [2, 1, 5, -5, 10],
        "Price": [10.0, 20.0, 4.0, 4.0, 1.0],
        "Customer ID": pd.array([1, 1, 2, 2, None], dtype="Int64"),
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"]),
    })))
    fully_cancelled = an.fully_cancelled_invoices(df)
    assert fully_cancelled == {"2"}
    k = an.kpis(df, exclude_orders=fully_cancelled).loc["All time"]
    assert k["gross_revenue"] == 40 + 20 + 10           # invoices 1, 2 and 4
    assert k["cancellations"] == -20
    assert k["net_revenue"] == k["gross_revenue"] + k["cancellations"] == 50
    assert k["cancellation_rate_pct"] == pytest.approx(20 / 70 * 100)
    assert k["orders"] == 2                              # invoice 2 was fully cancelled
    assert k["fully_cancelled_orders"] == 1
    assert k["customers"] == 2                           # missing Customer ID is not a customer
    assert k["avg_order_value"] == (40 + 10) / 2


def test_partially_cancelled_order_still_counts():
    df = cd.add_flags(pd.DataFrame({
        "Invoice": pd.array(["1", "1", "C2"], dtype="string"),
        "StockCode": pd.array(["A", "B", "A"], dtype="string"),
        "Quantity": [1, 1, -1],
        "Customer ID": pd.array([1, 1, 1], dtype="Int64"),
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-02"]),
    }))
    assert an.fully_cancelled_invoices(df) == set()


def test_cancellation_before_purchase_does_not_match():
    df = cd.add_flags(pd.DataFrame({
        "Invoice": pd.array(["C1", "2"], dtype="string"),
        "StockCode": pd.array(["A", "A"], dtype="string"),
        "Quantity": [-1, 1],
        "Customer ID": pd.array([1, 1], dtype="Int64"),
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-02"]),
    }))
    assert an.fully_cancelled_invoices(df) == set()


# ---------- committed reports (always run) ----------

def test_kpi_report_net_equals_gross_minus_cancellations(kpi_report):
    assert (kpi_report["cancellations"] <= 0).all()
    net = kpi_report["gross_revenue"] - kpi_report["cancellations"].abs()
    pd.testing.assert_series_equal(net, kpi_report["net_revenue"], check_names=False, atol=0.01)


def test_kpi_report_all_time_figures(kpi_report):
    k = kpi_report.loc["All time"]
    assert k["net_revenue"] == pytest.approx(NET_REVENUE, abs=1)
    assert k["orders"] == ORDERS
    assert k["customers"] == CUSTOMERS
    assert round(k["avg_order_value"]) == AOV
    assert round(k["cancellation_rate_pct"], 1) == CANCEL_RATE_PRODUCTS
    assert k["fully_cancelled_orders"] == FULLY_CANCELLED_ORDERS


def test_kpi_report_years_add_up_to_all_time(kpi_report):
    years = kpi_report.drop(index="All time")
    assert list(years.index) == ["2009", "2010", "2011"]
    total = kpi_report.loc["All time"]
    for col in ["net_revenue", "gross_revenue", "cancellations"]:
        assert years[col].sum() == pytest.approx(total[col], abs=0.01)
    for col in ["orders", "fully_cancelled_orders"]:              # invoices fall in one year only
        assert years[col].sum() == total[col]
    assert years["customers"].max() <= total["customers"] <= years["customers"].sum()


def test_kpi_report_aov_and_rate_follow_their_definitions(kpi_report):
    rate = -kpi_report["cancellations"] / kpi_report["gross_revenue"] * 100
    pd.testing.assert_series_equal(rate, kpi_report["cancellation_rate_pct"], check_names=False)
    # AOV excludes fully cancelled orders, so orders * AOV is a little below gross revenue
    order_revenue = kpi_report["orders"] * kpi_report["avg_order_value"]
    assert (order_revenue <= kpi_report["gross_revenue"]).all()


def test_cleaning_log_revenue_summary_reconciles(cleaning_log_revenue, kpi_report):
    gbp = cleaning_log_revenue.set_index("Metric")["GBP"].map(to_number)
    share = cleaning_log_revenue.set_index("Metric")["Share"].map(to_number)
    gross, cancelled, net = (gbp["Gross revenue (excl. cancellations)"], gbp["Cancellations"],
                             gbp["Net revenue"])
    assert net == pytest.approx(gross - abs(cancelled), abs=1)          # rounded to whole pounds
    assert share["Cancellations"] == -CANCEL_RATE_ALL_CODES
    assert round(-cancelled / gross * 100, 1) == CANCEL_RATE_ALL_CODES
    parts = ["Products, net", "Postage (POST, DOT, C2, C3), net",
             "Fees (AMAZONFEE, BANK CHARGES, CRUK), net",
             "Other non-product (M, D, S, ADJUST, vouchers, TEST), net"]
    assert gbp[parts].sum() == pytest.approx(net, abs=len(parts))
    assert gbp["Products, net"] == pytest.approx(kpi_report.loc["All time", "net_revenue"], abs=1)


# ---------- recomputed from the raw data ----------

@pytest.fixture(scope="module")
def kpi_recomputed(products, fully_cancelled) -> pd.DataFrame:
    """The KPI table exactly as notebook 03 builds it."""
    k = pd.concat([
        an.kpis(products, exclude_orders=fully_cancelled),
        an.kpis(products, by=products["InvoiceDate"].dt.year.rename("period"), exclude_orders=fully_cancelled),
    ])
    k.index = k.index.astype(str)
    return k


def test_net_revenue_equals_gross_minus_cancellations(products):
    gross = products.loc[~products["is_cancelled"], "line_revenue"].sum()
    cancellations = products.loc[products["is_cancelled"], "line_revenue"].sum()
    net = products["line_revenue"].sum()
    assert cancellations < 0
    assert net == pytest.approx(gross - abs(cancellations), abs=0.01)
    assert net == pytest.approx(NET_REVENUE, abs=1)


def test_orders_customers_and_aov(kpi_recomputed):
    k = kpi_recomputed.loc["All time"]
    assert k["orders"] == ORDERS
    assert k["fully_cancelled_orders"] == FULLY_CANCELLED_ORDERS
    assert k["customers"] == CUSTOMERS
    assert round(k["avg_order_value"]) == AOV


def test_cancellation_rate_all_codes(clean):
    assert round(cancellation_rate(clean), 1) == CANCEL_RATE_ALL_CODES


def test_cancellation_rate_products_only(products):
    assert round(cancellation_rate(products), 1) == CANCEL_RATE_PRODUCTS


def test_kpi_table_matches_report(kpi_recomputed, kpi_report):
    pd.testing.assert_frame_equal(kpi_recomputed, kpi_report, check_names=False, check_dtype=False)


def test_revenue_summary_matches_cleaning_log(clean, cleaning_log_revenue):
    summary = cd.revenue_summary(clean)
    summary["Metric"] = summary["Metric"].str.strip()      # the Markdown table loses the indent
    pd.testing.assert_frame_equal(summary, cleaning_log_revenue)


def test_two_bulk_orders_explain_the_2011_cancellation_spike(products):
    # README finding 5: 2010 rate 2.5%, 2011 rate 4.8%; two bulk orders of about £245K;
    # without them the 2011 rate is 2.3%
    bulk = an.bulk_order_invoices(products)
    assert bulk == ["541431", "581483", "C541433", "C581484"]
    bulk_value = products.loc[products["Invoice"].isin(bulk) & ~products["is_cancelled"], "line_revenue"].sum()
    assert round(bulk_value / 1000) == 246            # £245.7K, "about £245K"
    year = products["InvoiceDate"].dt.year
    assert round(cancellation_rate(products[year == 2010]), 1) == 2.5
    assert round(cancellation_rate(products[year == 2011]), 1) == 4.8
    y2011 = products[year == 2011]
    assert round(cancellation_rate(y2011[~y2011["Invoice"].isin(bulk)]), 1) == 2.3
