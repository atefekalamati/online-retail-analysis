"""Cleaning rules: one test per step of reports/cleaning_log.md, each on a tiny hand-made DataFrame.

The last tests check that the code and the committed cleaning log describe the same steps,
and (with the raw data) that re-running the pipeline gives exactly the logged row counts.
"""
import pandas as pd
import pytest

from src import clean_data as cd

from conftest import make_rows, to_number


# ---------- step 1: sheet overlap ----------

def test_step1_drops_second_sheet_rows_before_2010_12_10():
    df = make_rows([
        {"source_sheet": "Year 2009-2010", "InvoiceDate": "2010-12-05"},     # kept: first sheet
        {"source_sheet": "Year 2010-2011", "InvoiceDate": "2010-12-05"},     # dropped: overlap
        {"source_sheet": "Year 2010-2011", "InvoiceDate": "2010-12-09 23:59"},  # dropped: last overlap minute
        {"source_sheet": "Year 2010-2011", "InvoiceDate": "2010-12-10 00:00"},  # kept: after the overlap
    ])
    out = cd.drop_sheet_overlap(df)
    assert list(out.index) == [0, 3]


# ---------- step 2: exact duplicates ----------

def test_step2_drops_exact_duplicates_ignoring_source_sheet():
    df = make_rows([
        {"Invoice": "1", "source_sheet": "Year 2009-2010"},
        {"Invoice": "1", "source_sheet": "Year 2010-2011"},   # same row from the other sheet: dropped
        {"Invoice": "1", "source_sheet": "Year 2009-2010"},   # plain duplicate: dropped
        {"Invoice": "1", "Quantity": 2},                       # differs in Quantity: kept
    ])
    out = cd.drop_exact_duplicates(df)
    assert list(out.index) == [0, 3]


def test_step2_keeps_rows_that_differ_only_in_one_field():
    base = {"Invoice": "1"}
    df = make_rows([base, {**base, "Price": 2.56}, {**base, "Customer ID": None},
                    {**base, "InvoiceDate": "2011-01-10 10:01"}])
    assert len(cd.drop_exact_duplicates(df)) == 4


# ---------- step 3: StockCode normalisation ----------

def test_step3_uppercases_and_strips_stockcode():
    df = make_rows([{"StockCode": " 85123a "}, {"StockCode": "post"}, {"StockCode": "22423"}])
    out = cd.normalize_stockcode(df)
    assert list(out["StockCode"]) == ["85123A", "POST", "22423"]
    assert len(out) == len(df)


# ---------- step 4: stock adjustments ----------

def test_step4_drops_negative_quantity_without_cancellation_invoice():
    df = make_rows([
        {"Invoice": "500001", "Quantity": -5},     # stock adjustment: dropped
        {"Invoice": "C500002", "Quantity": -5},    # cancellation: kept
        {"Invoice": "500003", "Quantity": 5},      # sale: kept
        {"Invoice": "500004", "Quantity": 0},      # zero is not negative: kept
    ])
    out = cd.drop_stock_adjustments(df)
    assert list(out["Invoice"]) == ["C500002", "500003", "500004"]


# ---------- step 5a: zero or negative price ----------

def test_step5a_drops_zero_and_negative_price():
    df = make_rows([{"Price": 0.0}, {"Price": -11062.06}, {"Price": 0.001}, {"Price": 2.55}])
    out = cd.drop_nonpositive_price(df)
    assert list(out["Price"]) == [0.001, 2.55]


# ---------- step 5b: bad debt ----------

def test_step5b_drops_bad_debt_invoices():
    df = make_rows([{"Invoice": "A563185", "StockCode": "B"}, {"Invoice": "C500001"}, {"Invoice": "500001"}])
    out = cd.drop_bad_debt(df)
    assert list(out["Invoice"]) == ["C500001", "500001"]


# ---------- step 6: flags (cancellations, missing customer ID, non-product codes) ----------

def test_step6_flags_cancellations_and_keeps_them():
    df = make_rows([{"Invoice": "C500001", "Quantity": -2}, {"Invoice": "500001"}])
    out = cd.add_flags(df)
    assert list(out["is_cancelled"]) == [True, False]
    assert len(out) == 2                                   # flagged, not removed


def test_step6_flags_missing_customer_id_and_keeps_it():
    df = make_rows([{"Customer ID": None}, {"Customer ID": 12346}])
    out = cd.add_flags(df)
    assert list(out["has_customer"]) == [False, True]
    assert len(out) == 2


@pytest.mark.parametrize("code", sorted(cd.NON_PRODUCT_CODES) + ["GIFT_0001_20", "TEST001"])
def test_step6_non_product_codes(code):
    out = cd.add_flags(make_rows([{"StockCode": code}]))
    assert not out["is_product"].iloc[0]


@pytest.mark.parametrize("code", ["85123A", "22423", "DCGS0003", "DCGSSBOY", "PADS", "SP1002",
                                  "84879", "MUG"])   # MUG: starts with M but is not the M code
def test_step6_product_codes(code):
    out = cd.add_flags(make_rows([{"StockCode": code}]))
    assert out["is_product"].iloc[0]


def test_step6_flags_are_plain_booleans():
    out = cd.add_flags(make_rows([{"Customer ID": None}, {}]))
    for col in ["is_cancelled", "has_customer", "is_product"]:
        assert out[col].dtype == bool


# ---------- step 7: standard description ----------

def test_step7_uses_most_frequent_description_per_stockcode():
    df = make_rows([
        {"StockCode": "A1", "Description": "RED BAG "},
        {"StockCode": "A1", "Description": "RED BAG"},
        {"StockCode": "A1", "Description": "RED SPOTTY BAG"},
        {"StockCode": "A1", "Description": None},              # missing: filled
        {"StockCode": "B2", "Description": "ZEBRA"},           # tie: alphabetically first wins
        {"StockCode": "B2", "Description": "APPLE"},
    ])
    out = cd.standardize_description(df)
    assert list(out["Description"]) == ["RED BAG"] * 4 + ["APPLE"] * 2
    assert out["description_raw"].iloc[0] == "RED BAG "        # original kept
    assert pd.isna(out["description_raw"].iloc[3])
    assert len(out) == len(df)


# ---------- step 8: derived columns ----------

def test_step8_line_revenue_month_and_partial_month():
    df = make_rows([
        {"Quantity": 3, "Price": 2.5, "InvoiceDate": "2011-11-30 18:00"},
        {"Quantity": -2, "Price": 4.0, "Invoice": "C1", "InvoiceDate": "2011-12-09 12:50"},
    ])
    out = cd.add_derived_columns(df)
    assert list(out["line_revenue"]) == [7.5, -8.0]           # cancellations are negative revenue
    assert list(out["invoice_month"]) == [pd.Timestamp("2011-11-01"), pd.Timestamp("2011-12-01")]
    assert list(out["is_partial_month"]) == [False, True]


# ---------- step 9: outliers ----------

def test_step9_flags_but_does_not_remove_outliers():
    quantities = [1] * 9_998 + [80_995, -80_995]      # 99.9th percentile of |Quantity| is 1
    df = make_rows([{"Quantity": q} for q in quantities])
    out = cd.flag_outliers(df)
    assert len(out) == len(df)
    assert out["is_outlier"].sum() == 2
    assert out.loc[out["is_outlier"], "Quantity"].abs().eq(80_995).all()


# ---------- customer subset used for cohorts, RFM and LTV ----------

def test_customer_subset_needs_customer_and_product():
    df = cd.add_flags(make_rows([
        {"Customer ID": 1, "StockCode": "85123A"},                 # kept
        {"Customer ID": None, "StockCode": "85123A"},              # no customer
        {"Customer ID": 1, "StockCode": "POST"},                   # not a product
        {"Customer ID": 1, "Invoice": "C9", "Quantity": -1},       # cancellation: kept as negative row
    ]))
    out = cd.customer_subset(df)
    assert list(out["Invoice"]) == ["500001", "C9"]


# ---------- the whole pipeline ----------

def test_run_cleaning_logs_every_step_on_a_tiny_frame():
    df = make_rows([
        {"Invoice": "1"},
        {"Invoice": "1", "source_sheet": "Year 2009-2010"},             # duplicate
        {"Invoice": "2", "source_sheet": "Year 2010-2011", "InvoiceDate": "2010-12-01"},  # overlap
        {"Invoice": "3", "Quantity": -1},                                # stock adjustment
        {"Invoice": "4", "Price": 0.0},                                  # price 0
        {"Invoice": "A5", "StockCode": "B", "Price": -100.0},            # bad debt (negative price)
        {"Invoice": "A6", "StockCode": "B", "Price": 100.0},             # bad debt
        {"Invoice": "C7", "Quantity": -1, "Customer ID": None},          # cancellation, kept
    ])
    result = cd.run_cleaning(df)
    removed = result.log.set_index("step")["rows removed"].to_dict()
    assert removed == {"1": 1, "2": 1, "3": 0, "4": 1, "5a": 2, "5b": 1,
                       "6": 0, "7": 0, "8": 0, "9": 0}
    assert list(result.df["Invoice"]) == ["1", "C7"]
    log = result.log
    assert (log["rows before"] - log["rows removed"] == log["rows after"]).all()
    assert (log["rows after"].iloc[:-1].to_numpy() == log["rows before"].iloc[1:].to_numpy()).all()


def test_cleaning_log_lists_the_same_steps_as_the_code(cleaning_log_steps):
    assert list(cleaning_log_steps["step"]) == [s for s, _, _ in cd.STEPS]
    assert list(cleaning_log_steps["rule"]) == [r for _, r, _ in cd.STEPS]


def test_cleaning_log_row_counts_add_up(cleaning_log_steps):
    log = cleaning_log_steps.set_index("step")[["rows before", "rows removed", "rows after"]].map(to_number)
    assert (log["rows before"] - log["rows removed"] == log["rows after"]).all()
    assert log["rows before"].iloc[0] == 1_067_371
    assert log["rows removed"].sum() == 40_355                 # README: 3.8% of rows removed
    assert log["rows after"].iloc[-1] == 1_027_016
    assert round(40_355 / 1_067_371 * 100, 1) == 3.8


def test_pipeline_reproduces_the_cleaning_log(cleaning, cleaning_log_steps):
    expected = cleaning_log_steps[["rows before", "rows removed", "rows after"]].map(to_number)
    actual = cleaning.log[["rows before", "rows removed", "rows after"]].astype(float)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_clean_data_satisfies_every_rule(clean):
    assert not ((clean["source_sheet"] == cd.OVERLAP_SHEET) & (clean["InvoiceDate"] < cd.OVERLAP_END)).any()
    assert (clean["StockCode"] == clean["StockCode"].str.strip().str.upper()).all()
    assert not ((clean["Quantity"] < 0) & ~clean["is_cancelled"]).any()
    assert (clean["Price"] > 0).all()
    assert not clean["Invoice"].str.startswith("A").any()
    assert clean["Description"].notna().all()
    assert (clean["line_revenue"] == clean["Quantity"] * clean["Price"]).all()


def test_customer_subset_size(cust):
    # cleaning_log.md: 794,183 rows, 5,875 customers
    assert len(cust) == 794_183
    assert cust["Customer ID"].nunique() == 5_875
