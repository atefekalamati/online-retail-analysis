"""Cleaning rules for Online Retail II.

Each step takes a dataframe and returns a new one. ``run_cleaning`` applies the
steps in order and records rows before/removed/after for the cleaning log.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "transactions_clean.parquet"
CUSTOMER_PATH = PROCESSED_DIR / "customer_transactions.parquet"

OVERLAP_SHEET = "Year 2010-2011"
OVERLAP_END = pd.Timestamp("2010-12-10")
PARTIAL_MONTH = pd.Timestamp("2011-12-01")
OUTLIER_QUANTILE = 0.999

# Service, fee and adjustment codes. Codes starting with GIFT or TEST are added by pattern.
# DCGS*, PADS and SP1002 are real items sold through the web shop and count as products.
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "C3",          # postage / carriage
    "M", "D", "S",                       # manual, discount, samples
    "BANK CHARGES", "AMAZONFEE", "CRUK", # fees and commissions
    "ADJUST", "ADJUST2", "B",            # adjustments
}
NON_PRODUCT_PATTERN = r"^(GIFT|TEST)"
POSTAGE_CODES = {"POST", "DOT", "C2", "C3"}
FEE_CODES = {"AMAZONFEE", "BANK CHARGES", "CRUK"}


# ---------- steps ----------

def drop_sheet_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of the 2010-2011 sheet dated before 2010-12-10 (already in the 2009-2010 sheet)."""
    overlap = (df["source_sheet"] == OVERLAP_SHEET) & (df["InvoiceDate"] < OVERLAP_END)
    return df[~overlap]


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical on every column except source_sheet, keeping the first."""
    cols = [c for c in df.columns if c != "source_sheet"]
    return df.drop_duplicates(subset=cols, keep="first")


def normalize_stockcode(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and strip StockCode."""
    return df.assign(StockCode=df["StockCode"].str.strip().str.upper())


def drop_stock_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Drop negative quantities that are not cancellations."""
    return df[~((df["Quantity"] < 0) & ~df["Invoice"].str.startswith("C"))]


def drop_nonpositive_price(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with Price <= 0."""
    return df[df["Price"] > 0]


def drop_bad_debt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad-debt adjustment invoices (Invoice starts with 'A')."""
    return df[~df["Invoice"].str.startswith("A")]


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_cancelled, has_customer and is_product."""
    code = df["StockCode"]
    non_product = code.isin(NON_PRODUCT_CODES) | code.str.match(NON_PRODUCT_PATTERN)
    return df.assign(
        is_cancelled=df["Invoice"].str.startswith("C").astype(bool),
        has_customer=df["Customer ID"].notna(),
        is_product=~non_product.astype(bool),
    )


def standardize_description(df: pd.DataFrame) -> pd.DataFrame:
    """Set Description to the most frequent (stripped) description of its StockCode.

    The original text is kept in description_raw. Ties go to the alphabetically first one.
    """
    stripped = df["Description"].str.strip()
    canonical = (
        stripped.dropna().groupby(df["StockCode"]).value_counts()
        .rename("n").reset_index()
        .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    return df.assign(description_raw=df["Description"],
                     Description=df["StockCode"].map(canonical).astype("string"))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add line_revenue, invoice_month and is_partial_month."""
    month = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return df.assign(
        line_revenue=df["Quantity"] * df["Price"],
        invoice_month=month,
        is_partial_month=month == PARTIAL_MONTH,
    )


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with |Quantity| above the 99.9th percentile of |Quantity|. Nothing is removed."""
    threshold = df["Quantity"].abs().quantile(OUTLIER_QUANTILE)
    return df.assign(is_outlier=df["Quantity"].abs() > threshold)


STEPS: list[tuple[str, str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("1", "Drop sheet overlap: 'Year 2010-2011' rows with InvoiceDate < 2010-12-10", drop_sheet_overlap),
    ("2", "Drop exact duplicate rows (all columns except source_sheet)", drop_exact_duplicates),
    ("3", "Uppercase and strip StockCode", normalize_stockcode),
    ("4", "Drop Quantity < 0 where Invoice does not start with 'C' (stock adjustments)", drop_stock_adjustments),
    ("5a", "Drop Price <= 0", drop_nonpositive_price),
    ("5b", "Drop invoices starting with 'A' (bad debt adjustments)", drop_bad_debt),
    ("6", "Add flags is_cancelled, has_customer, is_product", add_flags),
    ("7", "Description = most frequent Description per StockCode (fills missing)", standardize_description),
    ("8", "Add line_revenue, invoice_month, is_partial_month", add_derived_columns),
    ("9", "Flag is_outlier: |Quantity| > 99.9th percentile (not removed)", flag_outliers),
]


# ---------- pipeline ----------

@dataclass
class CleaningResult:
    """Cleaned data plus one log row per step."""
    df: pd.DataFrame
    log: pd.DataFrame
    notes: dict[str, object] = field(default_factory=dict)


def run_cleaning(raw: pd.DataFrame) -> CleaningResult:
    """Apply every step in STEPS and log rows before, removed and after."""
    df, rows, notes = raw, [], {}
    for step, rule, fn in STEPS:
        before = len(df)
        if fn is standardize_description:
            notes["descriptions_missing_before"] = int(df["Description"].isna().sum())
        df = fn(df)
        rows.append({"step": step, "rule": rule, "rows before": before,
                     "rows removed": before - len(df), "rows after": len(df)})
    notes["descriptions_missing_after"] = int(df["Description"].isna().sum())
    notes["descriptions_changed"] = int(
        (df["Description"].fillna("") != df["description_raw"].fillna("")).sum())
    notes["outlier_threshold"] = float(df["Quantity"].abs().quantile(OUTLIER_QUANTILE))
    return CleaningResult(df.reset_index(drop=True), pd.DataFrame(rows), notes)


def customer_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for cohort/RFM/LTV: known customer and a real product."""
    return df[df["has_customer"] & df["is_product"]].reset_index(drop=True)


def revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Gross, cancellations, net, postage/fee/other non-product shares and no-customer share.

    Shares are of net revenue unless stated. Non-product groups are net amounts
    (their cancellations and credits included), so fees show up as negative.
    """
    rev = df["line_revenue"]
    gross = rev[~df["is_cancelled"]].sum()
    cancelled = rev[df["is_cancelled"]].sum()
    net = gross + cancelled
    code = df["StockCode"]
    postage = rev[code.isin(POSTAGE_CODES)].sum()
    fees = rev[code.isin(FEE_CODES)].sum()
    other = rev[~df["is_product"] & ~code.isin(POSTAGE_CODES | FEE_CODES)].sum()
    products = rev[df["is_product"]].sum()
    no_cust_gross = rev[~df["has_customer"] & ~df["is_cancelled"]].sum()
    no_cust_net = rev[~df["has_customer"]].sum()
    items = [
        ("Gross revenue (excl. cancellations)", gross, f"{gross / gross:.1%} of gross"),
        ("Cancellations", cancelled, f"{cancelled / gross:.1%} of gross"),
        ("Net revenue", net, f"{net / net:.1%} of net"),
        ("  Products, net", products, f"{products / net:.1%} of net"),
        ("  Postage (POST, DOT, C2, C3), net", postage, f"{postage / net:.1%} of net"),
        ("  Fees (AMAZONFEE, BANK CHARGES, CRUK), net", fees, f"{fees / net:.1%} of net"),
        ("  Other non-product (M, D, S, ADJUST, vouchers, TEST), net", other, f"{other / net:.1%} of net"),
        ("Revenue with no Customer ID, gross", no_cust_gross, f"{no_cust_gross / gross:.1%} of gross"),
        ("Revenue with no Customer ID, net", no_cust_net, f"{no_cust_net / net:.1%} of net"),
    ]
    out = pd.DataFrame(items, columns=["Metric", "GBP", "Share"])
    out["GBP"] = out["GBP"].map("{:,.0f}".format)
    return out


def save(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe as parquet and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
