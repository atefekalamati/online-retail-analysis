"""Shared fixtures for the test suite.

Three kinds of tests:

- Unit tests on tiny hand-made DataFrames. They always run.
- Report tests that read the committed files in ``reports/``. They always run.
- Data tests that rebuild the numbers from the raw data. They need
  ``data/raw/online_retail_II.xlsx`` and are skipped with a message when it is absent,
  for example when the project was unpacked from the zip.

The data tests read the raw rows from ``data/processed/online_retail_raw.parquet`` when it
exists. That file is the unchanged output of ``load_raw`` on the Excel file (written by
notebook 01), and reading the 45 MB Excel file takes minutes and a lot of memory. Delete the
parquet file to make the tests read the Excel file directly.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from src import analysis as an
from src import clean_data as cd
from src.load_data import PROCESSED_PATH, RAW_PATH, load_processed, load_raw

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
SEMANTIC_MODEL_TABLES = ROOT / "dashboard" / "online_retail.SemanticModel" / "definition" / "tables"

RAW_MISSING = (
    f"Raw data not found: {RAW_PATH.relative_to(ROOT)}. Download Online Retail II from "
    "https://archive.ics.uci.edu/dataset/502/online+retail+ii and place the Excel file there "
    "to run the data tests."
)
POWERBI_MISSING = (
    f"Power BI exports not found in {an.POWERBI_DIR.relative_to(ROOT)}. "
    "Run notebooks 01 to 03 to create them."
)

needs_raw = pytest.mark.skipif(not RAW_PATH.exists(), reason=RAW_MISSING)


# ---------- tiny hand-made data ----------

def make_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-shaped DataFrame (same columns and dtypes as ``load_raw``).

    Missing keys get defaults, so each test only spells out the fields it is about.
    """
    defaults = {
        "Invoice": "500001", "StockCode": "85123A", "Description": "WHITE HANGING HEART",
        "Quantity": 1, "InvoiceDate": "2011-01-10 10:00", "Price": 2.55,
        "Customer ID": 12345, "Country": "United Kingdom", "source_sheet": "Year 2010-2011",
    }
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    text_cols = ["Invoice", "StockCode", "Description", "Country"]
    df[text_cols] = df[text_cols].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="ISO8601")
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["source_sheet"] = df["source_sheet"].astype("category")
    return df


# ---------- real data (skipped without the raw Excel file) ----------

@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """All rows of both Excel sheets, as returned by ``load_raw``."""
    if not RAW_PATH.exists():
        pytest.skip(RAW_MISSING)
    if PROCESSED_PATH.exists():
        return load_processed(PROCESSED_PATH)
    return load_raw(RAW_PATH)


@pytest.fixture(scope="session")
def cleaning(raw) -> cd.CleaningResult:
    """Cleaning pipeline re-run on the raw rows (not read from the cached clean parquet)."""
    return cd.run_cleaning(raw)


@pytest.fixture(scope="session")
def clean(cleaning) -> pd.DataFrame:
    return cleaning.df


@pytest.fixture(scope="session")
def products(clean) -> pd.DataFrame:
    """Product rows, the scope of every analysis in notebook 03."""
    return an.products_only(clean)


@pytest.fixture(scope="session")
def cust(clean) -> pd.DataFrame:
    """Product rows with a Customer ID, used for cohorts, RFM and LTV."""
    return cd.customer_subset(clean)


@pytest.fixture(scope="session")
def fully_cancelled(products) -> set[str]:
    return an.fully_cancelled_invoices(products)


@pytest.fixture(scope="session")
def rfm(cust) -> pd.DataFrame:
    rfm_raw, _, _ = an.rfm_table(cust)
    return an.score_rfm(rfm_raw)


# ---------- committed report tables (always present) ----------

@pytest.fixture(scope="session")
def kpi_report() -> pd.DataFrame:
    """reports/tables/kpis.csv, indexed by period ('All time', '2009', '2010', '2011')."""
    return pd.read_csv(TABLES / "kpis.csv", index_col="period", dtype={"period": str})


@pytest.fixture(scope="session")
def cohort_report() -> pd.DataFrame:
    """reports/tables/cohort_retention_matrix.csv: cohorts x months since first purchase."""
    m = pd.read_csv(TABLES / "cohort_retention_matrix.csv", index_col="cohort", parse_dates=["cohort"])
    m.columns = m.columns.astype(int)
    return m


def markdown_table(path: Path, heading: str) -> pd.DataFrame:
    """Read the first Markdown table after ``heading`` in a report as strings."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            # split on | that is not escaped as \|
            cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
            rows.append([c.strip().replace("\\|", "|") for c in cells])
        elif rows:
            break
    header, _separator, *body = rows
    return pd.DataFrame(body, columns=header)


def to_number(text: str) -> float:
    """'1,067,371' -> 1067371.0, '-7.1% of gross' -> -7.1."""
    return float(text.split("%")[0].split()[0].replace(",", ""))


@pytest.fixture(scope="session")
def cleaning_log_steps() -> pd.DataFrame:
    return markdown_table(REPORTS / "cleaning_log.md", "## Steps")


@pytest.fixture(scope="session")
def cleaning_log_revenue() -> pd.DataFrame:
    return markdown_table(REPORTS / "cleaning_log.md", "## Revenue summary (transactions_clean)")
