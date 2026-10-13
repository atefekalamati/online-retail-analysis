"""Load the Online Retail II dataset from Excel and cache it as parquet.

No rows are dropped or modified here. This module only reads, types and saves.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "online_retail_raw.parquet"


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read both Excel sheets, stack them and set column types.

    Adds a ``source_sheet`` column with the name of the sheet each row came from.
    Invoice, StockCode, Description and Country become strings (some Description
    cells are numbers in Excel), Customer ID nullable Int64, InvoiceDate datetime.
    """
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str},
        engine="openpyxl",
    )
    df = pd.concat(
        [frame.assign(source_sheet=name) for name, frame in sheets.items()],
        ignore_index=True,
    )
    text_cols = ["Invoice", "StockCode", "Description", "Country"]
    df[text_cols] = df[text_cols].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["source_sheet"] = df["source_sheet"].astype("category")
    return df


def save_processed(df: pd.DataFrame, path: str | Path = PROCESSED_PATH) -> Path:
    """Save the dataframe as parquet and return the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_processed(path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    """Read the cached parquet file back into a dataframe."""
    return pd.read_parquet(path)
