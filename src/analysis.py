"""Reusable analysis logic: KPIs, trends, Pareto, cohorts, RFM, LTV, chart style.

All functions expect the cleaned data from ``src/clean_data.py`` and are pure:
they return new dataframes and never modify their input.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
POWERBI_DIR = PROJECT_ROOT / "data" / "processed" / "powerbi"

# ---------- chart style (validated default palette, light mode) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF_LINE = "#9a9893"   # reference lines (targets, thresholds)
BLUE = "#2a78d6"     # series 1
ORANGE = "#eb6834"   # series 2
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    """Set one simple matplotlib style for every chart in the project."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE]),
    })


def gbp(x: float, _pos: object = None) -> str:
    """Compact GBP label: £950, £1.4K, £12K, £1.2M."""
    a = abs(x)
    if a >= 1e6:
        return f"£{x / 1e6:.1f}M"
    if 1e3 <= a < 1e4:
        return f"£{x / 1e3:.1f}K"
    if a >= 1e3:
        return f"£{x / 1e3:.0f}K"
    return f"£{x:.0f}"


GBP_FMT = FuncFormatter(gbp)


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG in reports/figures and return the path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path


# ---------- KPIs and trends ----------

def products_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real products only (is_product == True)."""
    return df[df["is_product"]]


def fully_cancelled_invoices(df: pd.DataFrame) -> set[str]:
    """Purchase invoices whose every line was later reversed by a matching 'C' invoice.

    A cancellation line matches a purchase line with the same Customer ID, StockCode
    and absolute Quantity, dated at or before the cancellation. Matching is one-to-one:
    each cancellation line is paired with the most recent unmatched purchase line.
    Rows without a Customer ID cannot be matched and are never counted as cancelled.
    """
    keys = ["Customer ID", "StockCode", "qty"]
    d = df[df["has_customer"]].assign(qty=lambda x: x["Quantity"].abs())
    sales = d[~d["is_cancelled"]].sort_values("InvoiceDate")
    cancels = d[d["is_cancelled"]].sort_values("InvoiceDate")
    pool: dict[tuple, list[tuple[pd.Timestamp, int]]] = {}
    for idx, key, date in zip(sales.index, sales[keys].itertuples(index=False, name=None), sales["InvoiceDate"]):
        pool.setdefault(key, []).append((date, idx))
    matched: set[int] = set()
    for key, date in zip(cancels[keys].itertuples(index=False, name=None), cancels["InvoiceDate"]):
        candidates = pool.get(key, [])
        for i in range(len(candidates) - 1, -1, -1):          # most recent first
            if candidates[i][0] <= date:
                matched.add(candidates.pop(i)[1])
                break
    line_matched = pd.Series(sales.index.isin(list(matched)), index=sales.index)
    all_matched = line_matched.groupby(sales["Invoice"]).all()
    return set(all_matched[all_matched].index)


def kpis(df: pd.DataFrame, by: str | pd.Series | None = None,
         exclude_orders: set[str] | None = None) -> pd.DataFrame:
    """Net and gross revenue, cancellation rate, orders, customers and AOV.

    Orders = distinct non-cancelled invoices, minus ``exclude_orders`` (fully cancelled
    orders). AOV = gross revenue of those orders / orders. Gross revenue, cancellations
    and the cancellation rate still include every row. Customers = distinct Customer IDs
    on non-cancelled rows. ``by`` groups the result.
    """
    keys = by if by is not None else pd.Series("All time", index=df.index, name="period")
    if isinstance(keys, str):
        keys = df[keys]
    sales = df[~df["is_cancelled"]]
    kept = sales[~sales["Invoice"].isin(exclude_orders or set())]
    g_all = df.groupby(keys, observed=True)
    g_sales = sales.groupby(keys[sales.index], observed=True)
    g_kept = kept.groupby(keys[kept.index], observed=True)
    out = pd.DataFrame({
        "net_revenue": g_all["line_revenue"].sum(),
        "gross_revenue": g_sales["line_revenue"].sum(),
        "orders": g_kept["Invoice"].nunique(),
        "fully_cancelled_orders": g_sales["Invoice"].nunique() - g_kept["Invoice"].nunique(),
        "customers": g_sales["Customer ID"].nunique(),
        "order_revenue": g_kept["line_revenue"].sum(),
    })
    out["cancellations"] = out["net_revenue"] - out["gross_revenue"]
    out["cancellation_rate_pct"] = -out["cancellations"] / out["gross_revenue"] * 100
    out["avg_order_value"] = out["order_revenue"] / out["orders"]
    return out[["net_revenue", "gross_revenue", "cancellations", "cancellation_rate_pct",
                "orders", "fully_cancelled_orders", "customers", "avg_order_value"]]


def monthly_kpis(df: pd.DataFrame, exclude_orders: set[str] | None = None) -> pd.DataFrame:
    """KPIs per invoice_month, MoM growth of net revenue and the is_partial_month flag.

    MoM growth is left empty for the partial month so it never shows up as a fake drop.
    """
    m = kpis(df, by=df["invoice_month"].rename("month"), exclude_orders=exclude_orders).reset_index()
    partial = df.groupby("invoice_month")["is_partial_month"].first()
    m["is_partial_month"] = m["month"].map(partial)
    m["mom_growth_pct"] = m["net_revenue"].pct_change() * 100
    m.loc[m["is_partial_month"], "mom_growth_pct"] = np.nan
    return m


def calendar_month_comparison(monthly: pd.DataFrame, years: tuple[int, int] = (2010, 2011)) -> pd.DataFrame:
    """Net revenue by calendar month for two years side by side, plus YoY growth %."""
    full = monthly[~monthly["is_partial_month"]]
    t = full.assign(year=full["month"].dt.year, cal_month=full["month"].dt.month)
    t = t[t["year"].isin(years)].pivot(index="cal_month", columns="year", values="net_revenue")
    t["yoy_growth_pct"] = (t[years[1]] / t[years[0]] - 1) * 100
    t.index = pd.to_datetime(t.index, format="%m").strftime("%b")
    return t


# ---------- countries, products, Pareto ----------

def top_by_revenue(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Top n values of ``col`` by net revenue, with share of total net revenue."""
    rev = df.groupby(col, observed=True)["line_revenue"].sum().sort_values(ascending=False)
    out = rev.head(n).rename("net_revenue").to_frame()
    out["share_pct"] = out["net_revenue"] / rev.sum() * 100
    return out


def pareto(df: pd.DataFrame, col: str = "StockCode", target: float = 0.8) -> tuple[pd.DataFrame, float]:
    """Cumulative revenue curve over items sorted by net revenue.

    Returns the curve and the % of items needed to reach ``target`` of total net revenue.
    """
    rev = df.groupby(col)["line_revenue"].sum().sort_values(ascending=False)
    curve = pd.DataFrame({
        "item_pct": np.arange(1, len(rev) + 1) / len(rev) * 100,
        "cum_revenue_pct": rev.cumsum().to_numpy() / rev.sum() * 100,
    }, index=rev.index)
    n_needed = int((curve["cum_revenue_pct"] < target * 100).sum()) + 1
    return curve, n_needed / len(rev) * 100


# ---------- cohorts ----------

def month_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole calendar months between two month-start timestamps."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def with_cohort(cust: pd.DataFrame) -> pd.DataFrame:
    """Add cohort (month of first purchase) and month_index to full-month rows.

    The first purchase is taken from non-cancelled rows. Partial-month rows are dropped.
    """
    full = cust[~cust["is_partial_month"]]
    first = full[~full["is_cancelled"]].groupby("Customer ID")["invoice_month"].min().rename("cohort")
    out = full.join(first, on="Customer ID", how="inner")
    return out.assign(month_index=month_diff(out["invoice_month"], out["cohort"]))


def cohort_retention(cust: pd.DataFrame) -> pd.DataFrame:
    """Retention % matrix: rows = cohort month, columns = months since first purchase."""
    c = with_cohort(cust)
    c = c[~c["is_cancelled"]]
    active = c.groupby(["cohort", "month_index"])["Customer ID"].nunique().unstack()
    return active.div(active[0], axis=0) * 100


def cohort_sizes(cust: pd.DataFrame) -> pd.Series:
    """Number of new customers per cohort month."""
    c = with_cohort(cust)
    return c[~c["is_cancelled"]].groupby("cohort")["Customer ID"].nunique().rename("customers")


def mean_retention(matrix: pd.DataFrame, months: list[int], exclude: list[pd.Timestamp] | None = None) -> pd.Series:
    """Simple average retention across cohorts that have data for each month index."""
    m = matrix.drop(index=exclude or [], errors="ignore")
    return pd.Series({k: m[k].mean() for k in months}, name="avg_retention_pct")


# ---------- RFM ----------

RFM_SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions", "R >= 4 and F >= 4 and M >= 4"),
    ("Cannot Lose Them", "R <= 2 and F >= 4 and M >= 4"),
    ("Loyal", "R >= 3 and F >= 4"),
    ("At Risk", "R <= 2 and F >= 3"),
    ("New", "R >= 4 and F == 1"),
    ("Potential Loyalists", "R >= 4 and F >= 2"),
    ("Hibernating", "R == 3"),
    ("Lost", "R <= 2"),
]
SEGMENT_ORDER = [name for name, _ in RFM_SEGMENT_RULES]


def rfm_table(cust: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.Timestamp]:
    """Recency, frequency and net monetary value per customer.

    Snapshot = last InvoiceDate + 1 day. Customers with M <= 0 are dropped.
    Returns the table, the number of dropped customers and the snapshot date.
    """
    snapshot = cust["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    sales = cust[~cust["is_cancelled"]]
    rfm = pd.DataFrame({
        "recency_days": (snapshot - sales.groupby("Customer ID")["InvoiceDate"].max()).dt.days,
        "frequency": sales.groupby("Customer ID")["Invoice"].nunique(),
        "monetary": cust.groupby("Customer ID")["line_revenue"].sum(),
    })
    keep = rfm["monetary"] > 0
    return rfm[keep].astype({"recency_days": int, "frequency": int}), int((~keep).sum()), snapshot


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add 1-5 quintile scores (5 = best) and the segment from RFM_SEGMENT_RULES.

    Recency: fewer days is better. Frequency is ranked first so ties split evenly.
    """
    out = rfm.copy()
    out["R"] = pd.qcut(out["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_score"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    out["segment"] = None
    for name, rule in RFM_SEGMENT_RULES:
        hit = out["segment"].isna() & out.eval(rule)
        out.loc[hit, "segment"] = name
    out["segment"] = pd.Categorical(out["segment"], categories=SEGMENT_ORDER, ordered=True)
    return out


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Customers, revenue, shares and average R/F/M per segment."""
    s = rfm.groupby("segment", observed=False).agg(
        customers=("monetary", "size"), revenue=("monetary", "sum"),
        avg_recency_days=("recency_days", "mean"), avg_frequency=("frequency", "mean"),
        avg_revenue_per_customer=("monetary", "mean"),
    )
    s.insert(1, "customers_pct", s["customers"] / s["customers"].sum() * 100)
    s.insert(3, "revenue_pct", s["revenue"] / s["revenue"].sum() * 100)
    return s


def bulk_order_invoices(df: pd.DataFrame, min_quantity: int = 50_000) -> list[str]:
    """Invoices with a line of |Quantity| >= min_quantity (one-off bulk orders and their cancellations)."""
    return sorted(df.loc[df["Quantity"].abs() >= min_quantity, "Invoice"].unique())


def top_share(values: pd.Series, top_frac: float = 0.2) -> float:
    """% of the total contributed by the top ``top_frac`` of values."""
    v = values.sort_values(ascending=False)
    n = int(np.ceil(len(v) * top_frac))
    return v.head(n).sum() / v.sum() * 100


# ---------- LTV ----------

def cumulative_ltv(cust: pd.DataFrame, horizons: tuple[int, ...] = (3, 6, 12),
                   exclude_cohorts: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Average cumulative net revenue per customer in their first k months.

    "First k months" = month_index 0 .. k-1, the first-purchase month included.
    Only cohorts whose k-th month is complete in the data are used for each k.
    """
    c = with_cohort(cust)
    c = c[~c["cohort"].isin(exclude_cohorts or [])]
    last_full = c["invoice_month"].max()
    per_cust = c.groupby(["Customer ID", "cohort", "month_index"])["line_revenue"].sum().reset_index()
    rows = []
    for k in horizons:
        eligible = per_cust[month_diff(pd.Series(last_full, index=per_cust.index), per_cust["cohort"]) >= k - 1]
        rev = eligible[eligible["month_index"] < k].groupby("Customer ID")["line_revenue"].sum()
        n_customers = eligible["Customer ID"].nunique()
        rows.append({
            "months": k,
            "avg_cum_revenue": rev.reindex(eligible["Customer ID"].unique(), fill_value=0).mean(),
            "customers": n_customers,
            "cohorts": eligible["cohort"].nunique(),
            "last_cohort": eligible["cohort"].max().strftime("%Y-%m"),
        })
    return pd.DataFrame(rows).set_index("months")


# ---------- export ----------

def to_csv(df: pd.DataFrame, name: str) -> Path:
    """Write a CSV for Power BI to data/processed/powerbi and return the path.

    Floats are rounded to 4 decimals. Date columns with no time part are written as YYYY-MM-DD.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]) and (out[col].dt.normalize() == out[col]).all():
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.round(4)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    path = POWERBI_DIR / f"{name}.csv"
    out.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
    return path
