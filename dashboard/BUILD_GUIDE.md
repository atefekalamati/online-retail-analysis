# Power BI Build Guide

Builds the two-page dashboard from the CSVs exported by `notebooks/03_analysis.ipynb`.
Files used: `measures.dax` (DAX measures) and `theme.json` (colors, matching the PNG charts).

Before you start, run notebook 03 so that the CSVs in `data/processed/powerbi/` are current.

---

## 1. Load the CSVs and set data types

**Home > Get data > Text/CSV.** Load the five files from `data/processed/powerbi/`.
For each file, choose **Transform Data** rather than Load, then set the types below in Power Query.
Use **File Origin 65001: Unicode (UTF-8)** so the £ signs and accents display correctly.

**fact_sales** (product rows only, about 1M rows)

| Column | Type |
|---|---|
| date | Date/Time |
| invoice, stock_code, description, country | Text |
| customer_id | Whole Number (blank = no customer) |
| quantity | Whole Number |
| price, line_revenue | Decimal Number |
| is_cancelled, has_customer, is_fully_cancelled_order | True/False |

Add a date-only column for the relationship. Select `date`, then **Add Column > Duplicate Column**. Set the new column's type to **Date** and rename it `order_date`.

**monthly_kpis**

| Column | Type |
|---|---|
| month | Date |
| net_revenue, gross_revenue, cancellations, cancellation_rate_pct, avg_order_value, mom_growth_pct | Decimal Number |
| orders, fully_cancelled_orders, customers | Whole Number |
| is_partial_month | True/False |

This table is only a reference copy of the notebook numbers. The visuals use the measures.

**cohort_retention**

| Column | Type |
|---|---|
| cohort | Date |
| month_index, cohort_size | Whole Number |
| retention_pct | Decimal Number |

**rfm_customers**

| Column | Type |
|---|---|
| customer_id | Whole Number |
| recency_days, frequency, R_score, F_score, M_score | Whole Number |
| net_revenue | Decimal Number |
| RFM_score, segment | **Text** (keeps "545" as a code, not a number) |

**segment_summary**

| Column | Type |
|---|---|
| segment | Text |
| customers | Whole Number |
| customers_pct, revenue, revenue_pct, avg_recency_days, avg_frequency, avg_revenue_per_customer | Decimal Number |

The file is already in segment order, from Champions to Lost. Add **Add Column > Index Column > From 1** and rename it `segment_order`. Then in the model, select `segment` and use **Column tools > Sort by column > segment_order**.

Click **Close & Apply**.

---

## 2. Date table and relationships

**Modeling > New table**, then paste:

```dax
Date =
ADDCOLUMNS (
    CALENDAR ( DATE ( 2009, 12, 1 ), DATE ( 2011, 12, 31 ) ),
    "Year", YEAR ( [Date] ),
    "Month Number", MONTH ( [Date] ),
    "Month", FORMAT ( [Date], "MMM" ),
    "Year Month", FORMAT ( [Date], "YYYY-MM" ),
    "Month Start", DATE ( YEAR ( [Date] ), MONTH ( [Date] ), 1 ),
    "Is Partial Month", [Date] >= DATE ( 2011, 12, 1 )   // data ends 9 Dec 2011
)
```

Then:
1. Select the table and choose **Table tools > Mark as date table**, using the `Date` column.
2. Sort `Month` by `Month Number`, using **Column tools > Sort by column**.
3. Set the format of `Year Month` and `Month Start` to `yyyy-mm`.

**Relationships** (Model view, drag the column onto its counterpart):

| From (many) | To (one) | Cardinality | Filter direction |
|---|---|---|---|
| fact_sales[order_date] | Date[Date] | Many to one | Single |
| fact_sales[customer_id] | rfm_customers[customer_id] | Many to one | Single |

`cohort_retention`, `segment_summary` and `monthly_kpis` stay **unrelated**. They are pre-aggregated tables.

37 customers with net revenue <= 0 are not in `rfm_customers`. Their sales, and sales without a Customer ID, show as segment "(Blank)" when you slice by segment.

**Measures:** select `fact_sales`, choose **Home > New measure**, and paste each measure from `measures.dax`, one at a time.
Format them as follows:
- **Net Revenue, Gross Revenue, Cancellations, Avg Order Value, Revenue Same Month LY:** Currency £, 0 decimals.
- **Cancellation Rate %, Revenue MoM %, YoY %:** Percentage, 1 decimal.
- **Orders, Customers:** Whole number with a thousands separator.

**Theme:** choose **View > Themes > Browse for themes** and select `dashboard/theme.json`.

---

## 3. Page 1 "Overview"

| Visual | Setup |
|---|---|
| **Slicers** (top row) | `Date[Year]` (dropdown) and `fact_sales[country]` (dropdown, search on) |
| **KPI cards** (one row) | Net Revenue, Gross Revenue, Cancellation Rate %, Orders, Customers, Avg Order Value |
| **Line chart**: Net revenue per month | X-axis `Date[Month Start]` (continuous), Y `Net Revenue`, tooltips `Revenue MoM %` and `YoY %`. Add a visual-level filter `Date[Is Partial Month]` = False to hide December 2011. |
| **Bar chart**: Top 10 countries excl. UK | Y-axis `fact_sales[country]`, X `Net Revenue`. Visual filters: `country` is not "United Kingdom", and `country` Top N = 10 by `Net Revenue`. Sort descending and turn data labels on. |
| **Bar chart**: Top 10 products | Y-axis `fact_sales[description]`, X `Net Revenue`. Visual filter: `description` Top N = 10 by `Net Revenue`. Data labels on. |

The cards keep December 2011, because all-time totals include it. Only the line chart hides it.
Add a text box under the country chart, for example: "UK not shown: 85% of net revenue".

---

## 4. Page 2 "Customers"

| Visual | Setup |
|---|---|
| **Matrix**: Cohort retention | Rows `cohort_retention[cohort]` (format `yyyy-mm`, use the column itself rather than the date hierarchy), columns `month_index`, values `retention_pct` set to **Sum**. Filter `month_index` >= 1. Use **Format > Cell elements > Background color > fx > Gradient** from `#cde2fb` (lowest) to `#0d366b` (highest), and set the font color to white for values >= 25 with rules. Turn off row and column subtotals. |
| **Clustered bar**: RFM segments | Y-axis `segment_summary[segment]`, X `customers_pct` and `revenue_pct`. Legend shows "share of customers" vs "share of revenue". Sort by `segment_order`. |
| **Bar chart**: LTV by segment | Y-axis `segment_summary[segment]`, X `avg_revenue_per_customer` (Currency £). Title: "Historical LTV by RFM segment (GBP)". |
| **Card** (optional) | Count of `rfm_customers[customer_id]`, labelled "Customers scored". |

Optional page filter: `rfm_customers[segment]` as a slicer. It filters fact_sales through the relationship, so the cards from page 1 can be copied here and will show values per segment.

---

## 5. Export to PDF and screenshots for the README

**PDF:** choose **File > Export > Export to PDF**. Save it as `dashboard/online_retail_dashboard.pdf`.

**Screenshots:**
1. Open each page with **View > Fit to page** and clear all slicers.
2. Press **Win + Shift + S**, select the report canvas, and paste into Paint or Snipping Tool.
3. Save the images as:
   - `reports/figures/dashboard_overview.png`
   - `reports/figures/dashboard_customers.png`
4. Add them to the README under **Dashboard**:

```markdown
![Overview](reports/figures/dashboard_overview.png)
![Customers](reports/figures/dashboard_customers.png)
```

Save the report as `dashboard/online_retail.pbix`.

**Check the numbers against the notebook.** With no slicers applied, the cards should show:

| Card | Value |
|---|---|
| Net Revenue | £18,927,399 |
| Gross Revenue | £19,643,862 |
| Cancellation Rate % | 3.6% |
| Orders | 39,246 |
| Customers | 5,852 |
| Avg Order Value | £489 |
