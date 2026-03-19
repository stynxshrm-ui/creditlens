"""
ingestion/derive_payments.py

Derives a monthly payment panel from Lending Club's cumulative payment fields.

Lending Club's public dataset does not provide month-by-month payment history.
We approximate monthly behaviour by:
  1. Building a loan-month grid (one row per loan per month of term)
  2. Estimating active payment months from issue_date to last_pymnt_d
  3. Distributing total_pymnt uniformly across active months
  4. Computing payment_ratio = estimated_payment / installment

This is a documented approximation. True monthly payment history
would require Lending Club's proprietary PMTHIST dataset.

Usage:
    python ingestion/derive_payments.py
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta

DB_PATH    = Path("data/creditlens.duckdb")
OUT_PATH   = Path("data/tables")


def load_source(con) -> pd.DataFrame:
    """Load the fields we need for derivation from DuckDB."""
    return con.execute("""
        SELECT
            l.loan_id,
            l.issue_date,
            l.term_months,
            l.installment,
            p.total_pymnt,
            p.last_pymnt_d,
            p.last_pymnt_amnt,
            o.default_flag
        FROM loans    l
        JOIN payments p USING (loan_id)
        JOIN outcomes o USING (loan_id)
        WHERE l.term_months IS NOT NULL
          AND l.installment  > 0
          AND l.issue_date   IS NOT NULL
    """).fetchdf()


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Lending Club date strings — format is 'Jan-2015'."""
    df["issue_date"]    = pd.to_datetime(df["issue_date"],    format="%b-%Y")
    df["last_pymnt_d"]  = pd.to_datetime(df["last_pymnt_d"],  format="%b-%Y",
                                          errors="coerce")
    return df


def derive_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per loan per month of term.
    Estimate payment behaviour from cumulative fields.
    """
    records = []

    for _, loan in df.iterrows():
        term      = int(loan["term_months"])
        issue     = loan["issue_date"]
        last_pymnt = loan["last_pymnt_d"]
        installment = loan["installment"]
        total_pymnt = loan["total_pymnt"] if pd.notna(loan["total_pymnt"]) else 0

        # How many months did the borrower actually pay?
        if pd.isna(last_pymnt) or last_pymnt < issue:
            active_months = 0
        else:
            delta = relativedelta(last_pymnt, issue)
            active_months = min(delta.years * 12 + delta.months + 1, term)

        # Uniform payment across active months
        per_month_payment = total_pymnt / active_months if active_months > 0 else 0

        for month_num in range(1, term + 1):
            month_date      = issue + relativedelta(months=month_num - 1)
            is_active       = month_num <= active_months
            est_payment     = per_month_payment if is_active else 0.0
            payment_ratio   = (est_payment / installment) if installment > 0 else 0.0

            records.append({
                "loan_id":            loan["loan_id"],
                "month_number":       month_num,
                "month_date":         month_date,
                "expected_payment":   round(installment, 2),
                "estimated_payment":  round(est_payment, 2),
                "payment_ratio":      round(min(payment_ratio, 2.0), 4),
                "is_active":          int(is_active),
                "default_flag":       int(loan["default_flag"]),
            })

    return pd.DataFrame(records)


def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH))

    print("Loading source data...")
    df = load_source(con)
    df = parse_dates(df)
    print(f"  {len(df):,} loans to expand")

    print("Deriving monthly panel (this takes a minute)...")
    panel = derive_monthly_panel(df)

    print(f"  Panel shape: {panel.shape}")
    print(f"  Months per loan (avg): {len(panel) / len(df):.1f}")
    print(f"  Active month ratio: {panel['is_active'].mean():.1%}")

    # Write to Parquet
    out_path = OUT_PATH / "monthly_payments.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Register in DuckDB
    con.execute(f"""
        CREATE OR REPLACE TABLE monthly_payments AS
        SELECT * FROM read_parquet('{out_path}')
    """)

    # Spot check — payment behaviour by default status
    print("\nPayment ratio by default status (first 6 months):")
    print(con.execute("""
        SELECT
            default_flag,
            ROUND(AVG(payment_ratio), 3) AS avg_payment_ratio,
            ROUND(AVG(is_active),     3) AS avg_active_rate,
            COUNT(DISTINCT loan_id)       AS loan_count
        FROM monthly_payments
        WHERE month_number <= 6
        GROUP BY default_flag
        ORDER BY default_flag
    """).fetchdf().to_string(index=False))

    print("\nDone. monthly_payments table ready.")
    con.close()


if __name__ == "__main__":
    main()