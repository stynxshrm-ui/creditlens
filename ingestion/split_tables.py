"""
ingestion/split_tables.py

Splits the raw Lending Club accepted loans CSV into four logical
Parquet tables that reflect how this data would exist in a production
warehouse. This normalisation step enables proper temporal feature
engineering and multi-table SQL joins in DuckDB.

Tables produced:
    loans       — static loan characteristics at origination
    borrowers   — borrower bureau snapshot at origination
    payments    — derived monthly payment behaviour
    outcomes    — default label and recovery information

Usage:
    python ingestion/split_tables.py --sample 50000
    python ingestion/split_tables.py --full
"""

import argparse
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
OUT_PATH = Path("data/tables")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column assignments — which raw columns belong to which table
# ---------------------------------------------------------------------------

LOAN_COLS = [
    "id", "issue_d", "loan_amnt", "funded_amnt", "term",
    "int_rate", "installment", "grade", "sub_grade",
    "purpose", "title", "initial_list_status", "application_type",
]

BORROWER_COLS = [
    "id", "emp_title", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "dti",
    "delinq_2yrs", "earliest_cr_line", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies",
]

OUTCOME_COLS = [
    "id", "loan_status", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
]

# Payment behaviour columns — used to derive monthly payment rows
PAYMENT_SOURCE_COLS = [
    "id", "issue_d", "term", "installment",
    "total_pymnt", "total_rec_prncp", "total_rec_int",
    "last_pymnt_amnt", "last_pymnt_d",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw(sample: int = None) -> pd.DataFrame:
    print(f"Loading raw CSV {'(full)' if sample is None else f'(sample={sample:,})'}")

    # Read in chunks to handle 1.6GB file gracefully
    if sample:
        df = pd.read_csv(
            RAW_PATH,
            nrows=sample,
            low_memory=False,
        )
    else:
        chunks = []
        for chunk in pd.read_csv(RAW_PATH, chunksize=100_000, low_memory=False):
            chunks.append(chunk)
            print(f"  loaded {sum(len(c) for c in chunks):,} rows...", end="\r")
        df = pd.concat(chunks, ignore_index=True)

    print(f"\nRaw shape: {df.shape}")
    return df


def clean_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing loan ID — these are footer/summary rows Lending Club appends."""
    before = len(df)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(str).str.strip()
    print(f"Dropped {before - len(df):,} rows with null id")
    return df


def add_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary default label.
    Charged Off and Default statuses are the positive class.
    Current, Fully Paid, In Grace Period are negative.
    Late statuses are excluded — outcome not yet determined.
    """
    positive = {"Charged Off", "Default"}
    negative = {"Fully Paid", "Current", "In Grace Period"}

    df["default_flag"] = df["loan_status"].map(
        lambda s: 1 if s in positive else (0 if s in negative else None)
    )
    before = len(df)
    df = df.dropna(subset=["default_flag"]).copy()
    df["default_flag"] = df["default_flag"].astype(int)
    print(f"Dropped {before - len(df):,} rows with ambiguous loan_status (Late, etc.)")
    print(f"Default rate: {df['default_flag'].mean():.1%}")
    return df


def write_parquet(df: pd.DataFrame, name: str) -> None:
    path = OUT_PATH / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote {name}.parquet — {len(df):,} rows, {len(df.columns)} cols ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_loans(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LOAN_COLS if c in df.columns]
    loans = df[cols].copy()
    loans = loans.rename(columns={"id": "loan_id", "issue_d": "issue_date"})

    # Clean term — "36 months" -> 36
    loans["term_months"] = loans["term"].str.extract(r"(\d+)").astype(float)

    # Clean int_rate — "13.56%" -> 13.56
    loans["int_rate"] = loans["int_rate"].astype(str).str.replace("%", "").astype(float)

    # Grade to numeric
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    loans["grade_numeric"] = loans["grade"].map(grade_map)

    return loans


def build_borrowers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in BORROWER_COLS if c in df.columns]
    borrowers = df[cols].copy()
    borrowers = borrowers.rename(columns={"id": "loan_id"})

    # Clean emp_length — "10+ years" -> 10, "< 1 year" -> 0
    borrowers["emp_length_years"] = (
        borrowers["emp_length"]
        .str.extract(r"(\d+)")
        .astype(float)
    )

    # Clean revol_util — "54.3%" -> 54.3
    borrowers["revol_util"] = (
        borrowers["revol_util"]
        .astype(str)
        .str.replace("%", "")
        .apply(pd.to_numeric, errors="coerce")
    )

    return borrowers


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OUTCOME_COLS if c in df.columns]
    outcomes = df[cols].copy()
    outcomes = outcomes.rename(columns={"id": "loan_id"})
    outcomes["default_flag"] = df["default_flag"].values
    return outcomes


def build_payments_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a simplified monthly payment table from Lending Club's
    cumulative payment fields.

    Lending Club does not provide month-by-month payment history —
    only cumulative totals. We derive a single payment behaviour
    summary row per loan as a proxy. Full monthly derivation requires
    the PMTHIST dataset (not available in this public release).

    This table captures: payment ratio (paid vs expected),
    payment completion, and implied delinquency signal.

    Note: documented in README as a known simplification.
    Full temporal features are built in features/payment_features.sql
    using available proxy signals.
    """
    cols = [c for c in PAYMENT_SOURCE_COLS if c in df.columns]
    pay = df[cols].copy()
    pay = pay.rename(columns={"id": "loan_id", "issue_d": "issue_date"})

    # Total expected payment over loan life
    pay["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
    pay["total_expected"] = pay["installment"] * pay["term_months"]

    # Payment ratio — how much of expected was actually paid
    pay["payment_ratio"] = (
        pay["total_pymnt"] / pay["total_expected"]
    ).clip(0, 1.5)  # cap at 150% to handle overpayments

    # Last payment amount relative to installment
    pay["last_pymnt_ratio"] = (
        pay["last_pymnt_amnt"] / pay["installment"]
    ).clip(0, 2)

    return pay


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample: int = None):
    df = load_raw(sample)
    df = clean_id(df)
    df = add_default_flag(df)

    print(f"\nBuilding tables from {len(df):,} loans...")

    loans     = build_loans(df)
    borrowers = build_borrowers(df)
    outcomes  = build_outcomes(df)
    payments  = build_payments_simple(df)

    print("\nWriting Parquet files...")
    write_parquet(loans,     "loans")
    write_parquet(borrowers, "borrowers")
    write_parquet(outcomes,  "outcomes")
    write_parquet(payments,  "payments")

    print("\nDone. Run ingestion/load_duckdb.py to register tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", type=int, default=50_000,
                       help="Number of rows to load (default: 50,000 for dev)")
    group.add_argument("--full", action="store_true",
                       help="Load full dataset (slow)")
    args = parser.parse_args()

    sample = None if args.full else args.sample
    main(sample)
