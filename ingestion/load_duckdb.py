"""
ingestion/load_duckdb.py

Registers the four Parquet tables in a local DuckDB database.
Run this after split_tables.py.

Usage:
    python ingestion/load_duckdb.py
"""

import duckdb
from pathlib import Path

DB_PATH    = Path("data/creditlens.duckdb")
TABLE_PATH = Path("data/tables")

TABLES = ["loans", "borrowers", "outcomes"]
# monthly_payments registered separately by ingestion/derive_payments.py

def main():
    print(f"Connecting to {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    for table in TABLES:
        parquet_path = TABLE_PATH / f"{table}.parquet"

        if not parquet_path.exists():
            print(f"  MISSING: {parquet_path} — run split_tables.py first")
            continue

        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols  = con.execute(f"DESCRIBE {table}").fetchdf().shape[0]
        print(f"  registered {table}: {count:,} rows, {cols} cols")

    # Verify joins work cleanly
    print("\nVerifying joins...")
    result = con.execute("""
        SELECT
            COUNT(*)                                    AS total_loans,
            SUM(o.default_flag)                         AS total_defaults,
            ROUND(AVG(o.default_flag) * 100, 2)         AS default_rate_pct,
            ROUND(AVG(b.dti), 2)                        AS avg_dti,
            ROUND(AVG(l.loan_amnt), 0)                  AS avg_loan_amount
        FROM loans l
        JOIN borrowers b USING (loan_id)
        JOIN outcomes  o USING (loan_id)
        JOIN payments  p USING (loan_id)
    """).fetchdf()

    print(result.to_string(index=False))
    print("\nDuckDB ready. Database written to data/creditlens.duckdb")
    con.close()


if __name__ == "__main__":
    main()