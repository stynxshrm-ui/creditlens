"""
features/build_features.py

Executes all feature SQL views and registers them in DuckDB.
Final output: model_features view — one row per loan, all features joined.

Usage:
    python features/build_features.py
"""

import duckdb
from pathlib import Path

DB_PATH      = Path("data/creditlens.duckdb")
FEATURES_DIR = Path("features")


def execute_sql_file(con, path: Path):
    sql = path.read_text()
    con.execute(sql)
    print(f"  registered: {path.name}")


def build_model_features(con):
    """Join all feature views into a single model-ready table."""
    con.execute("""
        CREATE OR REPLACE VIEW model_features AS
        SELECT
            o.*,

            -- Payment features
            p.avg_payment_ratio_m6,
            p.min_payment_ratio_m6,
            p.payment_volatility_m6,
            p.active_months_m6,
            p.still_active_m6,
            p.payment_trend_m6,
            p.first_month_ratio,
            p.last_month_ratio,
            p.avg_monthly_change,
            p.underpayment_flag,
            p.missed_payment_flag,
            p.deteriorating_flag,
            p.early_dropout_flag,

            -- Vintage features
            v.issue_quarter,
            v.issue_year,
            v.vintage_default_rate,
            v.vintage_size

        FROM origination_features o
        JOIN payment_features     p USING (loan_id)
        JOIN vintage_features     v USING (loan_id)
    """)
    print("  registered: model_features (joined view)")


def main():
    con = duckdb.connect(str(DB_PATH))
    print("Building feature views...")

    execute_sql_file(con, FEATURES_DIR / "origination_features.sql")
    execute_sql_file(con, FEATURES_DIR / "payment_features.sql")
    execute_sql_file(con, FEATURES_DIR / "vintage_features.sql")
    build_model_features(con)

    # Final spot check
    print("\nModel features spot check:")
    print(con.execute("""
        SELECT
            COUNT(*)                                        AS loans,
            ROUND(AVG(default_flag) * 100, 2)              AS default_rate_pct,
            ROUND(AVG(avg_payment_ratio_m6), 3)            AS avg_pay_ratio,
            ROUND(AVG(missed_payment_flag) * 100, 1)       AS pct_missed_payment
        FROM model_features
    """).fetchdf().to_string(index=False))

    print("\nPayment features by default status:")
    print(con.execute("""
        SELECT
            default_flag,
            ROUND(AVG(avg_payment_ratio_m6), 3)    AS avg_pay_ratio,
            ROUND(AVG(payment_volatility_m6), 3)   AS avg_volatility,
            ROUND(AVG(missed_payment_flag), 3)     AS pct_missed,
            ROUND(AVG(deteriorating_flag), 3)      AS pct_deteriorating,
            COUNT(*)                               AS loans
        FROM model_features
        GROUP BY default_flag
        ORDER BY default_flag
    """).fetchdf().to_string(index=False))

    print("\nDefault rate by vintage year:")
    print(con.execute("""
        SELECT
            issue_year,
            COUNT(*)                                AS loans,
            ROUND(AVG(default_flag) * 100, 2)      AS default_rate_pct
        FROM model_features
        GROUP BY issue_year
        ORDER BY issue_year
    """).fetchdf().to_string(index=False))

    con.close()
    print("\nAll feature views ready.")


if __name__ == "__main__":
    main()