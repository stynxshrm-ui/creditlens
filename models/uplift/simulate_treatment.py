"""
models/uplift/simulate_treatment.py

Simulates a historical restructuring intervention experiment.

In production, this flag would come from actual campaign records.
Here we simulate it: among borrowers showing early payment stress
(avg payment ratio < 0.9 in months 1-3), 50% were randomly offered
a restructuring plan. We then observe whether the intervention
reduced eventual default probability.

This is a realistic credit operations scenario — banks routinely
run early intervention programmes for at-risk borrowers.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path("data/creditlens.duckdb")


def simulate_treatment(con) -> pd.DataFrame:
    # Identify early-struggling borrowers
    struggling = con.execute("""
        SELECT
            loan_id,
            AVG(payment_ratio) AS avg_ratio_m1_3
        FROM monthly_payments
        WHERE month_number <= 3
        GROUP BY loan_id
        HAVING AVG(payment_ratio) < 0.9
    """).fetchdf()

    print(f"Early-struggling borrowers: {len(struggling):,}")
    print(f"  ({len(struggling)/49342*100:.1f}% of portfolio)")

    # Random 50/50 treatment assignment
    np.random.seed(42)
    struggling["treatment"] = np.random.binomial(1, 0.5, len(struggling))

    print(f"  Treatment group: {struggling['treatment'].sum():,}")
    print(f"  Control group:   {(struggling['treatment']==0).sum():,}")

    return struggling


def main():
    con = duckdb.connect(str(DB_PATH))

    struggling = simulate_treatment(con)

    # Register in DuckDB
    con.execute("DROP TABLE IF EXISTS treatment_flags")
    con.execute("""
        CREATE TABLE treatment_flags AS
        SELECT * FROM struggling
    """)

    # Verify
    result = con.execute("""
        SELECT
            t.treatment,
            COUNT(*)                            AS loans,
            ROUND(AVG(o.default_flag)*100, 2)  AS default_rate_pct
        FROM treatment_flags t
        JOIN outcomes o USING (loan_id)
        GROUP BY t.treatment
        ORDER BY t.treatment
    """).fetchdf()

    print("\nDefault rate by treatment group:")
    print(result.to_string(index=False))
    print("\nNote: raw difference may reflect randomisation noise.")
    print("T-Learner captures heterogeneous treatment effects.")

    con.close()


if __name__ == "__main__":
    main()