-- features/vintage_features.sql
--
-- Cohort-level features based on loan origination quarter.
-- Captures macro-economic environment at time of issuance.
-- Loans from the same vintage share similar economic conditions.
CREATE OR REPLACE VIEW vintage_features AS
SELECT
    loan_id,
    issue_date,

    DATE_TRUNC('quarter', STRPTIME(issue_date, '%b-%Y'))
        AS issue_quarter,

    YEAR(STRPTIME(issue_date, '%b-%Y'))
        AS issue_year,

    ROUND(AVG(o.default_flag) OVER (
        PARTITION BY DATE_TRUNC('quarter', STRPTIME(issue_date, '%b-%Y'))
    ), 4) AS vintage_default_rate,

    COUNT(*) OVER (
        PARTITION BY DATE_TRUNC('quarter', STRPTIME(issue_date, '%b-%Y'))
    ) AS vintage_size

FROM loans l
JOIN outcomes o USING (loan_id)