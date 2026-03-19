-- features/origination_features.sql
--
-- Static loan and borrower characteristics known at origination.
-- These are the features a credit model would have at application time,
-- before any payment behaviour is observed.
--
-- Observation point: loan issue date (month 0)

CREATE OR REPLACE VIEW origination_features AS
SELECT
    l.loan_id,
    l.issue_date,
    l.grade,
    l.grade_numeric,
    l.term_months,
    l.int_rate,
    l.loan_amnt,
    l.installment,
    l.purpose,

    -- Borrower financials
    b.annual_inc,
    b.dti,
    b.revol_util,
    b.revol_bal,

    -- Bureau history
    b.delinq_2yrs,
    b.open_acc,
    b.pub_rec,
    b.total_acc,
    b.mort_acc,
    b.pub_rec_bankruptcies,

    -- Derived origination features
    CASE
        WHEN b.annual_inc > 0
        THEN ROUND(l.loan_amnt / b.annual_inc, 4)
        ELSE NULL
    END AS loan_to_income,

    CASE
        WHEN b.annual_inc > 0
        THEN ROUND(l.installment * 12 / b.annual_inc, 4)
        ELSE NULL
    END AS payment_to_income,

    -- Risk flags
    CASE WHEN b.dti          > 28   THEN 1 ELSE 0 END AS high_dti_flag,
    CASE WHEN b.revol_util   > 75   THEN 1 ELSE 0 END AS high_util_flag,
    CASE WHEN b.delinq_2yrs  > 0    THEN 1 ELSE 0 END AS prior_delinq_flag,
    CASE WHEN b.pub_rec      > 0    THEN 1 ELSE 0 END AS public_record_flag,

    -- Combined stress flag — both high DTI and high utilisation
    CASE
        WHEN b.dti > 28 AND b.revol_util > 75
        THEN 1 ELSE 0
    END AS dual_stress_flag,

    -- Employment stability encoding
    b.emp_length_years,
    CASE
        WHEN b.emp_length_years >= 10 THEN 'senior'
        WHEN b.emp_length_years >= 3  THEN 'established'
        WHEN b.emp_length_years >= 1  THEN 'early'
        ELSE 'new'
    END AS emp_stability,

    -- Home ownership encoding
    b.home_ownership,
    CASE b.home_ownership
        WHEN 'MORTGAGE' THEN 3
        WHEN 'OWN'      THEN 2
        WHEN 'RENT'     THEN 1
        ELSE 0
    END AS home_ownership_enc,

    -- Target
    o.default_flag

FROM loans     l
JOIN borrowers b USING (loan_id)
JOIN outcomes  o USING (loan_id)
WHERE l.loan_amnt  > 0
  AND b.annual_inc > 0
  AND b.dti        IS NOT NULL