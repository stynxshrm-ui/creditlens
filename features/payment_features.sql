-- features/payment_features.sql
--
-- Temporal payment behaviour features derived from monthly_payments.
-- Observation point: end of month 6. Only months 1-6 are used.
-- Using data beyond month 6 would constitute data leakage.
--
-- These features consistently outperform origination features
-- for default prediction because they reflect actual borrower
-- behaviour rather than stated financials at application time.

CREATE OR REPLACE VIEW payment_features AS
WITH

-- Restrict strictly to observation window
obs_window AS (
    SELECT *
    FROM monthly_payments
    WHERE month_number <= 6
),

-- Core payment behaviour aggregates
payment_aggs AS (
    SELECT
        loan_id,

        -- Average payment ratio across first 6 months
        -- Key predictor: defaulters pay closer to minimum
        ROUND(AVG(payment_ratio), 4)                AS avg_payment_ratio_m6,

        -- Minimum payment ratio — captures worst month
        ROUND(MIN(payment_ratio), 4)                AS min_payment_ratio_m6,

        -- Payment consistency — low std = reliable payer
        ROUND(STDDEV(payment_ratio), 4)             AS payment_volatility_m6,

        -- How many of 6 months were active
        SUM(is_active)                              AS active_months_m6,

        -- Early drop-off flag — stopped paying before month 6
        MAX(CASE WHEN month_number = 6
            THEN is_active ELSE 0 END)              AS still_active_m6,

        -- Payment trend: later months vs earlier months
        -- Negative = payment behaviour deteriorating
        ROUND(
            AVG(CASE WHEN month_number >= 4
                THEN payment_ratio END)
            -
            AVG(CASE WHEN month_number <= 3
                THEN payment_ratio END)
        , 4)                                        AS payment_trend_m6,

        -- First month payment ratio — immediate signal
        ROUND(AVG(CASE WHEN month_number = 1
            THEN payment_ratio END), 4)             AS first_month_ratio,

        -- Last observed payment ratio
        ROUND(AVG(CASE WHEN month_number = 6
            THEN payment_ratio END), 4)             AS last_month_ratio

    FROM obs_window
    GROUP BY loan_id
),

-- Month-over-month volatility using LAG
-- Pre-compute month-over-month change using LAG first
lagged AS (
    SELECT
        loan_id,
        month_number,
        payment_ratio,
        LAG(payment_ratio) OVER (
            PARTITION BY loan_id
            ORDER BY month_number
        ) AS prev_ratio
    FROM obs_window
),

-- Now aggregate the pre-computed changes
payment_changes AS (
    SELECT
        loan_id,
        ROUND(AVG(ABS(payment_ratio - prev_ratio)), 4) AS avg_monthly_change
    FROM lagged
    WHERE month_number > 1
      AND prev_ratio IS NOT NULL
    GROUP BY loan_id
)

SELECT
    p.*,
    COALESCE(c.avg_monthly_change, 0)   AS avg_monthly_change,

    -- Derived risk flags from payment behaviour
    CASE WHEN p.avg_payment_ratio_m6 < 0.95  THEN 1 ELSE 0 END
        AS underpayment_flag,
    CASE WHEN p.min_payment_ratio_m6 = 0     THEN 1 ELSE 0 END
        AS missed_payment_flag,
    CASE WHEN p.payment_trend_m6 < -0.1      THEN 1 ELSE 0 END
        AS deteriorating_flag,
    CASE WHEN p.still_active_m6 = 0          THEN 1 ELSE 0 END
        AS early_dropout_flag

FROM payment_aggs p
LEFT JOIN payment_changes c USING (loan_id)