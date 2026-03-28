-- Daily Active Users (DAU)
SELECT
    date_trunc('day', session_time)::date AS activity_date,
    COUNT(DISTINCT user_id) AS dau
FROM sessions
WHERE session_time IS NOT NULL
GROUP BY activity_date
ORDER BY activity_date;

-- Monthly Active Users (MAU)
SELECT
    date_trunc('month', session_time)::date AS activity_month,
    COUNT(DISTINCT user_id) AS mau
FROM sessions
GROUP BY activity_month
ORDER BY activity_month;

-- Retention rate (weekly) by cohort
WITH cohort AS (
    SELECT
        user_id,
        date_trunc('week', signup_date)::date AS cohort_week,
        MIN(date_trunc('week', session_time))::date AS first_active_week
    FROM users
    LEFT JOIN sessions USING (user_id)
    WHERE session_time IS NOT NULL
    GROUP BY user_id, cohort_week
)
SELECT
    cohort_week,
    COUNT(DISTINCT user_id) AS new_users,
    COUNT(DISTINCT CASE WHEN first_active_week <= cohort_week + INTERVAL '4 weeks' THEN user_id END) AS retained_4w
FROM cohort
GROUP BY cohort_week
ORDER BY cohort_week;

-- Funnel: signup -> active -> paid -> churn
WITH active_users AS (
    SELECT DISTINCT user_id FROM sessions WHERE session_time >= now() - INTERVAL '30 days'
),
paid_users AS (
    SELECT DISTINCT user_id FROM payments WHERE plan <> 'free' AND revenue > 0
),
churned_users AS (
    SELECT user_id FROM user_metrics WHERE churn_flag
)
SELECT
    (SELECT COUNT(*) FROM users) AS signups,
    (SELECT COUNT(*) FROM active_users) AS active_last_30d,
    (SELECT COUNT(*) FROM paid_users) AS paid,
    (SELECT COUNT(*) FROM churned_users) AS churned;
