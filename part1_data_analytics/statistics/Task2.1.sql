-- =========================================================
-- Task 2.1: Descriptive Statistics of Traffic Volume
-- =========================================================
WITH hourly_traffic AS (
    SELECT
        date_time,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),

ranked AS (
    SELECT
        traffic_volume,
        ROW_NUMBER() OVER (ORDER BY traffic_volume) AS rn,
        COUNT(*) OVER () AS n
    FROM hourly_traffic
),

median_value AS (
    SELECT
        AVG(traffic_volume) AS median
    FROM ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
),

stats AS (
    SELECT
        AVG(traffic_volume) AS mean,
        AVG(1.0 * traffic_volume * traffic_volume)
            - AVG(traffic_volume) * AVG(traffic_volume) AS variance,
        MAX(traffic_volume) - MIN(traffic_volume) AS range
    FROM hourly_traffic
)

SELECT
    ROUND(mean, 2) AS mean,
    ROUND(median, 2) AS median,
    ROUND(variance, 2) AS variance,
    ROUND(SQRT(variance), 2) AS standard_deviation,
    range
FROM stats
CROSS JOIN median_value;

/*
Interpretation:
The mean and median are quite close, so the data do not appear to show
marked skew based on these values alone.
However, the standard deviation is quite large, indicating substantial
variation in traffic volume per observed hour.
The wide range also confirms that traffic volume varies considerably
across different time periods.
*/