
-- =========================================================
-- Task 1.2: Annual Traffic Trends
-- =========================================================
--total traffic
WITH hourly_traffic AS (
    SELECT
        date_time,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    strftime('%Y', date_time) AS year,
    SUM(traffic_volume) AS total_traffic_volume
FROM hourly_traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

--changes between years
WITH yearly_traffic AS (
    SELECT
        strftime('%Y', date_time) AS year,
        SUM(traffic_volume) AS total_traffic
    FROM (
        SELECT
            date_time,
            MAX(traffic_volume) AS traffic_volume
        FROM traffic
        GROUP BY date_time
    )
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT
    year,
    total_traffic,
    total_traffic - LAG(total_traffic) OVER (ORDER BY year) AS change_from_prev
FROM yearly_traffic
ORDER BY year;


--traffic record spanning time
SELECT
    strftime('%Y', date_time) AS year,
    MIN(date_time) AS first_record,
    MAX(date_time) AS last_record
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

--traffic per hours
WITH hourly_traffic AS (
    SELECT
        date_time,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS observed_hours,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_per_observed_hour
FROM hourly_traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

/*
Results:
2012 = 6,785,754
2013 = 24,139,878
2014 = 14,718,915
2015 = 11,706,145
2016 = 25,032,183
2017 = 29,420,221

Interpretation:
- Traffic increased from 2012 to 2013 by 17354124
- Traffic decreased from 2013 to 2014 by -9420963
- Traffic decreased from 2014 to 2015 by -3012770
- Traffic increased from 2015 to 2016 by 13326038
- Traffic increased from 2016 to 2017 by 4388038

Meaningful observations:
1. The low totals in 2012, 2014 and 2015 are partly explained by incomplete calendar-year coverage.
2. The number of unique observed hours becomes progressively more complete in the later full-calendar-span years. (I've compared from 2013, 2015, and 2017 which is full-calendar year)
3. Average traffic per observed hour showed stable trend overall with 2017 showing the highest average hourly traffic.
*/