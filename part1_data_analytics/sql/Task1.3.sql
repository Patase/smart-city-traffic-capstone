-- =========================================================
-- Task 1.3: Temperature Around Holidays
-- =========================================================

WITH holiday_dates AS (
    SELECT DISTINCT
        DATE(date_time) AS holiday_date,
        holiday
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND strftime('%Y', date_time) IN ('2015', '2016', '2017')
),
hourly_data AS (
    SELECT
        date_time,
        DATE(date_time) AS date_only,
        MAX(temp) AS temp,
        MAX(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    strftime('%Y', h.holiday_date) AS year,
    h.holiday,
    COUNT(*) AS observed_hours,
    ROUND(AVG(d.temp) - 273.15, 2) AS avg_temp_C,
    ROUND(AVG(d.traffic_volume), 2) AS avg_hourly_traffic
FROM holiday_dates h
JOIN hourly_data d
    ON d.date_only = h.holiday_date
GROUP BY h.holiday_date, h.holiday
ORDER BY h.holiday, h.holiday_date;

/*
Interpretation:
- Labor Day average temperature decreased from 2015 to 2017. Traffic volume did not follow the same trend consistently. (Decrease then increase respectively)
- Only 2016 and 2017 were  comparable because no New Year’s Day data were available for 2015. From 2016 to 2017, average temperature increased from, while average hourly traffic increased from
 vehicles/hour. The limited number of comparable years makes it difficult to draw a conclusion about the relationship between temperature and traffic volume.
*/