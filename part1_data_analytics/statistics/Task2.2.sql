-- =========================================================
-- Task 2.2: Correlation using all rows directly
-- =========================================================
SELECT ROUND(
    (
        COUNT(*) * SUM(temp * traffic_volume)
        - SUM(temp) * SUM(traffic_volume)
    )
    /
    SQRT(
        (COUNT(*) * SUM(temp * temp) - SUM(temp) * SUM(temp))
        *
        (COUNT(*) * SUM(traffic_volume * traffic_volume)
        - SUM(traffic_volume) * SUM(traffic_volume))
    ),
    4
) AS correlation
FROM traffic;

/*
Interpretation:
The correlation coefficient between temperature and traffic volume is 0.1303, indicating a weak positive relationship. 
This means that higher temperatures tend to be associated with slightly higher traffic volume, but the relationship is weak. (Max of 1)

Correlation shows that two variables are related, but it does not prove that one variable causes the other. 
The relationship may be affected by other factors (in this case, for example, rain, holidays and such )or may occur indirectly or by chance.
*/