-- =========================================================
-- Task 3.2: Conditional Probability
-- =========================================================

SELECT
    ROUND(
        1.0 * SUM(
            CASE
                WHEN traffic_volume > 5500
                AND weather_main = 'Clear'
                THEN 1 ELSE 0
            END
        )
        /
        SUM(
            CASE
                WHEN traffic_volume > 5500
                THEN 1 ELSE 0
            END
        ),
        4
    ) AS P_Clear_given_Congestion,

    ROUND(
        1.0 * SUM(
            CASE
                WHEN traffic_volume > 5500
                AND temp > 292
                THEN 1 ELSE 0
            END
        )
        /
        SUM(
            CASE
                WHEN traffic_volume > 5500
                THEN 1 ELSE 0
            END
        ),
        4
    ) AS P_HighTemp_given_Congestion

FROM traffic;

/*
P(Congestion AND Clear) = 0.0366, and P(Congestion) × P(Clear) =  0.0409 so congestion and clear weather is not independent
*/