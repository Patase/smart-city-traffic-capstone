-- =========================================================
-- Task 3.1: Basic Probability
-- Congestion = traffic volume > 5,500 vehicles
-- =========================================================

SELECT
    ROUND(
        AVG(CASE WHEN traffic_volume > 5500 THEN 1.0 ELSE 0 END),
        4
    ) AS P_Congestion,

    ROUND(
        AVG(CASE WHEN weather_main = 'Clear' THEN 1.0 ELSE 0 END),
        4
    ) AS P_Clear_Weather,

    ROUND(
        AVG(
            CASE
                WHEN traffic_volume > 5500
                AND weather_main = 'Clear'
                THEN 1.0 ELSE 0
            END
        ),
        4
    ) AS P_Congestion_AND_Clear

FROM traffic;