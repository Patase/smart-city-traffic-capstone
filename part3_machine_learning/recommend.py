import argparse
import logging
import sys
import pandas as pd

from pathlib import Path


# File paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "part3_ml_data.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RECOMMENDATION_FILE = (
    RESULTS_DIR
    / "travel_recommendations.csv"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)


# Practical travel hours
# ============================================================
DAY_START = 6
DAY_END = 22


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

logger.propagate = False


# Load dataset
# ============================================================
def load_data():

    try:

        df = pd.read_csv(
            DATA_FILE
        )

        logger.info(
            "Part 3 dataset loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load Part 3 dataset: %s",
            e
        )

        sys.exit(1)


# Prepare recommendation dataset
# ============================================================
def prepare_data(df):

    df = df.copy()

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )


    before = len(df)


    # Keep one traffic-volume observation per timestamp
    df = df.drop_duplicates(
        subset=["date_time"]
    )


    removed = (
        before
        - len(df)
    )


    if removed > 0:

        logger.info(
            "%d repeated timestamps excluded from recommendation analysis.",
            removed
        )


    # Create weekday/weekend category
    df["day_type"] = (

        df["is_weekend"]

        .map({
            0: "Weekday",
            1: "Weekend"
        })

    )


    logger.info(
        "Recommendation dataset prepared: %d unique hourly records.",
        len(df)
    )


    return df


# Filter by day type and optional weather
# ============================================================
def filter_conditions(
    df,
    day_type,
    weather=None
):

    filtered_df = df[

        df["day_type"].str.lower()
        == day_type.lower()

    ].copy()


    if len(filtered_df) == 0:

        logger.error(
            "No records found for day type: %s",
            day_type
        )

        return None, None


    applied_weather = None


    # Optional weather condition
    # ========================================================
    if weather is not None:

        weather_df = filtered_df[

            filtered_df["weather_main"]
            .str.lower()
            == weather.lower()

        ].copy()


        # Only use weather-specific analysis when enough data exist
        if len(weather_df) >= 50:

            filtered_df = weather_df

            applied_weather = weather

            logger.info(
                "Weather filter applied: %s (%d records).",
                weather,
                len(filtered_df)
            )


        else:

            logger.warning(
                "Insufficient records for weather condition '%s'. "
                "Recommendation will use all weather conditions.",
                weather
            )


    return filtered_df, applied_weather


# Calculate daytime and overnight traffic
# ============================================================
def calculate_period_traffic(df):

    # Practical daytime / evening period: 06:00-22:00
    daytime_df = df[

        (df["hour"] >= DAY_START)
        & (df["hour"] < DAY_END)

    ].copy()


    # Overnight period: 22:00-06:00
    overnight_df = df[

        (df["hour"] >= DAY_END)
        | (df["hour"] < DAY_START)

    ].copy()


    daytime_average = (
        daytime_df["traffic_volume"]
        .mean()
    )


    overnight_average = (
        overnight_df["traffic_volume"]
        .mean()
    )


    logger.info(
        "Average practical-hours traffic: %.2f vehicles.",
        daytime_average
    )

    logger.info(
        "Average overnight traffic: %.2f vehicles.",
        overnight_average
    )


    return (
        daytime_df,
        daytime_average,
        overnight_average
    )


# Find recommended practical travel windows
# ============================================================
def find_recommendations(
    daytime_df,
    number_of_windows=3
):

    hourly_traffic = (

        daytime_df

        .groupby("hour")

        .agg(

            average_traffic=(
                "traffic_volume",
                "mean"
            ),

            observations=(
                "traffic_volume",
                "size"
            )

        )

        .reset_index()

    )


    # Lowest average traffic first
    hourly_traffic = (

        hourly_traffic

        .sort_values(
            by="average_traffic",
            ascending=True
        )

        .reset_index(drop=True)

    )


    recommendations = (

        hourly_traffic

        .head(
            number_of_windows
        )

        .copy()

    )


    recommendations["end_hour"] = (
        recommendations["hour"]
        + 1
    )


    logger.info(
        "%d practical lower-traffic travel windows identified.",
        len(recommendations)
    )


    return recommendations


# Convert hour to readable time
# ============================================================
def format_hour(hour):

    hour = int(
        hour
    )


    if hour == 0:

        return "12:00 AM"


    elif hour < 12:

        return (
            f"{hour}:00 AM"
        )


    elif hour == 12:

        return "12:00 PM"


    else:

        return (
            f"{hour - 12}:00 PM"
        )


# Display recommendation
# ============================================================
def display_recommendation(
    recommendations,
    day_type,
    daytime_average,
    overnight_average,
    requested_weather=None,
    applied_weather=None
):

    print("\n========================================")
    print("TRAFFIC TRAVEL RECOMMENDATION")
    print("========================================")


    print(
        f"\nDay type: {day_type.title()}"
    )


    if applied_weather is not None:

        print(
            f"Weather condition: {applied_weather}"
        )


    elif requested_weather is not None:

        print(
            f"Requested weather: {requested_weather}"
        )

        print(
            "Insufficient weather-specific records; "
            "general day-type traffic was used."
        )


    print(
        "\nPractical recommendation period: "
        "6:00 AM - 10:00 PM"
    )


    print(
        "\nRecommended lower-traffic travel windows:\n"
    )


    for index, row in recommendations.iterrows():

        start_time = format_hour(
            row["hour"]
        )

        end_time = format_hour(
            row["end_hour"]
        )


        print(

            f"{index + 1}. "
            f"{start_time} - {end_time} "
            f"| Historical average traffic: "
            f"{row['average_traffic']:.0f} vehicles"

        )


    # Best practical recommendation
    # ========================================================
    best = recommendations.iloc[0]


    best_start = format_hour(
        best["hour"]
    )

    best_end = format_hour(
        best["end_hour"]
    )


    print("\n----------------------------------------")
    print("RECOMMENDATION")
    print("----------------------------------------")


    if applied_weather is not None:

        print(

            f"For a {day_type.lower()} journey under "
            f"{applied_weather.lower()} weather conditions, "
            f"consider travelling between "
            f"{best_start} and {best_end}. "
            f"Historical traffic during this period "
            f"averaged approximately "
            f"{best['average_traffic']:.0f} vehicles."

        )


    else:

        print(

            f"For a {day_type.lower()} journey, "
            f"consider travelling between "
            f"{best_start} and {best_end}. "
            f"Historical traffic during this period "
            f"averaged approximately "
            f"{best['average_traffic']:.0f} vehicles."

        )


    # Overnight traffic information
    # ========================================================
    print("\n----------------------------------------")
    print("OVERNIGHT TRAFFIC")
    print("----------------------------------------")


    print(
        f"Average traffic from 6:00 AM to 10:00 PM: "
        f"{daytime_average:.0f} vehicles"
    )


    print(
        f"Average traffic from 10:00 PM to 6:00 AM: "
        f"{overnight_average:.0f} vehicles"
    )


    if overnight_average < daytime_average:

        print(

            "\nHistorical traffic is generally lower "
            "between 10:00 PM and 6:00 AM. "
            "However, the main recommendations above are "
            "restricted to the more practical travel period "
            "between 6:00 AM and 10:00 PM."

        )


    else:

        print(

            "\nOvernight traffic was not lower than the "
            "average traffic during the practical travel period "
            "for the selected conditions."

        )


# Save recommendation
# ============================================================
def save_recommendations(
    recommendations,
    day_type,
    applied_weather
):

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        output_df = (
            recommendations.copy()
        )


        output_df["day_type"] = (
            day_type.title()
        )


        if applied_weather is None:

            output_df["weather"] = (
                "All weather"
            )

        else:

            output_df["weather"] = (
                applied_weather
            )


        output_df.to_csv(
            RECOMMENDATION_FILE,
            index=False
        )


        logger.info(
            "Travel recommendations saved: %s",
            RECOMMENDATION_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to save travel recommendations: %s",
            e
        )


# Main program
# ============================================================
def main():

    parser = argparse.ArgumentParser(

        description=(
            "Traffic travel-timing recommendation system"
        )

    )


    # Day type
    # ========================================================
    parser.add_argument(

        "--day",

        choices=[
            "weekday",
            "weekend"
        ],

        required=True,

        help=(
            "Select weekday or weekend travel."
        )

    )


    # Optional weather
    # ========================================================
    parser.add_argument(

        "--weather",

        type=str,

        default=None,

        help=(
            "Optional weather condition, "
            "for example Clear, Clouds, Rain, Snow or Mist."
        )

    )


    args = parser.parse_args()


    logger.info(
        "Recommendation requested: day=%s, weather=%s",
        args.day,
        args.weather
    )


    # Load data
    # ========================================================
    df = load_data()


    # Prepare data
    # ========================================================
    df = prepare_data(
        df
    )


    # Filter requested conditions
    # ========================================================
    filtered_df, applied_weather = (
        filter_conditions(

            df,

            day_type=args.day,

            weather=args.weather

        )
    )


    if filtered_df is None:

        print(
            "\nUnable to generate recommendation."
        )

        sys.exit(1)


    # Calculate practical and overnight traffic
    # ========================================================
    (
        daytime_df,
        daytime_average,
        overnight_average

    ) = calculate_period_traffic(
        filtered_df
    )


    # Find practical recommendations
    # ========================================================
    recommendations = (
        find_recommendations(

            daytime_df,

            number_of_windows=3

        )
    )


    # Display recommendation
    # ========================================================
    display_recommendation(

        recommendations,

        day_type=args.day,

        daytime_average=daytime_average,

        overnight_average=overnight_average,

        requested_weather=args.weather,

        applied_weather=applied_weather

    )


    # Save recommendations
    # ========================================================
    save_recommendations(

        recommendations,

        day_type=args.day,

        applied_weather=applied_weather

    )


    logger.info(
        "Traffic recommendation completed successfully."
    )


if __name__ == "__main__":
    main()