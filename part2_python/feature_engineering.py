import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cleaned_traffic.csv"
OUTPUT_DIR = BASE_DIR / "data"
FEATURED_FILE = OUTPUT_DIR / "engineered_traffic.csv"


def setup_logging():
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # Append Task 2 logs to the existing pipeline.log
    file_handler = logging.FileHandler(
        BASE_DIR / "pipeline.log",
        mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False


# Load the cleaned data from Task 1
# ============================================================
def load_data(file_path):

    try:
        df = pd.read_csv(file_path)

        df["date_time"] = pd.to_datetime(
            df["date_time"],
            errors="coerce"
        )

        logger.info(
            "Cleaned data loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError) as error:

        logger.error(
            "Failed to load cleaned data: %s",
            error,
            exc_info=True
        )

        return None


# Create a simple min-max scaling function
# ============================================================
def min_max_scale(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# Feature engineering
# ============================================================
def engineer_features(df):

    df = df.copy()

    logger.info(
        "Dataset shape before feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )


    # Create time-based features
    # ============================================================
    df["hour"] = df["date_time"].dt.hour

    df["day_of_week"] = (
        df["date_time"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    logger.info(
        "Time features created: hour, day_of_week and is_weekend."
    )


    # Create cyclical encoding for hour and day of week
    # ============================================================
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    logger.info(
        "Cyclical time features created for hour and day of week."
    )


    # Create weather features
    # ============================================================
    precipitation_weather = [
        "Rain",
        "Snow",
        "Drizzle",
        "Thunderstorm"
    ]

    df["is_precipitation"] = (
        df["weather_main"]
        .isin(precipitation_weather)
        .astype(int)
    )

    weather_encoded = pd.get_dummies(
        df["weather_main"],
        prefix="weather",
        dtype=int
    )

    df = pd.concat(
        [df, weather_encoded],
        axis=1
    )

    logger.info(
        "Weather features created using a precipitation indicator "
        "and one-hot encoding of weather_main."
    )


    # Scale continuous numerical variables
    # ============================================================
    df["temp_scaled"] = min_max_scale(
        df["temp"]
    )

    df["rain_scaled"] = min_max_scale(
        df["rain_1h"]
    )

    logger.info(
        "Temperature and rainfall were scaled using min-max scaling."
    )


    # Create a data-driven congestion category using traffic-volume quartiles
    # ============================================================
    q1 = df["traffic_volume"].quantile(0.25)
    q2 = df["traffic_volume"].quantile(0.50)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug(
        "Traffic-volume quartile thresholds: "
        "Q1=%.2f, Q2=%.2f, Q3=%.2f",
        q1,
        q2,
        q3
    )


    def congestion_category(value):

        if value <= q1:
            return "Low"

        elif value <= q2:
            return "Moderate"

        elif value <= q3:
            return "High"

        else:
            return "Very High"


    df["congestion_category"] = (
        df["traffic_volume"]
        .apply(congestion_category)
    )

    logger.info(
        "Congestion category created using traffic-volume quartiles: "
        "Low, Moderate, High and Very High."
    )


    # Log the final dataset shape
    # ============================================================
    logger.info(
        "Dataset shape after feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )

    return df


# Save the engineered data to a CSV file
# ============================================================
def save_data(df, file_path):

    try:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            file_path,
            index=False
        )

        logger.info(
            "Engineered dataset saved successfully to: %s",
            file_path
        )

        return True

    except OSError as error:

        logger.error(
            "Failed to save engineered dataset: %s",
            error,
            exc_info=True
        )

        return False


# Run the feature engineering process
# ============================================================
def main():

    setup_logging()

    logger.info(
        "Feature engineering started."
    )

    df = load_data(DATA_FILE)

    if df is None:

        logger.error(
            "Feature engineering stopped: "
            "cleaned data could not be loaded."
        )

        sys.exit(1)


    df = engineer_features(df)


    if not save_data(
        df,
        FEATURED_FILE
    ):

        sys.exit(1)


    logger.info(
        "Feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()