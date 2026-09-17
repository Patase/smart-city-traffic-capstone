import logging
import sys
import pandas as pd
from pathlib import Path


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)


# File paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_FILE = (
    PROJECT_DIR
    / "part2_python"
    / "data"
    / "engineered_traffic.csv"
)

OUTPUT_DIR = BASE_DIR / "data"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "part3_ml_data.csv"
)


# Weather groups used for proxy accident-risk label
# ============================================================

# Operational definition used for this project
SEVERE_WEATHER = [
    "Thunderstorm",
    "Snow",
    "Squall"
]

LOW_VISIBILITY_WEATHER = [
    "Fog",
    "Mist",
    "Haze",
    "Smoke"
]


# Load Part 2 engineered dataset
# ============================================================
def load_data():

    try:

        df = pd.read_csv(INPUT_FILE)

        logger.info(
            "Part 2 engineered data loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load Part 2 engineered data: %s",
            e
        )

        sys.exit(1)


# Create Part 3 variables
# ============================================================
def create_part3_variables(df):

    df = df.copy()


    # Create holiday flag
    # ========================================================

    df["is_holiday"] = (
        df["holiday"].notna()
        & (df["holiday"].str.lower() != "none")
    ).astype(int)

    logger.info(
        "Holiday indicator created."
    )


    # Create congestion category using quartiles
    # ========================================================

    q1 = df["traffic_volume"].quantile(0.25)
    q2 = df["traffic_volume"].quantile(0.50)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.info(
        "Traffic volume quartiles calculated."
    )

    logger.info(
        "Q1 = %.2f, Q2 = %.2f, Q3 = %.2f",
        q1,
        q2,
        q3
    )


    def congestion_category(value):

        if value <= q1:
            return "Low"

        elif value <= q2:
            return "Medium"

        elif value <= q3:
            return "High"

        else:
            return "Severe"


    df["congestion_category_ml"] = (
        df["traffic_volume"]
        .apply(congestion_category)
    )

    logger.info(
        "Part 3 congestion category created."
    )


    # Create low-visibility indicator
    # ========================================================

    df["is_low_visibility"] = (
        df["weather_main"]
        .isin(LOW_VISIBILITY_WEATHER)
        .astype(int)
    )

    logger.info(
        "Low-visibility indicator created."
    )


    # Create severe-weather indicator
    # ========================================================

    df["is_severe_weather"] = (
        df["weather_main"]
        .isin(SEVERE_WEATHER)
        .astype(int)
    )

    logger.info(
        "Severe-weather indicator created."
    )


    # Create proxy accident-risk label
    # ========================================================

    high_congestion = (
        df["congestion_category_ml"]
        .isin(["High", "Severe"])
    )

    risky_weather = (
        (df["is_severe_weather"] == 1)
        | (df["is_low_visibility"] == 1)
    )

    df["high_risk"] = (
        high_congestion
        & risky_weather
    ).astype(int)

    logger.info(
        "Proxy high-risk label created."
    )


    return df


# Save Part 3 dataset
# ============================================================
def save_data(df):

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        logger.info(
            "Part 3 dataset saved successfully: %s",
            OUTPUT_FILE
        )

    except Exception as e:

        logger.error(
            "Unable to save Part 3 dataset: %s",
            e
        )

        sys.exit(1)


# Main program
# ============================================================
def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    logger.info(
        "Part 3 dataset preparation started."
    )


    # Load Part 2 dataset
    df = load_data()


    # Create Part 3 variables
    df = create_part3_variables(df)


    # Save new dataset
    save_data(df)


    # Show summary
    # ========================================================

    print("\n========================================")
    print("PART 3 DATASET SUMMARY")
    print("========================================")

    print(
        "\nDataset shape:",
        df.shape
    )


    print("\nCongestion categories:")

    print(
        df["congestion_category_ml"]
        .value_counts()
    )


    print("\nHigh-risk proxy label:")

    print(
        df["high_risk"]
        .value_counts()
    )


    print("\nHigh-risk percentage:")

    print(
        df["high_risk"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


    print("\nHoliday flag:")

    print(
        df["is_holiday"]
        .value_counts()
    )


    print("\nLow-visibility indicator:")

    print(
        df["is_low_visibility"]
        .value_counts()
    )


    print("\nSevere-weather indicator:")

    print(
        df["is_severe_weather"]
        .value_counts()
    )


    logger.info(
        "Part 3 dataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()