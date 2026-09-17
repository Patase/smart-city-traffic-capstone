import logging
import sys
import pandas as pd
from pathlib import Path


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)

def setup_logging():
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(BASE_DIR / "pipeline.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Metro_Interstate_Traffic_Volume.csv"
OUTPUT_DIR = BASE_DIR / "data"
CLEANED_FILE = OUTPUT_DIR / "cleaned_traffic.csv"

#Load the data from the CSV file
# ============================================================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Data loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError) as error:
        logger.error(
            "Failed to load data: %s",
            error,
            exc_info=True
        )

        return None

#Validate the schema of the DataFrame against the expected columns
# ============================================================

EXP_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


def validate_schema(df):
    missing_columns = []

    for column in EXP_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        logger.error(
            "Schema validation failed. Missing columns: %s",
            missing_columns
        )
        return False

    logger.info("Schema validation successful.")
    return True

#Clean the data by stripping whitespace from categorical columns
# ============================================================

def clean_data(df):
    df = df.copy()

    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description"
    ]

    for column in categorical_columns:
        before = df[column].dropna().nunique()
        original = df[column].copy()

        df[column] = df[column].str.strip()
        if column == "weather_description":
            df[column] = df[column].str.lower()

        changed = (original.fillna("") != df[column].fillna("")).sum()
        after = df[column].dropna().nunique()

        if changed > 0:
            logger.warning(
                "%d values in '%s' were standardised (whitespace/case); "
                "unique categories %d -> %d.",
                changed, column, before, after
            )
        else:
            logger.info("No inconsistent values found in '%s'.", column)

#Clean the data by validating the date_time column 
# ============================================================
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        df = df.dropna(subset=["date_time"]).copy()

        logger.warning(
            "%d rows were dropped because date_time could not be parsed.",
            invalid_dates,
        )
    else:
        logger.info(
            "Date/time validation completed: no invalid dates found."
        )


#Clean the data by checking for duplicate rows
# ============================================================
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().copy()

        logger.warning(
            "%d duplicate rows were removed.",
            duplicate_count,
        )
    else:
        logger.info(
            "Duplicate-row checked: no exact duplicates found."
        )


#Clean the data by replacing impossible temperature values (0 K) with the median temperature for that month
# since we've already looked at the data and there are only single outliers value at 0 K
# ============================================================
    invalid_temp = df["temp"] == 0

    if invalid_temp.sum() == 0:
        logger.info("Temperature validation completed: no 0 K values found.")
    else:
        for month in range(1, 13):
            in_month = df["date_time"].dt.month == month
            to_fix = invalid_temp & in_month
            n_fix = to_fix.sum()

            if n_fix == 0:
                continue

            month_median = df.loc[in_month & ~invalid_temp, "temp"].median()
            df.loc[to_fix, "temp"] = month_median

            logger.warning(
                "%d temperature value(s) at 0 K in month %d were imputed "
                "with that month's median (%.2f K).",
                n_fix, month, month_median
            )

#Clean the data by replacing impossible rainfall values (> 9000 mm) with the median rainfall value
# ============================================================
    invalid_rain = df["rain_1h"] > 9000
    invalid_rain_count = invalid_rain.sum()

    if invalid_rain_count > 0:

        rain_median = df.loc[
            df["rain_1h"] <= 9000,
            "rain_1h"
        ].median()

        df.loc[
            invalid_rain,
            "rain_1h"
        ] = rain_median

        logger.warning(
            "%d rainfall value(s) above 9000 mm were imputed "
            "using the median rainfall value (%.2f mm).",
            invalid_rain_count,
            rain_median
        )

    else:
        logger.info(
            "Rainfall validation completed: no values above 9000 mm found."
        )
    return df

#Save the cleaned data to a CSV file
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
            "Cleaned dataset saved successfully to: %s",
            file_path,
        )

        return True

    except OSError as error:

        logger.error(
            "Failed to save cleaned dataset: %s",
            error,
            exc_info=True,
        )

        return False

def main():
    setup_logging()

    df = load_data(DATA_FILE)
    if df is None:
        logger.error("Pipeline stopped: data could not be loaded.")
        sys.exit(1)

    if not validate_schema(df):
        logger.error("Pipeline stopped: schema validation failed.")
        sys.exit(1)

    df = clean_data(df)
    logger.info("Cleaning finished: %d rows, %d columns.", df.shape[0], df.shape[1])

    if not save_data(df, CLEANED_FILE):
        sys.exit(1)


if __name__ == "__main__":
    main()