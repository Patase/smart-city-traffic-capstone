import argparse
import logging
import sys
import pandas as pd
from pathlib import Path


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "engineered_traffic.csv"


def setup_logging():
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # Append Task 4 logs to the existing pipeline.log
    file_handler = logging.FileHandler(
        BASE_DIR / "pipeline.log",
        mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False


# Load the data from Task 2
# ============================================================
def load_data(file_path):

    try:
        df = pd.read_csv(file_path)

        df["date_time"] = pd.to_datetime(
            df["date_time"],
            errors="coerce"
        )

        logger.info(
            "Engineered data loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError) as error:

        logger.error(
            "Failed to load engineered data: %s",
            error,
            exc_info=True
        )

        return None


# Prepare one traffic record per date and time
# ============================================================
def prepare_data(df):

    df = df.copy()

    duplicate_times = df.duplicated(
        subset=["date_time"]
    ).sum()

    if duplicate_times > 0:

        df = df.drop_duplicates(
            subset=["date_time"]
        ).copy()

        logger.info(
            "%d repeated date_time rows were excluded from CLI traffic queries.",
            duplicate_times
        )

    return df


# Command 1: Query traffic for a specific date and time
# ============================================================
def query_traffic(df, date_text, time_text):

    user_datetime = pd.to_datetime(
        date_text + " " + time_text,
        errors="coerce"
    )

    if pd.isna(user_datetime):

        logger.error(
            "Invalid date/time input: %s %s",
            date_text,
            time_text
        )

        print(
            "Error: invalid date or time. "
            "Please use YYYY-MM-DD HH:MM format."
        )

        return

    result = df[
        df["date_time"] == user_datetime
    ]

    if result.empty:

        print(
            f"No traffic record was found for {user_datetime}."
        )

        return

    row = result.iloc[0]

    print()
    print("Traffic information")
    print("-------------------")
    print(f"Date/time: {row['date_time']}")
    print(f"Traffic volume: {row['traffic_volume']}")
    print(f"Weather: {row['weather_main']}")
    print(f"Congestion category: {row['congestion_category']}")
    print()


# Command 2: Identify high-traffic periods
# ============================================================
def high_traffic(df, threshold, limit):

    result = (
        df[
            df["traffic_volume"] > threshold
        ]
        .sort_values(
            by="traffic_volume",
            ascending=False
        )
        .head(limit)
    )

    if result.empty:

        print(
            f"No traffic records were found above {threshold} vehicles."
        )

        return

    print()
    print(
        f"Top {len(result)} traffic periods above {threshold} vehicles"
    )
    print("--------------------------------------------------")

    print(
        result[
            [
                "date_time",
                "traffic_volume",
                "weather_main",
                "congestion_category"
            ]
        ].to_string(index=False)
    )

    print()


# Command 3: Compare weekday and weekend traffic
# ============================================================
def compare_days(df):

    traffic_average = (
        df.groupby("is_weekend")["traffic_volume"]
        .mean()
    )

    weekday_average = traffic_average.get(
        0,
        0
    )

    weekend_average = traffic_average.get(
        1,
        0
    )

    difference = (
        weekday_average - weekend_average
    )

    print()
    print("Weekday vs Weekend Traffic")
    print("--------------------------")
    print(
        f"Average weekday traffic: {weekday_average:.2f} vehicles"
    )
    print(
        f"Average weekend traffic: {weekend_average:.2f} vehicles"
    )
    print(
        f"Difference: {difference:.2f} vehicles"
    )
    print()


# Create the CLI commands
# ============================================================
def create_parser():

    parser = argparse.ArgumentParser(
        description="Mini Traffic Analytics Application"
    )

    commands = parser.add_subparsers(
        dest="command"
    )

    # Query command
    query_parser = commands.add_parser(
        "query",
        help="Query traffic for a specific date and time"
    )

    query_parser.add_argument(
        "date",
        help="Date in YYYY-MM-DD format"
    )

    query_parser.add_argument(
        "time",
        help="Time in HH:MM format"
    )

    # High-traffic command
    high_parser = commands.add_parser(
        "high-traffic",
        help="Show high-traffic periods"
    )

    high_parser.add_argument(
        "--threshold",
        type=int,
        default=5500,
        help="Traffic-volume threshold (default: 5500)"
    )

    high_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to display (default: 10)"
    )

    # Compare-days command
    commands.add_parser(
        "compare-days",
        help="Compare average weekday and weekend traffic"
    )

    return parser


# Run the mini traffic analytics application
# ============================================================
def main():

    setup_logging()

    logger.info(
        "Traffic analytics CLI started."
    )

    df = load_data(
        DATA_FILE
    )

    if df is None:

        logger.error(
            "CLI stopped: engineered data could not be loaded."
        )

        sys.exit(1)

    df = prepare_data(
        df
    )

    parser = create_parser()

    args = parser.parse_args()

    if args.command is None:

        parser.print_help()
        return

    logger.info(
        "CLI command invoked: %s | arguments: %s",
        args.command,
        vars(args)
    )

    try:

        if args.command == "query":

            query_traffic(
                df,
                args.date,
                args.time
            )

        elif args.command == "high-traffic":

            high_traffic(
                df,
                args.threshold,
                args.limit
            )

        elif args.command == "compare-days":

            compare_days(
                df
            )

        logger.info(
            "CLI command completed successfully: %s",
            args.command
        )

    except Exception as error:

        logger.error(
            "CLI command failed: %s",
            error,
            exc_info=True
        )

        print(
            "Error: the command could not be completed. "
            "Please check the input and try again."
        )


if __name__ == "__main__":
    main()