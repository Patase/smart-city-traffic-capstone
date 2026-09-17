import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Logger setup
# ============================================================
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "engineered_traffic.csv"
FIGURE_DIR = BASE_DIR / "figures"


def setup_logging():
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # Append Task 3 logs to the existing pipeline.log
    file_handler = logging.FileHandler(
        BASE_DIR / "pipeline.log",
        mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False


# Load the engineered data from Task 2
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


# Prepare the data for traffic visualisation
# ============================================================
def prepare_visualisation_data(df):

    df = df.copy()

    duplicate_times = df.duplicated(
        subset=["date_time"]
    ).sum()

    if duplicate_times > 0:

        df = df.drop_duplicates(
            subset=["date_time"]
        ).copy()

        logger.info(
            "%d repeated date_time rows were excluded from the visualisations "
            "to avoid counting the same hourly traffic volume more than once.",
            duplicate_times
        )

    else:
        logger.info(
            "No repeated date_time rows found for visualisation."
        )

    return df


# Figure 1: Average traffic volume by hour
# ============================================================
def plot_traffic_by_hour(df):

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o"
    )

    plt.title(
        "Average Traffic Volume by Hour"
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Average Traffic Volume"
    )

    plt.xticks(
        range(0, 24)
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR / "traffic_by_hour.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to: %s",
        figure_path
    )

    peak_hour = hourly_traffic.idxmax()
    peak_traffic = hourly_traffic.max()

    logger.info(
        "Hourly traffic interpretation: the highest average traffic "
        "occurred at %02d:00 with an average volume of %.2f vehicles.",
        peak_hour,
        peak_traffic
    )


# Figure 2: Weekday versus weekend traffic
# ============================================================
def plot_weekday_vs_weekend(df):

    traffic_type = (
        df.groupby("is_weekend")["traffic_volume"]
        .mean()
    )

    labels = [
        "Weekday",
        "Weekend"
    ]

    values = [
        traffic_type.get(0, 0),
        traffic_type.get(1, 0)
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Average Traffic Volume: Weekday vs Weekend"
    )

    plt.xlabel(
        "Day Type"
    )

    plt.ylabel(
        "Average Traffic Volume"
    )

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR / "weekday_vs_weekend.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to: %s",
        figure_path
    )

    logger.info(
        "Weekday/weekend interpretation: weekday average traffic was %.2f "
        "vehicles and weekend average traffic was %.2f vehicles.",
        values[0],
        values[1]
    )


# Figure 3: Temperature versus traffic volume
# ============================================================
def plot_temperature_vs_traffic(df):

    temperature_c = (
        df["temp"] - 273.15
    )

    correlation = (
        temperature_c.corr(
            df["traffic_volume"]
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        temperature_c,
        df["traffic_volume"],
        alpha=0.3,
        s=10
    )

    plt.title(
        "Temperature vs Traffic Volume"
    )

    plt.xlabel(
        "Temperature (°C)"
    )

    plt.ylabel(
        "Traffic Volume"
    )

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR / "temperature_vs_traffic.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to: %s",
        figure_path
    )

    logger.info(
        "Temperature/traffic interpretation: the Pearson correlation "
        "between temperature and traffic volume was %.4f.",
        correlation
    )


# Run all visualisations
# ============================================================
def main():

    setup_logging()

    logger.info(
        "Traffic visualisation started."
    )

    df = load_data(
        DATA_FILE
    )

    if df is None:

        logger.error(
            "Visualisation stopped: engineered data could not be loaded."
        )

        sys.exit(1)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = prepare_visualisation_data(
        df
    )

    plot_traffic_by_hour(
        df
    )

    plot_weekday_vs_weekend(
        df
    )

    plot_temperature_vs_traffic(
        df
    )

    logger.info(
        "Traffic visualisation completed successfully."
    )


if __name__ == "__main__":
    main()