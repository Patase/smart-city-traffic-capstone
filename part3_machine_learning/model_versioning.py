import logging
import sys
import pandas as pd

from pathlib import Path


# File paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

REGRESSION_RESULTS_FILE = (
    RESULTS_DIR
    / "regression_results.csv"
)

VERSION_FILE = (
    RESULTS_DIR
    / "model_versions.csv"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)


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


# Load regression results
# ============================================================
def load_results():

    try:

        df = pd.read_csv(
            REGRESSION_RESULTS_FILE
        )

        logger.info(
            "Regression results loaded successfully."
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load regression results: %s",
            e
        )

        sys.exit(1)


# Create model versions
# ============================================================
def create_model_versions(results_df):

    versions = []


    for _, row in results_df.iterrows():

        if row["Model"] == "Linear Regression":

            version = "v1.0"

            status = "Baseline"

            description = (
                "Baseline linear regression model "
                "for traffic-volume prediction."
            )


        elif row["Model"] == "Random Forest Regressor":

            version = "v2.0"

            status = "Improved"

            description = (
                "Random Forest regression model "
                "with improved predictive performance."
            )


        else:

            version = "Unknown"

            status = "Other"

            description = (
                "Additional regression model."
            )


        versions.append({

            "Version": version,

            "Model": row["Model"],

            "Status": status,

            "MAE": row["MAE"],

            "R2": row["R2"],

            "Description": description

        })


    version_df = pd.DataFrame(
        versions
    )


    logger.info(
        "Model version table created."
    )


    return version_df


# Save model version table
# ============================================================
def save_versions(version_df):

    try:

        version_df.to_csv(
            VERSION_FILE,
            index=False
        )

        logger.info(
            "Model version table saved: %s",
            VERSION_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to save model version table: %s",
            e
        )

        sys.exit(1)


# Main program
# ============================================================
def main():

    logger.info(
        "Model versioning started."
    )


    results_df = load_results()


    version_df = create_model_versions(
        results_df
    )


    save_versions(
        version_df
    )


    print("\n========================================")
    print("MODEL VERSION HISTORY")
    print("========================================")


    print(
        version_df
        .round(4)
        .to_string(index=False)
    )


    logger.info(
        "Model versioning completed successfully."
    )


if __name__ == "__main__":
    main()
    