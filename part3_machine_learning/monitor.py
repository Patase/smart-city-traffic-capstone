import logging
import sys
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "part3_ml_data.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "traffic_random_forest_v2.joblib"
)

FEATURE_FILE = (
    BASE_DIR
    / "models"
    / "traffic_random_forest_v2_features.txt"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

MONITORING_FILE = (
    RESULTS_DIR
    / "monitoring_results.csv"
)

STATUS_FILE = (
    RESULTS_DIR
    / "monitoring_status.txt"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)


# ============================================================
# Monitoring threshold
# ============================================================

# If current MAE increases by more than 25%
# compared with the reference MAE,
# the system will generate an ALERT.

DRIFT_THRESHOLD = 0.25


# ============================================================
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


# ============================================================
# Load dataset
# ============================================================

def load_data():

    try:

        df = pd.read_csv(
            DATA_FILE
        )

        logger.info(
            "Monitoring dataset loaded: %d rows, %d columns.",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load monitoring dataset: %s",
            e
        )

        sys.exit(1)


# ============================================================
# Load deployed model
# ============================================================

def load_model():

    try:

        model = joblib.load(
            MODEL_FILE
        )

        logger.info(
            "Deployed Random Forest v2.0 model loaded successfully."
        )

        return model

    except Exception as e:

        logger.error(
            "Unable to load deployed model: %s",
            e
        )

        sys.exit(1)


# ============================================================
# Load feature list
# ============================================================

def load_features():

    try:

        with open(
            FEATURE_FILE,
            "r"
        ) as file:

            feature_columns = [

                line.strip()

                for line in file

                if line.strip()

            ]


        logger.info(
            "Loaded %d deployment features.",
            len(feature_columns)
        )

        return feature_columns

    except Exception as e:

        logger.error(
            "Unable to load deployment feature list: %s",
            e
        )

        sys.exit(1)


# ============================================================
# Prepare monitoring data
# ============================================================

def prepare_monitoring_data(
    df,
    feature_columns
):

    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns

    ]


    if missing_columns:

        logger.error(
            "Missing monitoring features: %s",
            missing_columns
        )

        sys.exit(1)


    X = df[
        feature_columns
    ].copy()


    y = df[
        "traffic_volume"
    ].copy()


    # Reproduce the same held-out test set used
    # when the deployment model was trained.
    # ========================================================

    _, X_test, _, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42

        )
    )


    # Keep original row order so that the latter part
    # can act as a simulated "more recent" period.
    # ========================================================

    monitoring_df = (
        X_test.copy()
    )


    monitoring_df[
        "actual_traffic_volume"
    ] = y_test


    monitoring_df = (

        monitoring_df

        .sort_index()

        .reset_index(drop=True)

    )


    logger.info(
        "Held-out monitoring dataset prepared: %d records.",
        len(monitoring_df)
    )


    return monitoring_df


# ============================================================
# Split reference and current monitoring periods
# ============================================================

def split_monitoring_periods(
    monitoring_df
):

    midpoint = (
        len(monitoring_df)
        // 2
    )


    reference_df = (
        monitoring_df
        .iloc[:midpoint]
        .copy()
    )


    current_df = (
        monitoring_df
        .iloc[midpoint:]
        .copy()
    )


    logger.info(
        "Reference monitoring period: %d records.",
        len(reference_df)
    )

    logger.info(
        "Current monitoring period: %d records.",
        len(current_df)
    )


    return (
        reference_df,
        current_df
    )


# ============================================================
# Calculate MAE
# ============================================================

def calculate_mae(
    model,
    df,
    feature_columns
):

    X = df[
        feature_columns
    ]


    y_actual = df[
        "actual_traffic_volume"
    ]


    predictions = model.predict(
        X
    )


    mae = mean_absolute_error(
        y_actual,
        predictions
    )


    return mae


# ============================================================
# Check prediction-error drift
# ============================================================

def monitor_drift(
    reference_mae,
    current_mae
):

    # Avoid division by zero
    # ========================================================

    if reference_mae == 0:

        logger.error(
            "Reference MAE is zero. Drift cannot be calculated."
        )

        return (
            0,
            "ALERT",
            "Requires investigation"
        )


    drift_ratio = (

        (
            current_mae
            - reference_mae
        )

        / reference_mae

    )


    drift_percent = (
        drift_ratio
        * 100
    )


    # Alert if error increased by more than 25%
    # ========================================================

    if drift_ratio > DRIFT_THRESHOLD:

        status = (
            "ALERT"
        )

        message = (
            "Requires investigation"
        )


        logger.warning(
            "Prediction-error drift detected: %.2f%% increase.",
            drift_percent
        )


    else:

        status = (
            "PASS"
        )

        message = (
            "Normal"
        )


        logger.info(
            "Prediction-error drift remains within threshold."
        )


    return (
        drift_percent,
        status,
        message
    )


# ============================================================
# Save monitoring results
# ============================================================

def save_results(
    reference_mae,
    current_mae,
    drift_percent,
    status,
    message
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    results = pd.DataFrame({

        "Metric": [
            "Reference MAE",
            "Current MAE",
            "MAE Drift Percent",
            "Alert Threshold Percent"
        ],

        "Value": [
            reference_mae,
            current_mae,
            drift_percent,
            DRIFT_THRESHOLD * 100
        ]

    })


    results.to_csv(
        MONITORING_FILE,
        index=False
    )


    with open(
        STATUS_FILE,
        "w"
    ) as file:

        file.write(
            f"Status: {status}\n"
        )

        file.write(
            f"Interpretation: {message}\n"
        )

        file.write(
            f"Reference MAE: {reference_mae:.2f}\n"
        )

        file.write(
            f"Current MAE: {current_mae:.2f}\n"
        )

        file.write(
            f"MAE drift: {drift_percent:.2f}%\n"
        )

        file.write(
            "Alert threshold: 25.00%\n"
        )


    logger.info(
        "Monitoring results saved: %s",
        MONITORING_FILE
    )

    logger.info(
        "Monitoring status saved: %s",
        STATUS_FILE
    )


# ============================================================
# Display monitoring report
# ============================================================

def display_report(
    reference_mae,
    current_mae,
    drift_percent,
    status,
    message
):

    print("\n========================================")
    print("MODEL MONITORING REPORT")
    print("========================================")


    print(
        f"\nModel version: v2.0"
    )


    print(
        "Monitoring metric: Prediction MAE"
    )


    print(
        f"\nReference MAE: {reference_mae:.2f}"
    )


    print(
        f"Current MAE: {current_mae:.2f}"
    )


    print(
        f"MAE drift: {drift_percent:.2f}%"
    )


    print(
        f"Alert threshold: {DRIFT_THRESHOLD * 100:.0f}%"
    )


    print("\n----------------------------------------")
    print("MONITORING STATUS")
    print("----------------------------------------")


    print(
        f"{status} / {message}"
    )


    if status == "PASS":

        print(

            "\nThe change in prediction error is within "
            "the simulated monitoring threshold. "
            "No investigation is currently required."

        )


    else:

        print(

            "\nPrediction error has increased beyond "
            "the simulated monitoring threshold. "
            "The model should be investigated before "
            "continued operational use."

        )


# ============================================================
# Main
# ============================================================

def main():

    logger.info(
        "Model monitoring simulation started."
    )


    # Load resources
    # ========================================================

    df = load_data()

    model = load_model()

    feature_columns = (
        load_features()
    )


    # Prepare held-out monitoring dataset
    # ========================================================

    monitoring_df = (
        prepare_monitoring_data(

            df,
            feature_columns

        )
    )


    # Simulate reference and current periods
    # ========================================================

    (
        reference_df,
        current_df

    ) = split_monitoring_periods(
        monitoring_df
    )


    # Calculate prediction error
    # ========================================================

    reference_mae = (
        calculate_mae(

            model,
            reference_df,
            feature_columns

        )
    )


    current_mae = (
        calculate_mae(

            model,
            current_df,
            feature_columns

        )
    )


    # Monitor for drift
    # ========================================================

    (
        drift_percent,
        status,
        message

    ) = monitor_drift(

        reference_mae,
        current_mae

    )


    # Save results
    # ========================================================

    save_results(

        reference_mae,
        current_mae,
        drift_percent,
        status,
        message

    )


    # Display monitoring report
    # ========================================================

    display_report(

        reference_mae,
        current_mae,
        drift_percent,
        status,
        message

    )


    logger.info(
        "Model monitoring simulation completed successfully."
    )


if __name__ == "__main__":
    main()