import logging
import sys
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "part3_ml_data.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "traffic_random_forest_v2.joblib"
)

FEATURE_FILE = (
    MODEL_DIR
    / "traffic_random_forest_v2_features.txt"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)


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
            "Part 3 dataset loaded: %d rows, %d columns.",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception as e:

        logger.error(
            "Unable to load dataset: %s",
            e
        )

        sys.exit(1)


# ============================================================
# Prepare same 23-feature set used previously
# ============================================================

def prepare_features(df):

    weather_features = [

        column
        for column in df.columns

        if column.startswith("weather_")

        and column not in [
            "weather_main",
            "weather_description"
        ]

    ]


    feature_columns = [

        "hour",
        "day_of_week",
        "is_weekend",

        "hour_sin",
        "hour_cos",

        "day_sin",
        "day_cos",

        "is_holiday",

        "temp_scaled",
        "rain_scaled",
        "snow_1h",
        "clouds_all"

    ] + weather_features


    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns

    ]


    if missing_columns:

        logger.error(
            "Missing model features: %s",
            missing_columns
        )

        sys.exit(1)


    X = df[
        feature_columns
    ].copy()


    y = df[
        "traffic_volume"
    ].copy()


    logger.info(
        "Deployment feature set prepared: %d features.",
        len(feature_columns)
    )


    return X, y, feature_columns


# ============================================================
# Train deployment model
# ============================================================

def train_model(
    X,
    y
):

    # Use same split as Task 1
    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42

        )
    )


    model = RandomForestRegressor(

        n_estimators=200,

        random_state=42,

        n_jobs=-1

    )


    logger.info(
        "Training Random Forest deployment model v2.0."
    )


    model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Deployment model training completed."
    )


    return model


# ============================================================
# Save model
# ============================================================

def save_model(
    model,
    feature_columns
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_FILE
    )


    with open(
        FEATURE_FILE,
        "w"
    ) as file:

        for feature in feature_columns:

            file.write(
                feature + "\n"
            )


    logger.info(
        "Deployment model saved: %s",
        MODEL_FILE
    )

    logger.info(
        "Deployment feature list saved: %s",
        FEATURE_FILE
    )


# ============================================================
# Main
# ============================================================

def main():

    logger.info(
        "Deployment model preparation started."
    )


    df = load_data()


    X, y, feature_columns = (
        prepare_features(
            df
        )
    )


    model = train_model(
        X,
        y
    )


    save_model(
        model,
        feature_columns
    )


    print("\n========================================")
    print("DEPLOYMENT MODEL READY")
    print("========================================")

    print(
        "\nModel version: v2.0"
    )

    print(
        "Model: Random Forest Regressor"
    )

    print(
        f"Number of features: {len(feature_columns)}"
    )

    print(
        f"Saved model: {MODEL_FILE}"
    )


    logger.info(
        "Deployment model preparation completed successfully."
    )


if __name__ == "__main__":
    main()