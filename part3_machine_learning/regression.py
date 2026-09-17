import logging
import sys
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)


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

RESULTS_FILE = (
    RESULTS_DIR
    / "regression_results.csv"
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

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False


# Load Part 3 dataset
# ============================================================
def load_data():

    try:

        df = pd.read_csv(DATA_FILE)

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


# Create common feature set
# ============================================================
def prepare_features(df):

    # Weather one-hot encoded columns from Part 2
    weather_features = [

        column
        for column in df.columns

        if column.startswith("weather_")

        and column not in [
            "weather_main",
            "weather_description"
        ]

    ]


    # Common engineered features
    feature_columns = [

        # Time-based features
        "hour",
        "day_of_week",
        "is_weekend",

        # Cyclical encodings
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",

        # Holiday flag
        "is_holiday",

        # Weather-related continuous features
        "temp_scaled",
        "rain_scaled",
        "snow_1h",
        "clouds_all"

    ]


    # Add weather encodings
    feature_columns = (
        feature_columns
        + weather_features
    )


    # Check that all features exist
    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns

    ]


    if missing_columns:

        logger.error(
            "Missing regression features: %s",
            missing_columns
        )

        sys.exit(1)


    X = df[feature_columns].copy()

    logger.info(
        "Regression feature set prepared: %d features.",
        X.shape[1]
    )


    return X, feature_columns


# Evaluate regression model
# ============================================================
def evaluate_regressor(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    print("\n========================================")
    print(model_name)
    print("========================================")

    print(
        f"MAE : {mae:.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


    logger.info(
        "%s evaluation completed.",
        model_name
    )


    return {

        "Model": model_name,
        "MAE": mae,
        "R2": r2

    }


# Main program
# ============================================================
def main():

    logger.info(
        "Supervised regression started."
    )


    # Load dataset
    # ========================================================
    df = load_data()


    # Prepare features
    # ========================================================
    X, feature_columns = (
        prepare_features(df)
    )


    # Regression target
    # ========================================================
    y = df["traffic_volume"].copy()


    print("\n========================================")
    print("REGRESSION DATA")
    print("========================================")

    print(
        f"Number of records: {len(df)}"
    )

    print(
        f"Number of features: {len(feature_columns)}"
    )

    print(
        "Target variable: traffic_volume"
    )


    # Train/test split
    # ========================================================
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )


    logger.info(
        "Train/test split completed: %d training rows, %d testing rows.",
        X_train.shape[0],
        X_test.shape[0]
    )


    # Model 1: Linear Regression
    # ========================================================
    linear_model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "regressor",
            LinearRegression()
        )

    ])


    linear_model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Linear Regression trained."
    )


    # Model 2: Random Forest Regressor
    # ========================================================
    random_forest_model = (
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )


    random_forest_model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Random Forest Regressor trained."
    )


    # Evaluate both models
    # ========================================================
    linear_results = evaluate_regressor(

        "Linear Regression",

        linear_model,

        X_test,

        y_test

    )


    random_forest_results = evaluate_regressor(

        "Random Forest Regressor",

        random_forest_model,

        X_test,

        y_test

    )


    # Compare models
    # ========================================================
    results_df = pd.DataFrame([

        linear_results,
        random_forest_results

    ])


    print("\n========================================")
    print("REGRESSION MODEL COMPARISON")
    print("========================================")

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )


    # Save results
    # ========================================================
    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False
        )

        logger.info(
            "Regression results saved: %s",
            RESULTS_FILE
        )

    except Exception as e:

        logger.error(
            "Unable to save regression results: %s",
            e
        )

        sys.exit(1)


    logger.info(
        "Supervised regression completed successfully."
    )


if __name__ == "__main__":
    main()