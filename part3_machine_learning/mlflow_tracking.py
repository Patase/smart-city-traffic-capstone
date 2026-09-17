import logging
import sys
import pandas as pd
import mlflow
import mlflow.sklearn

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

MLFLOW_DB = (
    BASE_DIR
    / "mlflow.db"
)

MLFLOW_ARTIFACTS_DIR = (
    BASE_DIR
    / "mlartifacts"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "mlflow_model_comparison.csv"
)

FEATURES_FILE = (
    RESULTS_DIR
    / "mlflow_features.txt"
)

LOG_FILE = (
    BASE_DIR
    / "part3_ml.log"
)

EXPERIMENT_NAME = (
    "Traffic_Volume_Regression"
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


# Prepare common feature set
# ============================================================
def prepare_features(df):

    # Weather one-hot encoded features
    weather_features = [

        column
        for column in df.columns

        if column.startswith(
            "weather_"
        )

        and column not in [
            "weather_main",
            "weather_description"
        ]

    ]


    # Same common feature set used previously
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

        # Holiday
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


    # Check for missing variables
    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns

    ]


    if missing_columns:

        logger.error(
            "Missing MLflow features: %s",
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
        "MLflow feature set prepared: %d features.",
        X.shape[1]
    )


    return X, y, feature_columns


# Evaluate regression model
# ============================================================
def evaluate_model(
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


    return mae, r2


# Configure MLflow
# ============================================================
def configure_mlflow():

    try:

        # Create artifact folder
        MLFLOW_ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # Use SQLite instead of deprecated file tracking backend
        tracking_uri = (
            f"sqlite:///{MLFLOW_DB.resolve()}"
        )


        mlflow.set_tracking_uri(
            tracking_uri
        )


        # Check whether experiment already exists
        experiment = (
            mlflow.get_experiment_by_name(
                EXPERIMENT_NAME
            )
        )


        # Create experiment only if it does not already exist
        if experiment is None:

            mlflow.create_experiment(

                name=EXPERIMENT_NAME,

                artifact_location=(
                    MLFLOW_ARTIFACTS_DIR
                    .resolve()
                    .as_uri()
                )

            )

            logger.info(
                "New MLflow experiment created: %s",
                EXPERIMENT_NAME
            )


        else:

            logger.info(
                "Existing MLflow experiment found: %s",
                EXPERIMENT_NAME
            )


        mlflow.set_experiment(
            EXPERIMENT_NAME
        )


        logger.info(
            "MLflow tracking database: %s",
            MLFLOW_DB
        )


    except Exception as e:

        logger.error(
            "Unable to configure MLflow: %s",
            e
        )

        sys.exit(1)


# Save feature list
# ============================================================
def save_feature_list(
    feature_columns
):

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            FEATURES_FILE,
            "w"
        ) as file:

            for feature in feature_columns:

                file.write(
                    feature + "\n"
                )


        logger.info(
            "Feature list saved: %s",
            FEATURES_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to save feature list: %s",
            e
        )

        sys.exit(1)


# Run Linear Regression experiment
# ============================================================
def run_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_columns
):

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


    with mlflow.start_run(
        run_name="Linear_Regression"
    ):


        # Train model
        # ====================================================
        linear_model.fit(
            X_train,
            y_train
        )


        # Evaluate model
        # ====================================================
        mae, r2 = evaluate_model(
            linear_model,
            X_test,
            y_test
        )


        # Log model parameters
        # ====================================================
        mlflow.log_param(
            "model_type",
            "Linear Regression"
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "number_of_features",
            len(feature_columns)
        )


        # Log evaluation metrics
        # ====================================================
        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "R2",
            r2
        )


        # Log feature list
        # ====================================================
        mlflow.log_artifact(
            FEATURES_FILE
        )


        # Log trained model
        # ====================================================
        mlflow.sklearn.log_model(
            linear_model,
            name="model"
        )


        logger.info(
            "Linear Regression logged to MLflow: MAE=%.2f, R2=%.4f",
            mae,
            r2
        )


    return {

        "Model": "Linear Regression",

        "MAE": mae,

        "R2": r2

    }


# Run Random Forest experiment
# ============================================================
def run_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_columns
):

    random_forest_model = (
        RandomForestRegressor(

            n_estimators=200,

            random_state=42,

            n_jobs=-1

        )
    )


    with mlflow.start_run(
        run_name="Random_Forest_Regressor"
    ):


        # Train model
        # ====================================================
        random_forest_model.fit(
            X_train,
            y_train
        )


        # Evaluate model
        # ====================================================
        mae, r2 = evaluate_model(
            random_forest_model,
            X_test,
            y_test
        )


        # Log parameters
        # ====================================================
        mlflow.log_param(
            "model_type",
            "Random Forest Regressor"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "number_of_features",
            len(feature_columns)
        )


        # Log evaluation metrics
        # ====================================================
        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "R2",
            r2
        )


        # Log feature list
        # ====================================================
        mlflow.log_artifact(
            FEATURES_FILE
        )


        # Log trained model
        # ====================================================
        mlflow.sklearn.log_model(
            random_forest_model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )


        logger.info(
            "Random Forest Regressor logged to MLflow: MAE=%.2f, R2=%.4f",
            mae,
            r2
        )


    return {

        "Model": "Random Forest Regressor",

        "MAE": mae,

        "R2": r2

    }


# Save model comparison
# ============================================================
def save_results(
    results
):

    try:

        results_df = pd.DataFrame(
            results
        )


        results_df.to_csv(
            RESULTS_FILE,
            index=False
        )


        logger.info(
            "MLflow model comparison saved: %s",
            RESULTS_FILE
        )


        return results_df


    except Exception as e:

        logger.error(
            "Unable to save MLflow results: %s",
            e
        )

        sys.exit(1)


# Main program
# ============================================================
def main():

    logger.info(
        "Task 4 MLflow experiment tracking started."
    )


    # Create results folder
    # ========================================================
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Configure MLflow
    # ========================================================
    configure_mlflow()


    # Load dataset
    # ========================================================
    df = load_data()


    # Prepare common features
    # ========================================================
    X, y, feature_columns = (
        prepare_features(
            df
        )
    )


    print("\n========================================")
    print("TASK 4 - MLFLOW EXPERIMENT TRACKING")
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

    print(
        f"Experiment: {EXPERIMENT_NAME}"
    )


    # Save feature list
    # ========================================================
    save_feature_list(
        feature_columns
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


    # Store experiment results
    # ========================================================
    results = []


    # Experiment 1: Linear Regression
    # ========================================================
    print("\nRunning Linear Regression experiment...")


    linear_results = (
        run_linear_regression(

            X_train,
            X_test,

            y_train,
            y_test,

            feature_columns

        )
    )


    results.append(
        linear_results
    )


    # Experiment 2: Random Forest
    # ========================================================
    print(
        "\nRunning Random Forest Regressor experiment..."
    )


    random_forest_results = (
        run_random_forest(

            X_train,
            X_test,

            y_train,
            y_test,

            feature_columns

        )
    )


    results.append(
        random_forest_results
    )


    # Save comparison
    # ========================================================
    results_df = save_results(
        results
    )


    # Display comparison
    # ========================================================
    print("\n========================================")
    print("MLFLOW EXPERIMENT RESULTS")
    print("========================================")


    print(

        results_df

        .round(4)

        .to_string(
            index=False
        )

    )


    print("\nMLflow experiment:")

    print(
        EXPERIMENT_NAME
    )


    print("\nMLflow tracking database:")

    print(
        MLFLOW_DB
    )


    print("\nMLflow artifact directory:")

    print(
        MLFLOW_ARTIFACTS_DIR
    )


    logger.info(
        "Task 4 MLflow experiment tracking completed successfully."
    )


if __name__ == "__main__":
    main()