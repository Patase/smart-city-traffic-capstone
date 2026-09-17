import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

import shap


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

NN_RESULTS_FILE = (
    RESULTS_DIR
    / "neural_network_results.csv"
)

SHAP_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "shap_feature_importance.csv"
)

LOSS_FIGURE_FILE = (
    RESULTS_DIR
    / "neural_network_loss.png"
)

SHAP_FIGURE_FILE = (
    RESULTS_DIR
    / "shap_summary.png"
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


# Prepare common feature set
# ============================================================
def prepare_features(df):

    # Weather one-hot encoded features from Part 2
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

        # Time-based features
        "hour",
        "day_of_week",
        "is_weekend",

        # Cyclical features
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",

        # Holiday
        "is_holiday",

        # Weather-related features
        "temp_scaled",
        "rain_scaled",
        "snow_1h",
        "clouds_all"

    ]


    feature_columns = (
        feature_columns
        + weather_features
    )


    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns
    ]


    if missing_columns:

        logger.error(
            "Missing deep-learning features: %s",
            missing_columns
        )

        sys.exit(1)


    X = df[feature_columns].copy()

    y = df["traffic_volume"].copy()


    logger.info(
        "Deep-learning feature set prepared: %d features.",
        X.shape[1]
    )


    return X, y, feature_columns


# Train neural network
# ============================================================
def train_neural_network(
    X_train,
    y_train
):

    neural_model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "neural_network",
            MLPRegressor(

                hidden_layer_sizes=(
                    64,
                    32,
                    16
                ),

                activation="relu",

                solver="adam",

                learning_rate_init=0.001,

                max_iter=300,

                early_stopping=True,

                validation_fraction=0.10,

                random_state=42

            )
        )

    ])


    neural_model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Neural network trained successfully."
    )


    return neural_model


# Evaluate neural network
# ============================================================
def evaluate_neural_network(
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
    print("NEURAL NETWORK RESULTS")
    print("========================================")

    print(
        f"MAE : {mae:.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


    logger.info(
        "Neural network evaluation completed: MAE=%.2f, R2=%.4f",
        mae,
        r2
    )


    results = pd.DataFrame([

        {
            "Model": "Neural Network",
            "MAE": mae,
            "R2": r2
        }

    ])


    return results


# Create neural-network training-loss figure
# ============================================================
def create_loss_figure(
    model
):

    try:

        neural_network = (
            model
            .named_steps["neural_network"]
        )


        loss_curve = (
            neural_network.loss_curve_
        )


        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            range(
                1,
                len(loss_curve) + 1
            ),
            loss_curve
        )

        plt.xlabel(
            "Training Iteration"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Neural Network Training Loss"
        )

        plt.tight_layout()


        plt.savefig(
            LOSS_FIGURE_FILE,
            dpi=300
        )

        plt.close()


        logger.info(
            "Neural-network loss figure saved: %s",
            LOSS_FIGURE_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to create neural-network loss figure: %s",
            e
        )

        sys.exit(1)


# SHAP explainability
# ============================================================
def run_shap_explainability(
    X_train,
    y_train,
    X_test,
    feature_columns
):

    logger.info(
        "Training comparable Random Forest model for SHAP explainability."
    )


    # Comparable model trained on same prediction problem
    explainability_model = (
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )


    explainability_model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Comparable Random Forest model trained."
    )


    # Use a sample to keep SHAP computation manageable
    # ========================================================
    sample_size = min(
        1000,
        len(X_test)
    )


    X_shap = (
        X_test
        .sample(
            n=sample_size,
            random_state=42
        )
    )


    logger.info(
        "Calculating SHAP values using %d test records.",
        sample_size
    )


    explainer = shap.TreeExplainer(
        explainability_model
    )


    shap_values = explainer.shap_values(
        X_shap
    )


    # SHAP summary plot
    # ========================================================
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )

    plt.tight_layout()


    plt.savefig(
        SHAP_FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    logger.info(
        "SHAP summary figure saved: %s",
        SHAP_FIGURE_FILE
    )


    # Calculate mean absolute SHAP importance
    # ========================================================
    mean_shap = np.abs(
        shap_values
    ).mean(axis=0)


    importance_df = pd.DataFrame({

        "Feature": feature_columns,

        "Mean_Absolute_SHAP": mean_shap

    })


    importance_df = (

        importance_df

        .sort_values(
            by="Mean_Absolute_SHAP",
            ascending=False
        )

        .reset_index(drop=True)

    )


    importance_df.to_csv(
        SHAP_IMPORTANCE_FILE,
        index=False
    )


    logger.info(
        "SHAP feature importance saved: %s",
        SHAP_IMPORTANCE_FILE
    )


    return importance_df


# Save neural-network results
# ============================================================
def save_results(results):

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        results.to_csv(
            NN_RESULTS_FILE,
            index=False
        )


        logger.info(
            "Neural-network results saved: %s",
            NN_RESULTS_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to save neural-network results: %s",
            e
        )

        sys.exit(1)


# Main program
# ============================================================
def main():

    logger.info(
        "Task 3 deep learning with explainability started."
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Load data
    # ========================================================
    df = load_data()


    # Prepare features
    # ========================================================
    X, y, feature_columns = (
        prepare_features(df)
    )


    print("\n========================================")
    print("TASK 3 - DEEP LEARNING")
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


    print("\nNeural-network architecture:")

    print(
        "Input -> 64 -> 32 -> 16 -> Traffic Volume"
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


    # Train neural network
    # ========================================================
    neural_model = train_neural_network(
        X_train,
        y_train
    )


    # Evaluate neural network
    # ========================================================
    results = evaluate_neural_network(
        neural_model,
        X_test,
        y_test
    )


    # Save results
    # ========================================================
    save_results(
        results
    )


    # Create training-loss figure
    # ========================================================
    create_loss_figure(
        neural_model
    )


    # SHAP explainability
    # ========================================================
    print("\n========================================")
    print("SHAP EXPLAINABILITY")
    print("========================================")

    print(
        "SHAP is applied to a comparable Random Forest model"
    )

    print(
        "trained on the same features and traffic-volume target."
    )


    importance_df = run_shap_explainability(
        X_train,
        y_train,
        X_test,
        feature_columns
    )


    print("\nTop 10 features by mean absolute SHAP value:")

    print(
        importance_df
        .head(10)
        .round(4)
        .to_string(index=False)
    )


    logger.info(
        "Task 3 deep learning with explainability completed successfully."
    )


if __name__ == "__main__":
    main()
    