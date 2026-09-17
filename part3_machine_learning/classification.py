import logging
import sys
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
    / "classification_results.csv"
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


    # Common engineered features required for ML
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


    # Make sure all features exist
    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns

    ]


    if missing_columns:

        logger.error(
            "Missing feature columns: %s",
            missing_columns
        )

        sys.exit(1)


    X = df[feature_columns].copy()

    logger.info(
        "Common feature set prepared: %d features.",
        X.shape[1]
    )


    return X, feature_columns


# Evaluate classifier
# ============================================================
def evaluate_classifier(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    print("\n========================================")
    print(model_name)
    print("========================================")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC AUC   : {roc_auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    logger.info(
        "%s evaluation completed.",
        model_name
    )


    return {

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1,
        "ROC_AUC": roc_auc

    }


# Main program
# ============================================================
def main():

    logger.info(
        "Supervised classification started."
    )


    # Load data
    # ========================================================
    df = load_data()


    # Prepare common features
    # ========================================================
    X, feature_columns = prepare_features(df)


    # Classification target
    # ========================================================
    y = df["high_risk"].copy()


    print("\n========================================")
    print("CLASSIFICATION DATA")
    print("========================================")

    print(
        f"Number of records: {len(df)}"
    )

    print(
        f"Number of features: {len(feature_columns)}"
    )

    print(
        f"High-risk records: {int(y.sum())}"
    )

    print(
        f"Non-high-risk records: {int((y == 0).sum())}"
    )

    print(
        f"High-risk percentage: {(y.mean() * 100):.2f}%"
    )


    # Train/test split
    # ========================================================
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    logger.info(
        "Train/test split completed: %d training rows, %d testing rows.",
        X_train.shape[0],
        X_test.shape[0]
    )


    # Model 1: Logistic Regression
    # ========================================================
    logistic_model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )

    ])


    logistic_model.fit(
        X_train,
        y_train
    )


    logger.info(
        "Logistic Regression trained."
    )


    # Model 2: Random Forest Classifier
    # ========================================================
    random_forest_model = (
        RandomForestClassifier(
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
        "Random Forest Classifier trained."
    )


    # Evaluate both models
    # ========================================================
    logistic_results = evaluate_classifier(

        "Logistic Regression",

        logistic_model,

        X_test,

        y_test

    )


    random_forest_results = evaluate_classifier(

        "Random Forest Classifier",

        random_forest_model,

        X_test,

        y_test

    )


    # Compare models
    # ========================================================
    results_df = pd.DataFrame([

        logistic_results,
        random_forest_results

    ])


    print("\n========================================")
    print("CLASSIFICATION MODEL COMPARISON")
    print("========================================")

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )


    # Save comparison table
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
            "Classification results saved: %s",
            RESULTS_FILE
        )

    except Exception as e:

        logger.error(
            "Unable to save classification results: %s",
            e
        )

        sys.exit(1)


    logger.info(
        "Supervised classification completed successfully."
    )


if __name__ == "__main__":
    main()
    