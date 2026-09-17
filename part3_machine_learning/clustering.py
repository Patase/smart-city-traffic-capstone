import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

CLUSTERED_DATA_FILE = (
    RESULTS_DIR
    / "kmeans_clustered_data.csv"
)

CLUSTER_SUMMARY_FILE = (
    RESULTS_DIR
    / "kmeans_cluster_summary.csv"
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


# Create weather severity variable
# ============================================================
def create_weather_severity(df):

    df = df.copy()

    # Default = normal weather
    df["weather_severity"] = 0

    # Low-visibility weather
    df.loc[
        df["is_low_visibility"] == 1,
        "weather_severity"
    ] = 1

    # Severe weather
    df.loc[
        df["is_severe_weather"] == 1,
        "weather_severity"
    ] = 2

    logger.info(
        "Weather severity variable created."
    )

    return df


# Prepare clustering features
# ============================================================
def prepare_features(df):

    feature_columns = [
        "hour",
        "weather_severity",
        "traffic_volume"
    ]

    missing_columns = [

        column
        for column in feature_columns

        if column not in df.columns
    ]

    if missing_columns:

        logger.error(
            "Missing clustering features: %s",
            missing_columns
        )

        sys.exit(1)

    X = df[feature_columns].copy()

    # Scale features so traffic volume does not dominate
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    logger.info(
        "Clustering features prepared and standardised."
    )

    return X, X_scaled, feature_columns


# Run K-means clustering
# ============================================================
def run_kmeans(df, X_scaled):

    number_of_clusters = 4

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(
        X_scaled
    )

    df = df.copy()

    df["cluster"] = cluster_labels

    logger.info(
        "K-means clustering completed with %d clusters.",
        number_of_clusters
    )

    return df, model


# Create cluster summary
# ============================================================
def create_cluster_summary(df):

    summary = (

        df.groupby("cluster")
        .agg(

            records=(
                "cluster",
                "size"
            ),

            average_hour=(
                "hour",
                "mean"
            ),

            average_weather_severity=(
                "weather_severity",
                "mean"
            ),

            average_traffic_volume=(
                "traffic_volume",
                "mean"
            )

        )
        .reset_index()
    )

    summary["percentage"] = (

        summary["records"]

        / len(df)

        * 100
    )

    summary = summary[
        [
            "cluster",
            "records",
            "percentage",
            "average_hour",
            "average_weather_severity",
            "average_traffic_volume"
        ]
    ]

    logger.info(
        "Cluster summary created."
    )

    return summary


# Create K-means visualisations
# ============================================================
def create_visualisations(df):

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # Figure 1: Hour vs traffic volume
        # ====================================================
        plt.figure(
            figsize=(10, 6)
        )

        scatter = plt.scatter(
            df["hour"],
            df["traffic_volume"],
            c=df["cluster"],
            alpha=0.35,
            s=12
        )

        plt.xlabel(
            "Hour of Day"
        )

        plt.ylabel(
            "Traffic Volume"
        )

        plt.title(
            "K-means Clusters: Hour vs Traffic Volume"
        )

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        plt.tight_layout()


        hour_traffic_file = (
            RESULTS_DIR
            / "kmeans_hour_traffic.png"
        )

        plt.savefig(
            hour_traffic_file,
            dpi=300
        )

        plt.close()


        logger.info(
            "K-means hour vs traffic figure saved: %s",
            hour_traffic_file
        )


        # Figure 2: Weather severity vs traffic volume
        # ====================================================
        plt.figure(
            figsize=(8, 6)
        )

        scatter = plt.scatter(
            df["weather_severity"],
            df["traffic_volume"],
            c=df["cluster"],
            alpha=0.35,
            s=12
        )

        plt.xlabel(
            "Weather Severity"
        )

        plt.ylabel(
            "Traffic Volume"
        )

        plt.title(
            "K-means Clusters: Weather Severity vs Traffic Volume"
        )

        plt.xticks(
            [0, 1, 2],
            [
                "Normal",
                "Low visibility",
                "Severe"
            ]
        )

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        plt.tight_layout()


        weather_traffic_file = (
            RESULTS_DIR
            / "kmeans_weather_traffic.png"
        )

        plt.savefig(
            weather_traffic_file,
            dpi=300
        )

        plt.close()


        logger.info(
            "K-means weather vs traffic figure saved: %s",
            weather_traffic_file
        )


    except Exception as e:

        logger.error(
            "Unable to create K-means visualisations: %s",
            e
        )

        sys.exit(1)


# Save results
# ============================================================
def save_results(
    df,
    summary
):

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            CLUSTERED_DATA_FILE,
            index=False
        )

        summary.to_csv(
            CLUSTER_SUMMARY_FILE,
            index=False
        )

        logger.info(
            "Clustered dataset saved: %s",
            CLUSTERED_DATA_FILE
        )

        logger.info(
            "Cluster summary saved: %s",
            CLUSTER_SUMMARY_FILE
        )


    except Exception as e:

        logger.error(
            "Unable to save clustering results: %s",
            e
        )

        sys.exit(1)


# Main program
# ============================================================
def main():

    logger.info(
        "K-means clustering started."
    )


    # Load dataset
    # ========================================================
    df = load_data()


    # Create weather severity
    # ========================================================
    df = create_weather_severity(
        df
    )


    print("\n========================================")
    print("WEATHER SEVERITY")
    print("========================================")

    print(
        "\n0 = Normal weather"
    )

    print(
        "1 = Low-visibility weather"
    )

    print(
        "2 = Severe weather"
    )


    print("\nWeather severity distribution:")

    print(
        df["weather_severity"]
        .value_counts()
        .sort_index()
    )


    # Prepare features
    # ========================================================
    X, X_scaled, feature_columns = (
        prepare_features(df)
    )


    print("\n========================================")
    print("K-MEANS CLUSTERING")
    print("========================================")

    print(
        f"Number of records: {len(df)}"
    )

    print(
        f"Features used: {feature_columns}"
    )

    print(
        "Number of clusters: 4"
    )


    # Run K-means
    # ========================================================
    df, model = run_kmeans(
        df,
        X_scaled
    )


    # Create summary
    # ========================================================
    summary = create_cluster_summary(
        df
    )


    print("\n========================================")
    print("CLUSTER SUMMARY")
    print("========================================")

    print(
        summary
        .round(2)
        .to_string(index=False)
    )


    # Create visualisations
    # ========================================================
    create_visualisations(
        df
    )


    # Save results
    # ========================================================
    save_results(
        df,
        summary
    )


    logger.info(
        "K-means clustering completed successfully."
    )


if __name__ == "__main__":
    main()