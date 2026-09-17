import logging
import math
import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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
# FastAPI application
# ============================================================

app = FastAPI(

    title="Smart City Traffic Prediction API",

    description=(
        "Deployment simulation for predicting traffic volume "
        "using Random Forest Regressor v2.0."
    ),

    version="2.0"
)


# ============================================================
# Load model safely
# ============================================================

model = None
FEATURE_COLUMNS = []


def load_model():

    global model
    global FEATURE_COLUMNS

    try:

        if not MODEL_FILE.exists():

            logger.error(
                "Model file does not exist: %s",
                MODEL_FILE
            )

            return


        if not FEATURE_FILE.exists():

            logger.error(
                "Feature file does not exist: %s",
                FEATURE_FILE
            )

            return


        model = joblib.load(
            MODEL_FILE
        )


        with open(
            FEATURE_FILE,
            "r"
        ) as file:

            FEATURE_COLUMNS = [

                line.strip()

                for line in file

                if line.strip()

            ]


        logger.info(
            "Random Forest v2.0 model loaded successfully."
        )

        logger.info(
            "Deployment model expects %d features.",
            len(FEATURE_COLUMNS)
        )


    except Exception as e:

        logger.error(
            "Unable to load deployment model: %s",
            e
        )

        model = None
        FEATURE_COLUMNS = []


# Load when API module starts
load_model()


# ============================================================
# API input structure
# ============================================================

class TrafficInput(BaseModel):

    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day from 0 to 23."
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description=(
            "Monday=0, Tuesday=1, Wednesday=2, "
            "Thursday=3, Friday=4, Saturday=5, Sunday=6."
        )
    )

    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = weekday, 1 = weekend."
    )

    is_holiday: int = Field(
        0,
        ge=0,
        le=1,
        description="0 = normal day, 1 = holiday."
    )

    temp_scaled: float = Field(
        ...,
        description="Scaled temperature value."
    )

    rain_scaled: float = Field(
        0.0,
        description="Scaled rainfall value."
    )

    snow_1h: float = Field(
        0.0,
        ge=0,
        description="Snowfall during previous hour."
    )

    clouds_all: float = Field(
        ...,
        ge=0,
        le=100,
        description="Cloud coverage percentage."
    )

    weather: str = Field(
        "Clear",
        description=(
            "Main weather category such as "
            "Clear, Clouds, Rain, Mist or Snow."
        )
    )


# ============================================================
# Home endpoint
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Smart City Traffic Prediction API is running.",

        "model":
            "Random Forest Regressor",

        "model_version":
            "v2.0"

    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    if model is not None:

        return {

            "status":
                "PASS",

            "model_loaded":
                True,

            "model_version":
                "v2.0",

            "number_of_features":
                len(FEATURE_COLUMNS)

        }


    return {

        "status":
            "ALERT",

        "model_loaded":
            False,

        "model_version":
            "v2.0"

    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict_traffic(
    traffic: TrafficInput
):

    # Make sure model is available
    # ========================================================

    if model is None:

        logger.error(
            "Prediction requested but model is unavailable."
        )

        raise HTTPException(

            status_code=503,

            detail="Traffic prediction model is not available."

        )


    # Start every feature at zero
    # ========================================================

    input_data = {

        feature: 0.0

        for feature in FEATURE_COLUMNS

    }


    # Basic time variables
    # ========================================================

    input_data["hour"] = (
        traffic.hour
    )

    input_data["day_of_week"] = (
        traffic.day_of_week
    )

    input_data["is_weekend"] = (
        traffic.is_weekend
    )

    input_data["is_holiday"] = (
        traffic.is_holiday
    )


    # Cyclical encoding for hour
    # ========================================================

    input_data["hour_sin"] = math.sin(

        2
        * math.pi
        * traffic.hour
        / 24

    )


    input_data["hour_cos"] = math.cos(

        2
        * math.pi
        * traffic.hour
        / 24

    )


    # Cyclical encoding for weekday
    # ========================================================

    input_data["day_sin"] = math.sin(

        2
        * math.pi
        * traffic.day_of_week
        / 7

    )


    input_data["day_cos"] = math.cos(

        2
        * math.pi
        * traffic.day_of_week
        / 7

    )


    # Weather numeric variables
    # ========================================================

    input_data["temp_scaled"] = (
        traffic.temp_scaled
    )

    input_data["rain_scaled"] = (
        traffic.rain_scaled
    )

    input_data["snow_1h"] = (
        traffic.snow_1h
    )

    input_data["clouds_all"] = (
        traffic.clouds_all
    )


    # Weather one-hot variable
    # ========================================================

    weather_feature = (
        "weather_"
        + traffic.weather
    )


    if weather_feature in input_data:

        input_data[
            weather_feature
        ] = 1.0


    else:

        logger.warning(
            "Weather category '%s' does not match "
            "a trained weather feature.",
            traffic.weather
        )


    # Build dataframe using exact training order
    # ========================================================

    input_df = pd.DataFrame(

        [input_data],

        columns=FEATURE_COLUMNS

    )


    # Predict traffic volume
    # ========================================================

    prediction = model.predict(
        input_df
    )[0]


    prediction = max(
        0,
        float(prediction)
    )


    logger.info(
        "Traffic prediction generated: %.2f vehicles.",
        prediction
    )


    # Simple interpretation
    # ========================================================

    if prediction < 2000:

        traffic_level = (
            "Low"
        )


    elif prediction < 4000:

        traffic_level = (
            "Moderate"
        )


    elif prediction < 5500:

        traffic_level = (
            "High"
        )


    else:

        traffic_level = (
            "Very High"
        )


    # API response
    # ========================================================

    return {

        "predicted_traffic_volume":
            round(
                prediction,
                2
            ),

        "traffic_level":
            traffic_level,

        "model":
            "Random Forest Regressor",

        "model_version":
            "v2.0"

    }