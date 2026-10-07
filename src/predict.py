import joblib

import pandas as pd

from src.config import MODEL_DIR, FEATURES


def load_model(model_name):

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    return joblib.load(
        MODEL_DIR / filename
    )


def predict_status(
    model,
    sensor_data,
):

    if isinstance(sensor_data, dict):
        X = pd.DataFrame([sensor_data])
    else:
        X = sensor_data.copy()

    missing = [column for column in FEATURES if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = X[FEATURES]

    prediction = model.predict(X)[0]

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(X)[0]

        confidence = probabilities.max()

    else:

        confidence = None

    return prediction, confidence


def create_prediction_input(
    temperature,
    vibration,
    pressure,
    rpm,
    current,
    operating_hours,
):

    return {
        "Temperature": temperature,
        "Vibration": vibration,
        "Pressure": pressure,
        "RPM": rpm,
        "Current": current,
        "OperatingHours": operating_hours,
    }