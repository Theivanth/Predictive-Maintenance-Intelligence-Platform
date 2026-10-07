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


def predict_dataset(model, dataset):
    """Return a copy of a sensor dataset with predicted health and confidence."""
    if dataset.empty:
        raise ValueError("The uploaded dataset contains no rows.")

    missing = [column for column in FEATURES if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    result = dataset.copy()
    sensor_data = result[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid_columns = sensor_data.columns[sensor_data.isna().any()].tolist()
    if invalid_columns:
        raise ValueError(
            "Sensor columns contain missing or non-numeric values: "
            f"{invalid_columns}"
        )

    predictions = model.predict(sensor_data)
    result["Predicted_Status"] = predictions

    if hasattr(model, "predict_proba"):
        result["Confidence"] = model.predict_proba(sensor_data).max(axis=1)

    return result


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