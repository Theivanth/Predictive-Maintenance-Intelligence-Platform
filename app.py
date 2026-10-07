import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from src.config import (
    PROCESSED_DATA,
    MODEL_DIR,
    FEATURES,
)

from src.predict import predict_status
from src.alerts import generate_alert
from src.train_models import train_models, save_models, DEFAULT_MODEL_PARAMS


def load_dataset():

    return pd.read_csv(
        PROCESSED_DATA
    )


def load_model(model_name="Random Forest"):

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    return joblib.load(
        MODEL_DIR / filename
    )


def show_header():

    st.title(
        "Predictive Maintenance Intelligence Platform"
    )

    st.caption(
        "Machine health monitoring and predictive maintenance"
    )


def show_overview(df):

    st.header("Overview")

    total = len(df)

    healthy = (
        df["Status"] == "Healthy"
    ).sum()

    warning = (
        df["Status"] == "Warning"
    ).sum()

    critical = (
        df["Status"] == "Critical"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Machines",
        total
    )

    col2.metric(
        "Healthy",
        healthy
    )

    col3.metric(
        "Warning",
        warning
    )

    col4.metric(
        "Critical",
        critical
    )

    st.subheader(
        "Machine Status Distribution"
    )

    st.bar_chart(
        df["Status"].value_counts()
    )


def show_machine_monitor(df):

    st.header("Machine Monitor")

    machine = st.selectbox(
        "Select Machine",
        df["Machine_ID"].unique()
    )

    row = df[
        df["Machine_ID"] == machine
    ].iloc[0]

    st.subheader(
        f"Machine {machine}"
    )

    cols = st.columns(3)

    for index, feature in enumerate(FEATURES):

        cols[index % 3].metric(
            feature,
            row[feature]
        )

    st.write(
        "Current Status:",
        row["Status"]
    )


def get_model_tuning_controls():

    st.sidebar.header("Model Parameters")

    tuning = {
        "Random Forest": {
            "n_estimators": st.sidebar.slider(
                "Random Forest estimators",
                min_value=50,
                max_value=500,
                value=DEFAULT_MODEL_PARAMS["Random Forest"]["n_estimators"],
                step=50,
            ),
            "max_depth": st.sidebar.slider(
                "Random Forest max depth",
                min_value=2,
                max_value=20,
                value=DEFAULT_MODEL_PARAMS["Random Forest"]["max_depth"],
            ),
            "min_samples_leaf": st.sidebar.slider(
                "Random Forest min samples leaf",
                min_value=1,
                max_value=10,
                value=DEFAULT_MODEL_PARAMS["Random Forest"]["min_samples_leaf"],
            ),
        },
        "KNN": {
            "n_neighbors": st.sidebar.slider(
                "KNN neighbors",
                min_value=1,
                max_value=25,
                value=DEFAULT_MODEL_PARAMS["KNN"]["n_neighbors"],
            ),
        },
        "Logistic Regression": {
            "C": st.sidebar.slider(
                "Logistic Regression C",
                min_value=0.1,
                max_value=10.0,
                value=DEFAULT_MODEL_PARAMS["Logistic Regression"]["C"],
                step=0.1,
            ),
        },
        "Decision Tree": {
            "max_depth": st.sidebar.slider(
                "Decision Tree max depth",
                min_value=2,
                max_value=20,
                value=DEFAULT_MODEL_PARAMS["Decision Tree"]["max_depth"],
            ),
            "min_samples_leaf": st.sidebar.slider(
                "Decision Tree min samples leaf",
                min_value=1,
                max_value=10,
                value=DEFAULT_MODEL_PARAMS["Decision Tree"]["min_samples_leaf"],
            ),
        },
    }

    return tuning


def show_prediction():

    st.header("Prediction")

    selected_model = st.selectbox(
        "Model",
        [
            "Random Forest",
            "KNN",
            "Logistic Regression",
            "Decision Tree",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "Temperature",
            value=70.0
        )

        vibration = st.number_input(
            "Vibration",
            value=3.0
        )

        pressure = st.number_input(
            "Pressure",
            value=5.5
        )

    with col2:

        rpm = st.number_input(
            "RPM",
            value=1500
        )

        current = st.number_input(
            "Current",
            value=9.0
        )

        operating_hours = st.number_input(
            "Operating Hours",
            value=2000
        )

    if st.button(
        "PREDICT",
        type="primary"
    ):

        model = load_model(selected_model)

        input_data = pd.DataFrame([{
            "Temperature": temperature,
            "Vibration": vibration,
            "Pressure": pressure,
            "RPM": rpm,
            "Current": current,
            "OperatingHours": operating_hours,
        }])

        status, confidence = predict_status(
            model,
            input_data
        )

        alert = generate_alert(status)

        st.subheader(
            f"Prediction: {status}"
        )

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

        st.info(
            alert["message"]
        )


def show_model_performance():

    st.header("Model Performance")

    report_path = (
        Path("outputs")
        / "evaluation_report.json"
    )

    if not report_path.exists():

        st.warning(
            "Train the models first."
        )

        return

    st.json(
        report_path.read_text()
    )


def main():

    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="⚙️",
        layout="wide",
    )

    model_params = get_model_tuning_controls()

    if st.sidebar.button("Retrain Models"):
        with st.spinner("Training models with updated settings..."):
            trained_models, _, _, _, _ = train_models(load_dataset(), model_params=model_params)
            save_models(trained_models)
        st.sidebar.success("Models retrained successfully.")

    show_header()

    df = load_dataset()

    tabs = st.tabs([
        "Overview",
        "Machine Monitor",
        "Prediction",
        "Model Performance",
    ])

    with tabs[0]:
        show_overview(df)

    with tabs[1]:
        show_machine_monitor(df)

    with tabs[2]:
        show_prediction()

    with tabs[3]:
        show_model_performance()


if __name__ == "__main__":
    main()