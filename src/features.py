import pandas as pd


def create_features(df):
    """Create derived predictive-maintenance features."""

    df = df.copy()

    df["Power"] = df["Voltage"] * df["Current"] \
        if "Voltage" in df.columns else df["Current"] * df["RPM"]

    df["Temperature_Risk"] = (
        df["Temperature"] > 75
    ).astype(int)

    df["High_Vibration"] = (
        df["Vibration"] > 3.5
    ).astype(int)

    return df


def get_feature_columns(df):
    """Return model-ready feature columns."""

    excluded = [
        "Machine_ID",
        "Status",
    ]

    return [
        column
        for column in df.columns
        if column not in excluded
    ]