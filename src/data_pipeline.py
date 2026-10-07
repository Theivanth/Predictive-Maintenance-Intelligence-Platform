from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA,
    PROCESSED_DATA,
    FEATURES,
    TARGET,
    MACHINE_ID,
)


def load_data(path=RAW_DATA):
    """Load the raw maintenance dataset."""
    path = Path(path)

    if not path.exists():
        legacy_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "maintenance_data.csv"
        if legacy_path.exists():
            path = legacy_path

    return pd.read_csv(path)


def validate_data(df):
    """Validate required columns."""
    required_columns = [MACHINE_ID] + FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return True


def handle_missing_values(df):
    """Fill missing numerical values using column medians."""

    df = df.copy()

    for column in FEATURES:
        if df[column].isnull().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    df[TARGET] = df[TARGET].fillna("Healthy")

    return df


def remove_outliers(df):
    """Remove extreme numerical outliers using IQR."""

    df = df.copy()

    for column in FEATURES:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[
            (df[column] >= lower)
            & (df[column] <= upper)
        ]

    return df


def save_processed_data(df, path=PROCESSED_DATA):
    """Save cleaned dataset."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(path, index=False)


def preprocess_data(df):
    """Complete preprocessing pipeline."""

    validate_data(df)

    df = handle_missing_values(df)
    df = remove_outliers(df)

    return df


def run_pipeline():
    """Run the complete data pipeline."""

    df = load_data()

    df = preprocess_data(df)

    save_processed_data(df)

    return df


if __name__ == "__main__":
    run_pipeline()