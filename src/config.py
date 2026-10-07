from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "maintenance_data.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "maintenance_clean.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Current",
    "OperatingHours",
]

TARGET = "Status"
MACHINE_ID = "Machine_ID"

STATUS_ORDER = [
    "Healthy",
    "Warning",
    "Critical",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25