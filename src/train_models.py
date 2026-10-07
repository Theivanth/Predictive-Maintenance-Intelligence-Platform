import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_DIR,
)


DEFAULT_MODEL_PARAMS = {
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 1,
    },
    "KNN": {
        "n_neighbors": 5,
    },
    "Logistic Regression": {
        "C": 1.0,
    },
    "Decision Tree": {
        "max_depth": 10,
        "min_samples_leaf": 1,
    },
}


def _resolve_model_params(model_params=None):
    resolved = {
        name: dict(values)
        for name, values in DEFAULT_MODEL_PARAMS.items()
    }

    if model_params is None:
        return resolved

    for model_name, overrides in model_params.items():
        if model_name in resolved and isinstance(overrides, dict):
            resolved[model_name].update(overrides)

    return resolved


def split_data(df):

    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models(model_params=None):

    resolved_params = _resolve_model_params(model_params)

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=resolved_params["Random Forest"]["n_estimators"],
                max_depth=resolved_params["Random Forest"]["max_depth"],
                min_samples_leaf=resolved_params["Random Forest"]["min_samples_leaf"],
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),

        "KNN":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=resolved_params["KNN"]["n_neighbors"],
                        weights="distance",
                    )
                ),
            ]),

        "Logistic Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=resolved_params["Logistic Regression"]["C"],
                        max_iter=2000,
                        class_weight="balanced",
                    )
                ),
            ]),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=resolved_params["Decision Tree"]["max_depth"],
                min_samples_leaf=resolved_params["Decision Tree"]["min_samples_leaf"],
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
    }

    return models


def train_models(df, model_params=None):

    X_train, X_test, y_train, y_test = split_data(df)

    models = build_models(model_params=model_params)

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

    return (
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_models(models):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_paths = []

    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        path = MODEL_DIR / filename
        joblib.dump(model, path)
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline

    df = run_pipeline()
    trained_models, _, _, _, _ = train_models(df)
    saved_files = save_models(trained_models)
    print(f"Saved {len(saved_files)} models to {MODEL_DIR}")