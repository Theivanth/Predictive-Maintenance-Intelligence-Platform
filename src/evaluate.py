import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            ).tolist(),
    }

    return metrics


def evaluate_models(
    models,
    X_test,
    y_test,
):

    results = {}

    for name, model in models.items():

        results[name] = calculate_metrics(
            model,
            X_test,
            y_test,
        )

    return results


def save_evaluation_report(
    results,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )