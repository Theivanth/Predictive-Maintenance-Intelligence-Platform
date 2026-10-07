from src.data_pipeline import run_pipeline
from src.train_models import (
    train_models,
    save_models,
)
from src.evaluate import (
    evaluate_models,
    save_evaluation_report,
)

from src.config import (
    OUTPUT_DIR,
)


def main():

    print("Running data pipeline...")

    df = run_pipeline()

    print("Training models...")

    (
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_models(df)

    print("Saving models...")

    save_models(models)

    print("Evaluating models...")

    results = evaluate_models(
        models,
        X_test,
        y_test,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_evaluation_report(
        results,
        OUTPUT_DIR
        / "evaluation_report.json"
    )

    print("Training complete.")

    for name, metrics in results.items():

        print(
            f"{name}: "
            f"{metrics['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()