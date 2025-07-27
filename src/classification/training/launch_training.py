import mlflow
import typer
from mlflow.data.numpy_dataset import from_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.classification.training.evaluation import evaluate_model
from src.data.load_data import get_x_y, load_full_data, load_sample_train_test
from src.data.preprocessing import preprocess

from joblib import Parallel, delayed
from itertools import product


def get_models() -> tuple[Pipeline, ...]:
    return (
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        LogisticRegression(),
    )


def run_experiment(model: Pipeline, use_preprocess: bool, use_sample_data: bool):
    mlflow.set_experiment("classification")
    run_name = "with_preprocess" if use_preprocess else "without_preprocess"

    with mlflow.start_run(run_name=run_name):
        if use_sample_data:
            x_train, x_test, y_train, y_test = load_sample_train_test()
            dataset = from_numpy(x_train, targets=y_train, name="sample_train")
        else:
            data = load_full_data()
            x, y = get_x_y(data)

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, stratify=y, random_state=0, test_size=0.1
            )
            dataset = from_numpy(x_train, targets=y_train, name="full_train")

        if use_preprocess:
            x_train = preprocess(x_train)
            x_test = preprocess(x_test)

        model.fit(x_train, y_train)

        accuracy, f1, total_time, fig = evaluate_model(
            model, x_train, y_train, x_test, y_test
        )

        mlflow.log_input(dataset, context="train")

        # Parameters
        mlflow.log_param("use_preprocess", use_preprocess)
        mlflow.log_params(model.get_params())

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1.mean())
        mlflow.log_metric("eval_time_seconds", total_time)

        # Artifacts
        mlflow.log_dict({"f1_scores": list(f1)}, "f1_scores.json")
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Tags
        mlflow.set_tag("model_name", model.__class__.__name__)


def main(use_sample_data: bool = True) -> None:
    for model in get_models():
        run_experiment(model, use_preprocess=True, use_sample_data=use_sample_data)
        run_experiment(model, use_preprocess=False, use_sample_data=use_sample_data)


if __name__ == "__main__":
    typer.run(main)
