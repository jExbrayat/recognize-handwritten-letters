import mlflow
from sklearn.tree import DecisionTreeClassifier

from src.classification.training.evaluation import evaluate_model
from src.data.load_data import load_sample_train_test
from src.data.preprocessing import preprocess
from mlflow.data.numpy_dataset import from_numpy

# n_jobs = -1
#
#
# def train_decision_tree(x: np.ndarray, y: np.ndarray):
#     dt = DecisionTreeClassifier(random_state=0)
#     dt.fit(x, y)
#

def run_experiment(use_preprocess: bool):
    mlflow.set_experiment("classification")
    run_name = "with_preprocess" if use_preprocess else "without_preprocess"

    with mlflow.start_run(run_name=run_name):
        x_train, x_test, y_train, y_test = load_sample_train_test()
        dataset = from_numpy(x_train, targets=y_train, name="sample_train")

        if use_preprocess:
            x_train = preprocess(x_train)
            x_test = preprocess(x_test)

        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(x_train, y_train)

        accuracy, f1, total_time, fig = evaluate_model(dt, x_train, y_train, x_test, y_test)

        mlflow.log_input(dataset, context="train")

        # Parameters
        mlflow.log_param("use_preprocess", use_preprocess)
        mlflow.log_params(dt.get_params())

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1.mean())
        mlflow.log_metric("eval_time_seconds", total_time)

        # Artifacts
        mlflow.log_dict({"f1_scores": list(f1)}, "f1_scores.json")
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Tags
        mlflow.set_tag("model_name", dt.__class__.__name__)


if __name__ == "__main__":
    run_experiment(use_preprocess=True)
    run_experiment(use_preprocess=False)
