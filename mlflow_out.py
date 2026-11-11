import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

runs = mlflow.search_runs(
    experiment_names=["bi-lstm-crf sroie reproducible"],
    output_format="pandas"
)

runs = runs[
    (runs["tags.comment"] == "No ReLU in style") &
    (runs["params.patience"] == "10")
]

columns = [
    # "run_name",
    # "status",
    # "start_time",
    # "end_time",
    "params.use_features",
    "params.use_style",
    "params.patience",
    "params.seed",
    "tags.comment",
    "metrics.train_loss",
    "metrics.val_f1",
    "metrics.test_f1",
    # add the other columns from your dashboard here
]

runs = runs.sort_values(
    by=["params.use_style", "tags.comment"],
    na_position="last"
)

runs[columns].to_csv("mlflow_runs.csv", index=False)

# runs.to_csv("mlflow_runs.csv", index=False)
