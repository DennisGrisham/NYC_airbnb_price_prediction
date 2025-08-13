import json
import joblib
import pandas as pd
import mlflow
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb


def load_model_from_registry(model_name, stage):
    logged_model = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def load_test_data(test_data_path):
    df = pd.read_parquet(test_data_path)
    X_test = df.drop("price", axis=1)
    y_test = df["price"]
    return X_test, y_test


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return r2, mae, rmse


def go(args):
    with open(args.config) as f:
        config = json.load(f)

    model = load_model_from_registry(config["model_name"], config["stage"])
    X_test, y_test = load_test_data(config["test_data_path"])
    r2, mae, rmse = evaluate(model, X_test, y_test)

    # Log metrics to MLflow
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # Log metrics to W&B
    wandb.init(project=config["wandb_project"], job_type="evaluate_model")
    wandb.log({"r2": r2, "mae": mae, "rmse": rmse})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluate_config.json")
    args = parser.parse_args()
    go(args)
