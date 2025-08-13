import os
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score
import wandb

def main(args):
    # 1) Init W&B
    run = wandb.init(project="nyc_airbnb", job_type="evaluate_registered_model")

    # 2) Resolve model URI from registry
    #   If --version is provided, use that; else use "latest"
    if args.version:
        model_uri = f"models:/{args.model_name}/{args.version}"
        wandb.config.update({"model_version": args.version}, allow_val_change=True)
    else:
        model_uri = f"models:/{args.model_name}/latest"
        wandb.config.update({"model_version": "latest"}, allow_val_change=True)

    print(f"[INFO] Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # 3) Load test data
    test_csv = os.path.abspath(args.test_csv)
    print(f"[INFO] Reading test CSV: {test_csv}")
    df = pd.read_csv(test_csv)

    target = "price"
    X_test = df.drop(columns=[target])
    y_test = df[target]

    # 4) Predict and score
    print("[INFO] Scoring...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[METRICS] R2={r2:.5f}  MAE={mae:.5f}")

    # 5) Log to MLflow + W&B
    with mlflow.start_run(run_name="test_registered_model"):
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_version", args.version if args.version else "latest")
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    wandb.log({"r2": r2, "mae": mae})
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="nyc_airbnb_random_forest")
    parser.add_argument("--version", default=None, help="Specific version to test; omit to use latest")
    parser.add_argument("--test-csv", default="outputs/test_data.csv")
    args = parser.parse_args()
    main(args)
