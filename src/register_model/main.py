import argparse
import logging
import os
import shutil
import json
from datetime import datetime

import mlflow
import pandas as pd
import numpy as np
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from mlflow import set_experiment


def delta_date_feature(dates):
    """Calculate days between each date and the most recent date in the column."""
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    set_experiment("train_random_forest")

    with mlflow.start_run(nested=True):
        # Get and log run ID to MLflow
        current_run_id = mlflow.active_run().info.run_id
        logger.info(f"[DEBUG] Training run_id: {current_run_id}")
        mlflow.log_param("trained_model_run_id", current_run_id)

        # Also store it in params.json in outputs folder
        outputs_dir = os.path.join(os.getcwd(), "outputs", datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H-%M-%S"))
        os.makedirs(outputs_dir, exist_ok=True)
        params_path = os.path.join(outputs_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump({"trained_model_run_id": current_run_id}, f)
        logger.info(f"[DEBUG] Saved trained_model_run_id to {params_path}")

        # === Load your config ===
        rf_config_path = args.rf_config
        with open(rf_config_path, "r") as f:
            rf_config = json.load(f)

        # === Load training data ===
        logger.info("Loading training data...")
        # NOTE: You'll need your actual artifact loading code here
        # Placeholder for example:
        df = pd.read_csv("path_to_clean_sample.csv")  # Replace with real artifact fetching

        X = df.drop("price", axis=1)
        y = df["price"]

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.val_size, random_state=args.random_seed, stratify=X[args.stratify_by]
        )

        # Pipeline example
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["neighbourhood_group", "room_type"]),
            ("num", SimpleImputer(strategy="median"), ["minimum_nights", "number_of_reviews"])
        ])

        model = RandomForestRegressor(**rf_config)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        logger.info("Fitting model...")
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        r2 = pipeline.score(X_val, y_val)
        mae = mean_absolute_error(y_val, preds)

        logger.info(f"Score: {r2}, MAE: {mae}")

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save model to artifacts
        model_folder_name = f"random_forest_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_artifact_path = os.path.join("artifacts", model_folder_name)
        mlflow.sklearn.log_model(pipeline, artifact_path=model_folder_name)

        logger.info(f"[DEBUG] Model artifact folder confirmed: {model_artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)

    args = parser.parse_args()

    go(args)
