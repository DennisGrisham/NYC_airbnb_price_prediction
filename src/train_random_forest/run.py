#!/usr/bin/env python
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import json
import pandas as pd
import numpy as np
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def delta_date_feature(dates):
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

def go(args):

    run = wandb.init(job_type="train_random_forest")
    with open(args.rf_config) as f:
        rf_config = json.load(f)
    run.config.update(rf_config)

    with mlflow.start_run(nested=True):
        # Save run ID for register_model
        current_run_id = mlflow.active_run().info.run_id
        logger.info(f"[DEBUG] Training run_id: {current_run_id}")
        mlflow.log_param("trained_model_run_id", current_run_id)

        # ✅ Always save outputs in the *project root*/outputs folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        project_root_outputs = os.path.join(project_root, "outputs")
        os.makedirs(project_root_outputs, exist_ok=True)
        logger.info(f"[DEBUG] Using outputs directory: {project_root_outputs}")

        # ✅ Save params.json for register_model with trace logging
        params_path = os.path.join(project_root_outputs, "params.json")
        with open(params_path, "w") as f:
            json.dump({"trained_model_run_id": current_run_id}, f)
        logger.info(f"[DEBUG][WRITE] params.json written by train_random_forest/run.py "
                    f"at {datetime.now().isoformat()} to {params_path}")

        # ✅ Save rf_config.json for consistency with trace logging
        rf_config_path = os.path.join(project_root_outputs, "rf_config.json")
        with open(rf_config_path, "w") as f:
            json.dump(rf_config, f)
        logger.info(f"[DEBUG][WRITE] rf_config.json written by train_random_forest/run.py "
                    f"at {datetime.now().isoformat()} to {rf_config_path}")

        # Load training data
        logger.info(f"[DEBUG] Using training artifact: {args.trainval_artifact}")
        trainval_local_path = run.use_artifact(args.trainval_artifact).file()
        logger.info(f"[DEBUG] Local training file path: {trainval_local_path}")
        X = pd.read_csv(trainval_local_path)
        y = X.pop("price")
        logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=args.val_size,
            stratify=X[args.stratify_by],
            random_state=args.random_seed
        )

        logger.info("Preparing sklearn pipeline")
        sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

        logger.info("Fitting model...")
        sk_pipe.fit(X_train, y_train)

        # Metrics
        r_squared = sk_pipe.score(X_val, y_val)
        y_pred = sk_pipe.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        logger.info(f"Score: {r_squared}, MAE: {mae}")
        mlflow.log_metric("r2", r_squared)
        mlflow.log_metric("mae", mae)

        # ✅ Save model inside MLflow's run artifacts directory
        mlflow_model_subdir = f"random_forest_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_model_path = os.path.join(mlflow.get_artifact_uri(), mlflow_model_subdir)

        # Just get the local path version (strip the 'file://' prefix if it exists)
        if mlflow_model_path.startswith("file://"):
            mlflow_model_path = mlflow_model_path.replace("file://", "")

        mlflow.sklearn.save_model(
            sk_pipe,
            path=mlflow_model_path,
            input_example=X_train.iloc[:5]
        )
        logger.info(f"[DEBUG] Saved model to MLflow artifacts dir: {mlflow_model_path}")

        # ✅ W&B logging with correct path
        artifact = wandb.Artifact(
            args.output_artifact,
            type='model_export',
            description='Trained random forest artifact',
            metadata=rf_config
        )
        artifact.add_dir(mlflow_model_path)  # ✅ Corrected this line
        run.log_artifact(artifact)

        fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
        run.summary['r2'] = r_squared
        run.summary['mae'] = mae
        run.log({"feature_importance": wandb.Image(fig_feat_imp)})


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp

def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(binary=False, max_features=max_tfidf_features, stop_words='english'),
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",
    )
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]
    random_forest = RandomForestRegressor(**rf_config)
    sk_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("random_forest", random_forest)])
    return sk_pipe, processed_features

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

