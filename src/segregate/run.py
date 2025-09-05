import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import hydra
from omegaconf import DictConfig

# NEW: wandb import for artifact logging
import wandb

# Force Hydra to run in current working directory
@hydra.main(version_base=None, config_path="../../", config_name="config")
def go(config: DictConfig):

    cfg = config.segregate

    # Start MLflow run (local tracking)
    with mlflow.start_run():

        # --- Load input data from outputs directory ---
        # cfg.input is "clean_sample.csv:latest" -> take file name before colon
        input_file = cfg.input.split(":")[0]
        input_path = os.path.join("outputs", input_file)
        df = pd.read_csv(input_path)

        # --- Setup target and stratify ---
        target = "price"
        stratify_col = df[cfg.stratify_by] if str(cfg.stratify_by).lower() != "none" else None

        # --- Split data ---
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=stratify_col
        )

        # --- Recombine splits ---
        train_data = X_train.copy()
        train_data[target] = y_train
        test_data = X_test.copy()
        test_data[target] = y_test

        # --- Save to outputs ---
        os.makedirs("outputs", exist_ok=True)
        train_path = os.path.join("outputs", cfg.train_output)  # expects trainval_data.csv
        test_path  = os.path.join("outputs", cfg.test_output)   # expects test_data.csv

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # (Also write conventional names for visibility, optional)
        # Keep your existing artifact names if you use them elsewhere:
        mlflow.log_artifact(train_path, artifact_path="segregate")
        mlflow.log_artifact(test_path,  artifact_path="segregate")

        # --- Log artifacts to W&B (so downstream can use_artifact(...)) ---
        # Use existing env for routing; if unset, wandb will still try local/offline.
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            job_type="segregate"
        )

        # trainval_data.csv (this is what train_random_forest expects)
        tv_art = wandb.Artifact(
            name="trainval_data.csv",
            type="clean_data",
            description="Training+validation pre-split dataset (val split happens in training step)"
        )
        tv_art.add_file(train_path)
        run.log_artifact(tv_art)

        # test_data.csv (nice to have for later steps / inspection)
        tst_art = wandb.Artifact(
            name="test_data.csv",
            type="test_data",
            description="Held-out test split"
        )
        tst_art.add_file(test_path)
        run.log_artifact(tst_art)

        run.finish()

if __name__ == "__main__":
    go()

