import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import hydra
from omegaconf import DictConfig

# Force Hydra to run in current working directory
@hydra.main(version_base=None, config_path="../../", config_name="config")
def go(config: DictConfig):

    cfg = config.segregate

    # Start MLflow run
    with mlflow.start_run():

        # Load input data from outputs directory
        input_file = cfg.input.split(":")[0]
        input_path = os.path.join("outputs", input_file)
        df = pd.read_csv(input_path)

        # Setup target and stratify
        target = "price"
        stratify_col = df[cfg.stratify_by] if cfg.stratify_by.lower() != "none" else None

        # Split data
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=stratify_col
        )

        # Recombine splits
        train_data = X_train.copy()
        train_data[target] = y_train
        test_data = X_test.copy()
        test_data[target] = y_test

        # Save to outputs
        os.makedirs("outputs", exist_ok=True)
        train_path = os.path.join("outputs", cfg.train_output)
        test_path = os.path.join("outputs", cfg.test_output)

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # Log files as artifacts
        mlflow.log_artifact(train_path, artifact_path="segregate")
        mlflow.log_artifact(test_path, artifact_path="segregate")

if __name__ == "__main__":
    go()
