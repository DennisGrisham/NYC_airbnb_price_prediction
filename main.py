import json
import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig
import logging
import sys, subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_steps = [
    "download",
    "basic_cleaning",
    "segregate",            # local split step
    "data_check",
    "train_random_forest",
    "register_model",
    # "test_regression_model"
]

@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):

    # --- Always use a repo-local MLflow store & experiment (first-run safe) ---
    # Hydra changes the working dir; use the original project root:
    proj_root = hydra.utils.get_original_cwd()
    mlruns_dir = os.path.join(proj_root, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    # Ensure the experiment exists and is selected
    mlflow.set_experiment(str(config["main"]["experiment_name"]))
    # --------------------------------------------------------------------------

    # --- WandB wiring (read from config['wandb'] if available) ---
    wandb_entity = None
    wandb_project = None
    try:
        wandb_entity  = config["wandb"]["entity"]
        wandb_project = config["wandb"]["project"]
    except Exception:
        pass  # fall back to 'main' section if missing

    if wandb_entity:
        os.environ["WANDB_ENTITY"] = str(wandb_entity)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = str(wandb_project)
    os.environ["WANDB_RUN_GROUP"] = str(config["main"]["experiment_name"])
    # --------------------------------------------------------------

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Special case: register_model alone
    if len(active_steps) == 1 and active_steps[0].strip() == "register_model":
        logger.info("Running register_model as a standalone MLproject entry point...")
        mlflow.run(".", entry_point="register_model", env_manager="local")
        return

    if "download" in active_steps:
        _ = mlflow.run(
            f"{config['main']['components_repository']}/get_data",
            "main",
            version="main",
            env_manager="conda",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded",
            },
        )

    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_data",
                "output_description": "Data with outliers and nulls removed",
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # Ensure segregate sees outputs/clean_sample.csv and run from project root
    if "segregate" in active_steps:
        proj_root = hydra.utils.get_original_cwd()

        # First try the W&B artifact materialization path used by components
        src_art_1 = os.path.join(proj_root, "artifacts", "clean_sample.csv:latest", "clean_sample.csv")
        # Fallback: some templates also leave a local copy in the component's outputs/
        src_art_2 = os.path.join(proj_root, "src", "basic_cleaning", "outputs", "clean_sample.csv")

        dst_csv = os.path.join(proj_root, "outputs", "clean_sample.csv")
        os.makedirs(os.path.dirname(dst_csv), exist_ok=True)

        if os.path.exists(src_art_1):
            shutil.copy2(src_art_1, dst_csv)
            logger.info(f"Copied cleaned CSV from artifacts: {src_art_1} -> {dst_csv}")
        elif os.path.exists(src_art_2):
            shutil.copy2(src_art_2, dst_csv)
            logger.info(f"Copied cleaned CSV from component outputs: {src_art_2} -> {dst_csv}")
        else:
            logger.warning("Could not find clean_sample.csv in artifacts/ or src/basic_cleaning/outputs/. "
                           "segregate will still try if file already exists in outputs/.")

        logger.info("Running local 'segregate' step (src/segregate/run.py)...")
        subprocess.check_call([sys.executable, os.path.join("src", "segregate", "run.py")],
                              cwd=proj_root)
        logger.info("Finished 'segregate' step")

    if "data_check" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
            "main",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "sample.csv:latest",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    if "train_random_forest" in active_steps:
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
            "main",
            parameters={
                "trainval_artifact": config["data"]["trainval_artifact"],
                "val_size": config["data"]["val_size"],
                "random_seed": config["main"]["random_seed"],
                "stratify_by": config["data"]["stratify_by"],
                "rf_config": rf_config,
                "max_tfidf_features": config["max_tfidf_features"],
                "output_artifact": config["output_artifact"],
            },
        )

    if "register_model" in active_steps:
        expected_params_path = os.path.join(os.getcwd(), "outputs")
        found_params = any(
            "params.json" in files for _, _, files in os.walk(expected_params_path)
        )
        if not found_params:
            logger.error("params.json not found in outputs directory. Skipping register_model.")
            return

        logger.info("Starting register_model step...")
        logger.info(f"Using working directory: {os.getcwd()}")
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "register_model"),
            "main",
            env_manager="local",
            parameters={},
        )
        logger.info("Finished register_model step")

    if "test_regression_model" in active_steps:
        # Prefer the pinned version that was just registered; fallback to 'Staging'
        model_name = config["register_model"]["model_name"]
        model_uri = None
        try:
            with open(os.path.join(os.getcwd(), "outputs", "last_registered_model.json")) as f:
                data = json.load(f)
                model_uri = data.get("model_uri") or data.get("MODEL_URI")
        except Exception:
            pass
        if not model_uri:
            model_uri = f"models:/{model_name}/Staging"

        test_csv = os.path.join(os.getcwd(), "outputs", "test_data.csv")
        min_r2 = 0.50
        max_mae = 40.0

        logger.info("Evaluating registered model with gating...")
        cmd = [
            sys.executable, "scripts/eval_gate.py",
            "--model-uri", model_uri,
            "--test-csv", test_csv,
            "--target-col", "price",
            "--min-r2", str(min_r2),
            "--max-mae", str(max_mae),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            logger.error("Evaluation gates failed — stopping pipeline.")
            sys.exit(rc)
        logger.info("Evaluation gates passed.")

if __name__ == "__main__":
    go()

