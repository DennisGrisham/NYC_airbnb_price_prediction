import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import logging
import sys, subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_steps = [
    "download",
    "basic_cleaning",
    "segregate",            # ✅ ADDED: our local split step
    "data_check",
    "data_split",
    "train_random_forest",
    "register_model",       # ✅ Keep in sequence-capable list
    # "test_regression_model"  # intentionally not in default steps
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Special case: run register_model directly if it is the only step
    if len(active_steps) == 1 and active_steps[0].strip() == "register_model":
        logger.info("⚡ Running register_model as a standalone MLproject entry point...")
        mlflow.run(
            ".",
            entry_point="register_model",
            env_manager="local"
        )
        return

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
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

        # ✅ NEW: run our local segregation step (writes outputs/train_data.csv & outputs/test_data.csv)
        if "segregate" in active_steps:
            logger.info("▶ Running local 'segregate' step (src/segregate/run.py)...")
            subprocess.check_call([sys.executable, os.path.join("src", "segregate", "run.py")])
            logger.info("✅ Finished 'segregate' step")

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # Run training step
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

        if "register_model" in active_steps:  # ✅ Still works in sequence
            # ✅ Verify params.json exists before running register_model
            expected_params_path = os.path.join(os.getcwd(), "outputs")
            found_params = False
            for root, dirs, files in os.walk(expected_params_path):
                if "params.json" in files:
                    logger.info(f"✅ Found params.json at: {os.path.join(root, 'params.json')}")
                    found_params = True
                    break
            if not found_params:
                logger.error("❌ params.json not found in outputs directory. Skipping register_model.")
                return

            logger.info("🚀 Starting register_model step...")
            logger.info(f"📂 Using working directory: {os.getcwd()}")

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "register_model"),
                "main",
                env_manager="local",  # ✅ Keep local to avoid rebuilding env
                parameters={}
            )

            logger.info("✅ Finished register_model step")

        if "test_regression_model" in active_steps:
            # Evaluate the latest registered model (or pin a version)
            model_name = config["register_model"]["model_name"]  # nyc_airbnb_random_forest
            # You can choose a fixed version or a stage. Examples:
            # model_uri = f"models:/{model_name}/5"
            # or if you start using stages:
            # model_uri = f"models:/{model_name}/Staging"
            model_uri = f"models:/{model_name}/5"

            test_csv = os.path.join(os.getcwd(), "outputs", "test_data.csv")

            # Hard gates (kept also in script; this lets you override from config if you want)
            min_r2 = 0.50
            max_mae = 40.0

            logger.info("🧪 Evaluating registered model with gating...")
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
                logger.error("❌ Evaluation gates failed — stopping pipeline.")
                sys.exit(rc)
            logger.info("✅ Evaluation gates passed.")


if __name__ == "__main__":
    go()
