import mlflow
import logging
import hydra
import os
import json
from omegaconf import DictConfig
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../", config_name="config")
def go(config: DictConfig):
    logger.info("🚀 Starting register_model step...")

    # Determine project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    outputs_dir = os.path.join(project_root, "outputs")
    logger.info(f"[DEBUG] Project root resolved as: {project_root}")
    logger.info(f"[DEBUG] Looking for outputs directory at: {outputs_dir}")

    if not os.path.exists(outputs_dir):
        logger.error(f"❌ Outputs directory not found at {outputs_dir}")
        logger.info("🔍 Dumping top-level tree of project root for inspection:")
        for root, dirs, files in os.walk(project_root):
            logger.info(f"DIR: {root}")
            for f in files:
                logger.info(f"  FILE: {f}")
        return

    params_path = os.path.join(outputs_dir, "params.json")
    logger.info(f"[DEBUG] Expecting params.json at: {params_path}")

    if os.path.exists(params_path):
        logger.info("✅ Found params.json! Printing contents:")
        with open(params_path, "r") as f:
            logger.info(f.read())
    else:
        logger.error("❌ params.json NOT FOUND. Dumping outputs directory contents:")
        for item in os.listdir(outputs_dir):
            logger.info(f"  {item}")
        return

    rf_config_path = os.path.join(outputs_dir, "rf_config.json")
    if not os.path.exists(rf_config_path):
        logger.error(f"❌ rf_config.json not found at {rf_config_path}")
        return

    # Load params
    with open(params_path, "r") as f:
        trained_model_run_id = json.load(f).get("trained_model_run_id")

    if not trained_model_run_id:
        logger.error("❌ trained_model_run_id missing in params.json.")
        return

    logger.info(f"[DEBUG] Using trained_model_run_id: {trained_model_run_id}")

    model_name = config.register_model.model_name
    client = MlflowClient()

    # Check if already registered
    try:
        versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version))
            if latest_version.run_id == trained_model_run_id:
                logger.info("⏩ Skipped — same run ID as latest version")
                return
    except Exception as e:
        logger.warning(f"Could not fetch latest model version info: {e}")

    # Locate artifact folder
    artifacts = client.list_artifacts(trained_model_run_id)
    model_folder_candidates = [a.path for a in artifacts if a.path.startswith("random_forest_model_")]
    if not model_folder_candidates:
        logger.error("❌ No model artifact folder found for run.")
        return

    model_folder = model_folder_candidates[0]
    model_uri = f"runs:/{trained_model_run_id}/{model_folder}"
    logger.info(f"[DEBUG] Model URI to register: {model_uri}")

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    result = client.create_model_version(name=model_name, source=model_uri, run_id=trained_model_run_id)
    logger.info(f"✅ Registered new model version: {result.version}")

    # NEW: emit model uri to file + stdout so downstream steps can read it
    try:
        os.makedirs(outputs_dir, exist_ok=True)
        model_uri_registry = f"models:/{model_name}/{result.version}"
        payload = {
            "model_name": model_name,
            "version": int(result.version),
            "run_id": trained_model_run_id,
            "model_uri_runs": model_uri,               # runs:/.../random_forest_model_...
            "model_uri": model_uri_registry            # models:/name/version   <-- use this for eval
        }
        out_json = os.path.join(outputs_dir, "last_registered_model.json")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"📝 Wrote {out_json}")
        # Also print a simple line parser-friendly:
        print(f"MODEL_URI={model_uri_registry}")
    except Exception as e:
        logger.warning(f"Could not write last_registered_model.json: {e}")


if __name__ == "__main__":
    go()
