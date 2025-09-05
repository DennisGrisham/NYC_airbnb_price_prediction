import os, shutil, glob, wandb
api = wandb.Api()
run = wandb.init(project="nyc_airbnb", job_type="materialize_clean")
art = api.artifact("dgrish1-western-governors-university/nyc_airbnb/clean_sample.csv:latest", type="clean_data")
dst = "outputs"
local_dir = art.download(root=dst)
# Find the clean CSV inside the downloaded folder and copy to outputs/clean_sample.csv
candidates = glob.glob(os.path.join(local_dir, "**", "clean_sample.csv"), recursive=True)
if not candidates:
    # sometimes the artifact is a single file at the first level
    candidates = glob.glob(os.path.join(local_dir, "clean_sample.csv"))
if not candidates:
    raise RuntimeError(f"Couldn't find clean_sample.csv inside {local_dir}")
shutil.copyfile(candidates[0], os.path.join(dst, "clean_sample.csv"))
print("Materialized outputs/clean_sample.csv from W&B artifact")
run.finish()
