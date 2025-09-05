import argparse
import os
import pandas as pd
import wandb
from typing import Tuple


def _clean(df: pd.DataFrame, min_price: float, max_price: float) -> pd.DataFrame:
    """
    Basic cleaning for the initial release (no NYC bounds yet).

    - Filter rows with price in [min_price, max_price]
    - Convert last_review to datetime (kept for model/pipeline consistency)
    - Do NOT filter lat/long yet (we add later per rubric)
    """
    df = df.copy()
    # Filter price bounds
    idx = df["price"].between(min_price, max_price)
    df = df.loc[idx].copy()

    # Convert last_review to datetime (ok if NaT)
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    return df


def go(input_artifact: str,
       output_artifact: str,
       output_type: str,
       output_description: str,
       min_price: float,
       max_price: float) -> Tuple[str, int]:
    """
    Download input CSV from W&B, clean, save locally, and log to W&B.

    Parameters
    ----------
    input_artifact : str
        W&B artifact spec for the raw/sample CSV (e.g., "sample.csv:latest")
    output_artifact : str
        Name for the cleaned CSV artifact to create (e.g., "clean_sample.csv")
    output_type : str
        Artifact type (e.g., "clean_data")
    output_description : str
        Short description for the artifact
    min_price : float
        Minimum allowed price
    max_price : float
        Maximum allowed price

    Returns
    -------
    Tuple[str, int]
        (path_to_local_csv, n_rows)
    """

    # --- Force W&B entity/project from environment ---
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "dgrish1-western-governors-university")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "nyc_airbnb")
    WANDB_RUN_GROUP = os.getenv("WANDB_RUN_GROUP", "development")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_RUN_GROUP,
        job_type="basic_cleaning",
        save_code=True
    )

    # Download and read input CSV from W&B
    input_path = run.use_artifact(input_artifact).file()
    df = pd.read_csv(input_path)

    # Clean
    cleaned = _clean(df, min_price=min_price, max_price=max_price)

    # Save locally for downstream steps
    os.makedirs("outputs", exist_ok=True)
    local_csv = os.path.join("outputs", "clean_sample.csv")
    cleaned.to_csv(local_csv, index=False)

    # Log cleaned CSV as a W&B artifact
    art = wandb.Artifact(output_artifact, type=output_type, description=output_description)
    art.add_file(local_csv)
    run.log_artifact(art)
    run.finish()

    return local_csv, len(cleaned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True, help="e.g. 'sample.csv:latest'")
    parser.add_argument("--output_artifact", type=str, required=True, help="e.g. 'clean_sample.csv'")
    parser.add_argument("--output_type", type=str, required=True, help="e.g. 'clean_data'")
    parser.add_argument("--output_description", type=str, required=True, help="Description for the cleaned artifact")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")
    args = parser.parse_args()

    go(args.input_artifact,
       args.output_artifact,
       args.output_type,
       args.output_description,
       args.min_price,
       args.max_price)
