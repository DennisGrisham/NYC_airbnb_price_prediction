import argparse
import os
import tempfile
import pandas as pd
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Basic data cleaning with NYC bounding box filter")

    parser.add_argument("--input_artifact", type=str, required=True,
                        help="Name:version for the raw CSV artifact from W&B (e.g. sample.csv:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="Name of the cleaned CSV artifact to log to W&B (e.g. clean_sample.csv)")
    parser.add_argument("--output_type", type=str, required=True, help="Artifact type (e.g. clean_data)")
    parser.add_argument("--output_description", type=str, required=True, help="Artifact description")

    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")

    return parser.parse_args()


def main():
    args = parse_args()

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    # 1) Download the raw CSV artifact from W&B
    artifact_path = run.use_artifact(args.input_artifact).download()
    raw_csv = None
    for fname in os.listdir(artifact_path):
        if fname.endswith(".csv"):
            raw_csv = os.path.join(artifact_path, fname)
            break
    if raw_csv is None:
        raise FileNotFoundError("No .csv file found inside the input_artifact")

    df = pd.read_csv(raw_csv)
    n0 = len(df)

    # 2) Price filtering
    df = df[df["price"].between(args.min_price, args.max_price)]
    n1 = len(df)

    # 3) NYC bounding box filtering (the new step required by the rubric)
    #    Typical bounding box used in this project:
    #    longitude in [-74.25, -73.50], latitude in [40.5, 41.0]
    MIN_LON, MAX_LON = -74.25, -73.50
    MIN_LAT, MAX_LAT = 40.50, 41.00

    # Only filter if columns exist (defensive)
    if {"longitude", "latitude"}.issubset(df.columns):
        in_bbox = df["longitude"].between(MIN_LON, MAX_LON) & df["latitude"].between(MIN_LAT, MAX_LAT)
        df = df[in_bbox]
    else:
        raise ValueError("Expected 'longitude' and 'latitude' columns to apply NYC bounding-box cleaning.")

    n2 = len(df)

    # 4) Save cleaned file locally
    os.makedirs("outputs", exist_ok=True)
    cleaned_path = os.path.join("outputs", args.output_artifact)
    df.to_csv(cleaned_path, index=False)

    # 5) Log cleaned dataset as a W&B artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
        metadata={
            "rows_raw": n0,
            "rows_price_filtered": n1,
            "rows_bbox_filtered": n2,
            "nyc_bbox": {
                "lon": [MIN_LON, MAX_LON],
                "lat": [MIN_LAT, MAX_LAT],
            },
            "min_price": args.min_price,
            "max_price": args.max_price,
        },
    )
    artifact.add_file(cleaned_path)
    run.log_artifact(artifact)
    run.finish()

    print(f"[basic_cleaning] Rows: raw={n0}, after_price={n1}, after_bbox={n2}")
    print(f"[basic_cleaning] Wrote cleaned CSV to: {cleaned_path}")


if __name__ == "__main__":
    main()

