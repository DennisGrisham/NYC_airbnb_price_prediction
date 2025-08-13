import argparse
import pandas as pd
import mlflow
import os
import json

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with null values
    df = df.dropna()

    # Drop high-cardinality or personally identifying columns
    high_card_cols = ["host_name", "name", "last_review"]
    df = df.drop(columns=[col for col in high_card_cols if col in df.columns])

    # Filter out unrealistic price values
    df = df[df["price"] < 1000]

    return df

def go(config):
    with mlflow.start_run():

        # Load raw dataset
        df = pd.read_csv(config["input_artifact"])

        # Clean the data
        cleaned_df = clean_data(df)

        # Save cleaned data to output path
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", config["output_artifact"])
        cleaned_df.to_parquet(output_path, index=False)

        # Log artifact to MLflow
        mlflow.log_artifact(output_path, artifact_path="clean_data")

        # Save cleaned data to outputs folder (for local access)
        cleaned_df.to_csv("outputs/clean_sample.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/clean_data/clean_config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    go(config)
