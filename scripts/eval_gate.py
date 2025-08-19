#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, tempfile
import pandas as pd
import numpy as np

MIN_R2 = 0.50
MAX_MAE = 40.0

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-uri", required=True, help="e.g. models:/nyc_airbnb_random_forest/5")
    p.add_argument("--test-csv", required=True, help="Path to test_data.csv")
    p.add_argument("--target-col", default="price")
    p.add_argument("--min-r2", type=float, default=MIN_R2)
    p.add_argument("--max-mae", type=float, default=MAX_MAE)
    args = p.parse_args()

    if not os.path.exists(args.test_csv):
        print(f"❌ Test CSV not found: {args.test_csv}", file=sys.stderr)
        sys.exit(2)

    with tempfile.TemporaryDirectory() as tmp:
        preds_path = os.path.join(tmp, "preds.csv")
        cmd = [
            "mlflow", "models", "predict",
            "-m", args.model_uri,
       	    "--env-manager", "conda",
            "--input-path", args.test_csv,   # <- changed
            "--content-type", "csv",
            "--output-path", preds_path      # <- changed
        ]
        print(f"[INFO] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)
        if res.returncode != 0 or not os.path.exists(preds_path):
            print("❌ Failed to generate predictions with mlflow models predict.", file=sys.stderr)
            sys.exit(3)

        # Load truth/preds (mlflow may write JSON or CSV depending on version)
        df_test = pd.read_csv(args.test_csv)
        y_true = df_test[args.target_col].values

        # Read preds file; try JSON first, fall back to CSV
        raw = open(preds_path, "r").read().strip()
        try:
            obj = json.loads(raw)  # MLflow often writes: {"predictions": [...]}
            y_pred = np.asarray(obj["predictions"])
        except json.JSONDecodeError:
            df_pred = pd.read_csv(preds_path)
            if "predictions" in df_pred.columns:
                y_pred = df_pred["predictions"].to_numpy()
            elif df_pred.shape[1] == 1:
                # Single unnamed column -> treat as predictions
                y_pred = df_pred.iloc[:, 0].to_numpy()
            else:
                print(f"❌ 'predictions' column not found in {preds_path}. Got cols: {list(df_pred.columns)}", file=sys.stderr)
                sys.exit(4)

        # If MLflow returned shape (n, 1), squeeze to (n,)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()


        r2 = r2_score(y_true, y_pred)
        m = mae(y_true, y_pred)
        print(f"[METRICS] R2={r2:.4f}  MAE={m:.4f}")

        ok = True
        if r2 < args.min_r2:
            print(f"❌ Gate failed: R2 {r2:.3f} < {args.min_r2:.3f}", file=sys.stderr)
            ok = False
        if m > args.max_mae:
            print(f"❌ Gate failed: MAE {m:.2f} > {args.max_mae:.2f}", file=sys.stderr)
            ok = False

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/eval_metrics.json", "w") as f:
            json.dump({"r2": float(r2), "mae": float(m), "min_r2_gate": float(args.min_r2), "max_mae_gate": float(args.max_mae)}, f, indent=2)
        print("✅ Wrote outputs/eval_metrics.json")

        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
