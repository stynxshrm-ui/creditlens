"""
models/train.py

Unified training entry point for CreditLens models.
Run this to train the scorecard and register it as champion in MLflow.

Usage:
    python models/train.py --model scorecard
"""

import argparse
import duckdb
import pandas as pd
import mlflow
from pathlib import Path

DB_PATH = Path("data/creditlens.duckdb")


def load_features() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM model_features").fetchdf()
    con.close()
    print(f"Loaded {len(df):,} loans with {len(df.columns)} features")
    return df


def main(model: str):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("creditlens")

    df = load_features()

    if model == "scorecard":
        from models.scorecard.scorecard import train_scorecard
        metrics, _ = train_scorecard(df)
    elif model == "xgboost":
        from models.pd.xgboost_model import train_xgboost
        metrics, _, _, _ = train_xgboost(df)
    elif model == "neural_net":
        from models.pd.neural_net import train_neural_net
        metrics = train_neural_net(df)
    else:
        raise ValueError(f"Unknown model: {model}")

    print(f"\nDone. Gini: {metrics['gini']:.4f} | KS: {metrics['ks']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="scorecard",
                        choices=["scorecard", "xgboost", "neural_net"])
    args = parser.parse_args()
    main(args.model)