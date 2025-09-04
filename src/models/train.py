import argparse, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from joblib import dump
from ..utils.logs import get_logger
from ..utils.io import read_parquet
from ..config import MODEL_DIR

log = get_logger("train")

FEATURES = [
    "elo_diff","home_gf_ma5","home_ga_ma5","home_wr5",
    "away_gf_ma5","away_ga_ma5","away_wr5",
    "p_home_imp","p_draw_imp","p_away_imp"
]

def main(inp: str, model_path: str):
    df = read_parquet(inp)
    df = df.dropna(subset=["result"])
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["result"]
    X, y = shuffle(X, y, random_state=42)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, multi_class="multinomial"))
    ])
    clf.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, model_path)
    log.info(f"Model saved to {model_path}. Train size={len(y)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", dest="model", required=True)
    args = ap.parse_args()
    main(args.inp, args.model)
