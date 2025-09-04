import argparse, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from joblib import load
from ..utils.logs import get_logger
from ..utils.io import read_parquet

log = get_logger("eval")

FEATURES = [
    "elo_diff","home_gf_ma5","home_ga_ma5","home_wr5",
    "away_gf_ma5","away_ga_ma5","away_wr5",
    "p_home_imp","p_draw_imp","p_away_imp"
]

def main(inp: str, model_path: str, holdout_weeks: int):
    df = read_parquet(inp).sort_values(["season","matchweek"])
    # holdout ultime N giornate della stagione più recente
    latest = df["season"].dropna().astype(str).max()
    cur = df[df["season"].astype(str)==latest]
    maxmw = cur["matchweek"].max()
    mask = (df["season"].astype(str)==latest) & (df["matchweek"]> maxmw - holdout_weeks)
    test = df[mask].dropna(subset=["result"])
    train = df[~mask].dropna(subset=["result"])

    if len(test)==0 or len(train)==0:
        log.warning("Holdout troppo piccolo o dati insufficienti.")
        return

    clf = load(model_path)
    X_test = test[FEATURES].fillna(train[FEATURES].median())
    y_test = test["result"]
    proba = clf.predict_proba(X_test)
    y_pred = clf.classes_[proba.argmax(axis=1)]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, proba, labels=clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    log.info(f"Holdout season={latest}, weeks={holdout_weeks}, n={len(test)}")
    log.info(f"Accuracy={acc:.3f}  LogLoss={ll:.3f}")
    log.info(f"Confusion Matrix (classes={list(clf.classes_)}):\n{cm}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", dest="model", required=True)
    ap.add_argument("--holdout_weeks", type=int, default=5)
    args = ap.parse_args()
    main(args.inp, args.model, args.holdout_weeks)
