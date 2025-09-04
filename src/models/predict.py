import argparse, pandas as pd, numpy as np
from joblib import load
from ..utils.io import write_csv
from ..utils.logs import get_logger

log = get_logger("predict")

FEATURES = [
    "elo_diff","home_gf_ma5","home_ga_ma5","home_wr5",
    "away_gf_ma5","away_ga_ma5","away_wr5",
    "p_home_imp","p_draw_imp","p_away_imp"
]

def main(model_path: str, fixtures_csv: str, out_csv: str):
    clf = load(model_path)
    fx = pd.read_csv(fixtures_csv, parse_dates=["date"])
    # placeholder feature engineering minima per fixtures future:
    fx["elo_diff"] = fx.get("elo_diff", 0)  # se non disponibile, 0
    for col in ["home_gf_ma5","home_ga_ma5","home_wr5","away_gf_ma5","away_ga_ma5","away_wr5"]:
        if col not in fx: fx[col] = np.nan
    for col in ["home_odds","draw_odds","away_odds"]:
        if col in fx: fx[col] = pd.to_numeric(fx[col], errors="coerce")
    fx["p_home_imp"] = 1/fx["home_odds"] if "home_odds" in fx else np.nan
    fx["p_draw_imp"] = 1/fx["draw_odds"] if "draw_odds" in fx else np.nan
    fx["p_away_imp"] = 1/fx["away_odds"] if "away_odds" in fx else np.nan

    X = fx[FEATURES].fillna(0)
    proba = clf.predict_proba(X)
    classes = clf.classes_
    fx["p_H"] = proba[:, list(classes).index("H")] if "H" in classes else 0.0
    fx["p_D"] = proba[:, list(classes).index("D")] if "D" in classes else 0.0
    fx["p_A"] = proba[:, list(classes).index("A")] if "A" in classes else 0.0
    fx["pick"] = ["HDA"[i] for i in proba.argmax(axis=1)]

    write_csv(fx, out_csv)
    log.info(f"Predictions saved to {out_csv} (rows={len(fx)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.model, args.fixtures, args.out)
