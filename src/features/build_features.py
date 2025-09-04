import argparse, pandas as pd, numpy as np
from pathlib import Path
from .elo import Elo
from ..utils.io import write_parquet
from ..utils.logs import get_logger

log = get_logger("features")

def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    # long format per rolling per squadra
    home = df[["date","season","matchweek","home_team","home_goals","away_goals"]].copy()
    home.columns = ["date","season","matchweek","team","goals_for","goals_against"]
    home["home"] = 1
    away = df[["date","season","matchweek","away_team","away_goals","home_goals"]].copy()
    away.columns = ["date","season","matchweek","team","goals_for","goals_against"]
    away["home"] = 0
    long = pd.concat([home, away], ignore_index=True).sort_values("date")
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["loss"] = (long["goals_for"] < long["goals_against"]).astype(int)

    long[["gf_ma5","ga_ma5","win_rate5"]] = (
        long.groupby("team")
            .rolling(5, on="date")[["goals_for","goals_against","win"]]
            .mean()
            .reset_index(drop=True)
            .rename(columns={"goals_for":"gf_ma5","goals_against":"ga_ma5","win":"win_rate5"})
    )

    # unisci di nuovo in formato match
    df = df.sort_values("date").copy()
    df = df.merge(
        long[["date","team","gf_ma5","ga_ma5","win_rate5"]],
        left_on=["date","home_team"],
        right_on=["date","team"],
        how="left"
    ).drop(columns=["team"]).rename(columns={
        "gf_ma5":"home_gf_ma5","ga_ma5":"home_ga_ma5","win_rate5":"home_wr5"
    })
    df = df.merge(
        long[["date","team","gf_ma5","ga_ma5","win_rate5"]],
        left_on=["date","away_team"],
        right_on=["date","team"],
        how="left"
    ).drop(columns=["team"]).rename(columns={
        "gf_ma5":"away_gf_ma5","ga_ma5":"away_ga_ma5","win_rate5":"away_wr5"
    })
    return df

def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    elo = Elo(k=18, base=1500, home_adv=60)
    elos_home = []
    elos_away = []
    for _, row in df.sort_values("date").iterrows():
        elos_home.append(elo.rating(row.home_team))
        elos_away.append(elo.rating(row.away_team))
        # update after match if goals known
        if not pd.isna(row.home_goals) and not pd.isna(row.away_goals):
            elo.update_match(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))
    df["elo_home_pre"] = elos_home
    df["elo_away_pre"] = elos_away
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    return df

def main(inp: str, out: str):
    df = pd.read_csv(inp, parse_dates=["date"])
    # outcome label
    def res(h,a):
        if pd.isna(h) or pd.isna(a): return np.nan
        if h > a: return "H"
        if h < a: return "A"
        return "D"
    df["result"] = [res(h,a) for h,a in zip(df["home_goals"], df["away_goals"])]

    df = add_elo(df)
    df = add_rolling_stats(df)

    # odds → implied prob (clipping)
    for col in ["home_odds","draw_odds","away_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["p_home_imp"] = 1/df["home_odds"] if "home_odds" in df else np.nan
    df["p_draw_imp"] = 1/df["draw_odds"] if "draw_odds" in df else np.nan
    df["p_away_imp"] = 1/df["away_odds"] if "away_odds" in df else np.nan

    write_parquet(df, out)
    log.info(f"Features saved to {out} with shape {df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    main(args.inp, args.out)
