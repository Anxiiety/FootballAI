"""
Esempio provider.
Da adattare a una tua fonte (API ufficiali o HTML di cui hai il permesso).
Per demo: genera 3 gare fittizie con quote plausibili.
"""
import argparse, pandas as pd
from datetime import datetime, timedelta
from .provider_base import ProviderBase

TEAMS = ["Inter","Milan","Juventus","Roma","Napoli","Lazio","Atalanta","Fiorentina"]

class ExampleProvider(ProviderBase):
    def fetch_upcoming(self) -> pd.DataFrame:
        now = datetime.now()
        rows = []
        pairs = [("Inter","Fiorentina"),("Milan","Roma"),("Napoli","Juventus")]
        odds = [(1.70,3.80,5.00),(2.05,3.30,3.60),(2.60,3.20,2.80)]
        for i,(h,a) in enumerate(pairs):
            rows.append({
                "date": (now + timedelta(days=2+i)).date().isoformat(),
                "home_team": h, "away_team": a,
                "home_odds": odds[i][0], "draw_odds": odds[i][1], "away_odds": odds[i][2]
            })
        return pd.DataFrame(rows)

def main(out: str, league: str):
    df = ExampleProvider().fetch_upcoming()
    ExampleProvider().save(df, out)
    print(f"Saved fixtures to {out} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="Serie A")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out, args.league)
