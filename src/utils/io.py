from pathlib import Path
import pandas as pd

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def write_parquet(df, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
