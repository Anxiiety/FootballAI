from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class ProviderBase(ABC):
    @abstractmethod
    def fetch_upcoming(self) -> pd.DataFrame:
        ...

    def save(self, df: pd.DataFrame, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
