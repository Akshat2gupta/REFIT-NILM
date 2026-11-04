"""Write each home's 1-minute aggregate for E5. Run once with the main environment."""
from __future__ import annotations

import pandas as pd

from features import CACHE, HOUSES
from src.preprocessing import prepare_clean_house


def main() -> None:
    for house in HOUSES:
        path = CACHE / f"house{house}_series.parquet"
        if not path.exists():
            prepare_clean_house(house)[["aggregate"]].astype("float32").to_parquet(path)
        print(f"House {house}: {len(pd.read_parquet(path)):,} minutes", flush=True)


if __name__ == "__main__":
    main()
