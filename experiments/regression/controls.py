"""Try to break the two largest gains (cycle context and 8-second input) before believing them.

1. Clock alignment: means of the ~8 s file must line up with the 1-minute aggregate at lag 0.
2. Past-only check for ``fine``: changing later 8 s statistics must leave earlier values unchanged.
3. Negative control: move each group's values 7 days along the same home. The features keep their
   distribution and the home's character, but lose their link to this minute, so the gain should vanish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from features import CACHE, HOUSES, ensure_house
from reglib import OUT, report, run
from src.data_loader import read_clean_csv
from src.preprocessing import clean_file_true_utc, prepare_clean_house

SHIFT_ROWS = 7 * 1440
LAGS = range(-2, 3)


def alignment() -> None:
    rows = []
    for house in HOUSES:
        raw = read_clean_csv(house, ["Unix", "Aggregate"])
        raw["utc"] = clean_file_true_utc(raw["Unix"])
        fine_mean = raw.dropna(subset=["utc"]).groupby(pd.Series(raw["utc"].dt.floor("min"), name="minute"))["Aggregate"].mean()
        minute = prepare_clean_house(house)["aggregate"]
        fine_mean = fine_mean.reindex(minute.index)
        change_a, change_b = minute.diff(), fine_mean.diff()
        row = {"house": house, "shared_minutes": int(fine_mean.notna().sum())}
        for lag in LAGS:
            row[f"lag{lag}"] = change_a.corr(change_b.shift(lag))
        rows.append(row)
        print(f"House {house}: " + " ".join(f"{lag}:{row[f'lag{lag}']:.3f}" for lag in LAGS), flush=True)
        del raw, fine_mean, minute
    pd.DataFrame(rows).to_csv(OUT / "control_alignment.csv", index=False)


def fine_is_past_only() -> None:
    house, cached = 16, pd.read_parquet(F.FINE_CACHE / "house16_fine_every1.parquet")
    index = cached.index
    cut = len(index) // 2
    changed = cached.copy()
    changed.iloc[cut + 1:] = changed.iloc[cut + 1:].to_numpy() + 500.0
    original = pd.read_parquet
    results = {}
    for label, table in (("same", cached), ("changed", changed)):
        F.pd.read_parquet = lambda *a, table=table, **k: table
        results[label] = (F.fine(house, index, False), F.fine(house, index, True))
    F.pd.read_parquet = original
    past = np.allclose(results["same"][0].iloc[:cut + 1], results["changed"][0].iloc[:cut + 1], equal_nan=True)
    future = np.allclose(results["same"][1].iloc[:cut + 1], results["changed"][1].iloc[:cut + 1], equal_nan=True)
    print(f"fine unchanged by later data: {past} (expected True); futfine unchanged: {future} (expected False)", flush=True)
    assert past and not future


def shifted_group(source: str, name: str) -> None:
    for house in HOUSES:
        ensure_house(house, (source,))
        path = CACHE / f"house{house}_{name}.parquet"
        if not path.exists():
            frame = pd.read_parquet(CACHE / f"house{house}_{source}.parquet")
            pd.DataFrame(np.roll(frame.to_numpy(), SHIFT_ROWS, axis=0), index=frame.index, columns=frame.columns).to_parquet(path)


def main() -> None:
    fine_is_past_only()
    alignment()
    shifted_group("ctx", "ctxshift")
    shifted_group("fine", "fineshift")
    run("C_ctx_shift7d", ("fut30", "fut60", "ctxshift"), note="Negative control for ctx: values moved 7 days within each home")
    run("C_fine_shift7d", ("fineshift",), note="Negative control for fine: values moved 7 days within each home")
    report(["C_ctx_shift7d", "E1_L60_ctx"], "E1_L60", "ctx control against E1_L60")
    report(["C_fine_shift7d", "E2b_fine_past"], "P0", "fine control against P0")


if __name__ == "__main__":
    main()
