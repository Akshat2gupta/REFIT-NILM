"""E0: harness control and simple baselines.

P0 uses the baseline's 34 features and settings with the sampled training rows. It must sit close to the
saved full-data gradient boosting, otherwise later comparisons cannot be trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import HOUSES
from reglib import PRED, load_home, prediction_path, report, run, score


def hour_profile() -> None:
    """Average washer power by hour of day in the training homes. Knows nothing about this home's total."""
    tables = {}
    for h in HOUSES:
        _, labels = load_home(h, ("clock",))
        hours = pd.read_parquet(f"{__import__('features').CACHE}/house{h}_clock.parquet", columns=["hour"])["hour"].astype(int)
        tables[h] = labels[["wm"]].assign(hour=hours.to_numpy())
    for house in HOUSES:
        path = prediction_path("E0_hour_profile", house)
        if path.exists():
            continue
        profile = pd.concat([t for h, t in tables.items() if h != house]).groupby("hour")["wm"].mean()
        pd.DataFrame({"pred": profile.reindex(tables[house]["hour"]).to_numpy(dtype="float32")}, index=tables[house].index).to_parquet(path)


def main() -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    run("P0", note="Baseline features and settings on sampled rows: the harness control")
    run("E0_ridge", kind="ridge", note="Linear baseline on the same 34 features")
    hour_profile()
    score("E0_hour_profile")
    show = ["MAE", "RMSE", "R2", "MAE_ON", "MAE_OFF", "pred_heat", "off_energy_share", "energy_abs", "weekly_r", "weekly_abs"]
    rows = {n: score(n).mean(numeric_only=True)[show] for n in ("saved:hist_gradient_boosting", "saved:random_forest", "saved:always_off", "P0", "E0_ridge", "E0_hour_profile")}
    print(pd.DataFrame(rows).T.round(3).to_string())
    diff = score("P0").set_index("house")["RMSE"] - score("saved:hist_gradient_boosting").set_index("house")["RMSE"]
    print("P0 minus saved RMSE by home:", diff.round(2).to_dict(), "max abs", round(float(np.abs(diff).max()), 2))
    report(["P0", "E0_ridge", "E0_hour_profile"], "saved:hist_gradient_boosting", "E0")


if __name__ == "__main__":
    main()
