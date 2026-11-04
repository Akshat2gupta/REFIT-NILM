"""Compare finished E5 folds with other models on the same homes. Run with the main environment."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from reglib import NON_FEATURES, base_path, home_metrics, prediction_path

NAMES = ("P0", "E2_jitter", "E1_L60_ctx", "E5_past", "E5_look")


def main() -> None:
    show = ["RMSE", "MAE", "R2", "MAE_ON", "MAE_OFF", "pred_heat", "off_energy_share", "energy_abs", "weekly_r", "weekly_abs"]
    for house in map(int, sys.argv[1:]):
        labels = pd.read_parquet(base_path(house), columns=list(NON_FEATURES))
        ok = labels["eval_ok"].to_numpy()
        y = labels["wm"].to_numpy(float)[ok]
        rows = {}
        for name in NAMES:
            path = prediction_path(name, house)
            if path.exists():
                pred = pd.read_parquet(path)["pred"].to_numpy(float)[ok]
                assert np.isfinite(pred).all(), name
                rows[name] = home_metrics(y, pred, labels.index[ok])
        print(f"\n--- House {house} ---")
        print(pd.DataFrame(rows).T[show].round(2).to_string())


if __name__ == "__main__":
    main()
