"""Stress test: apply the final model to the washer-dryer homes left out of model development.

Houses 1, 8, 9 and 18 have a second washing load that their washing-machine label does not include, so this
is a harder setting than the 11 eligible homes, not a clean independent test. Models train on all 11 homes.
ON/OFF thresholds are the median of the leave-one-house-out thresholds, so these homes do not set them.
Usage: ``python scripts/evaluate_excluded_homes.py``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "10")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src.data_loader import eligible_houses  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.lookahead import build_house, feature_names, training_rows  # noqa: E402
from src.modeling import load_predictions, make_hgb  # noqa: E402

EXCLUDED = (1, 8, 9, 18)


def weekly(table: pd.DataFrame) -> dict:
    rows = table[table["eval_ok"]]
    week = rows[["wm", "pred"]].resample("7D").sum() / 60_000
    week = week[rows["wm"].resample("7D").count() >= 0.8 * 10_080]
    return {"Weekly_r": week["wm"].corr(week["pred"]), "Weekly_abs_error_pct": 100 * (week["pred"] - week["wm"]).abs().mean() / week["wm"].mean()}


def main() -> None:
    houses = sorted(eligible_houses())
    rows, names = {}, None
    for house in houses:
        table = build_house(house)
        names = feature_names(table)
        rows[house] = training_rows(table, house)
        del table
    past = [i for i, n in enumerate(names) if not n.startswith(("fut_", "ctx_"))]
    print(f"{len(names)} features, of which {len(past)} past-only", flush=True)
    X = np.vstack([rows[h][0] for h in houses])
    y = np.concatenate([rows[h][1] for h in houses])
    w = np.concatenate([rows[h][2] for h in houses])
    del rows
    models = {"Final model": (make_hgb().fit(X, y, sample_weight=w), list(range(len(names))), "lookahead_ctx"),
              "Past-only baseline": (make_hgb().fit(X[:, past], y, sample_weight=w), past, "hist_gradient_boosting")}
    del X, y, w
    output = []
    for name, (model, columns, method) in models.items():
        threshold = float(np.median(list(load_predictions("loho", method)[1].values())))
        for house in EXCLUDED:
            table = build_house(house)
            pred = np.clip(model.predict(table[names].to_numpy(dtype="float32")[:, columns]), 0, None)
            scored = table[["wm", "eval_ok", "issue_frac"]].assign(pred=pred)
            result = evaluate(scored, threshold, with_events=False)
            rows_ok = scored[scored["eval_ok"]]
            output.append({"Model": name, "House": house, "Threshold_W": threshold, **result, **weekly(scored), "Energy_abs_error_pct": abs(result["Energy_error_pct"]),
                           "Off_energy_share_pct": 100 * rows_ok.loc[rows_ok["wm"] <= cfg.ACTIVE_W, "pred"].sum() / rows_ok["pred"].sum()})
            print(f"{name}, House {house}: done", flush=True)
            del table, scored
    frame = pd.DataFrame(output)
    frame.round(4).to_csv(cfg.METRICS / "final_model_excluded_homes.csv", index=False)
    show = ["RMSE", "MAE", "R2", "MAE_ON", "Precision", "Recall", "F1", "Energy_error_pct", "Energy_abs_error_pct", "Off_energy_share_pct", "Weekly_r", "Weekly_abs_error_pct"]
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print(frame[["Model", "House", *show]].round(2).to_string(index=False))
    print("\nMean of 4 homes:")
    print(frame.groupby("Model")[show].mean().round(3).T.to_string())
    print("Stress test finished.", flush=True)


if __name__ == "__main__":
    main()
