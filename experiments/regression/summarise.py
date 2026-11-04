"""Write one table of every completed regression experiment, with the reason for each decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from reglib import METRIC, NON_FEATURES, ROOT, base_path, home_metrics, prediction_path

ONE_MINUTE, EIGHT_SECOND = "1-minute", "8-second, outside the brief"
ROWS = [
    # name, idea, input, needs future data, decision, reason
    ("saved_hist_gradient_boosting", "Past-only baseline: 34 features, gradient boosting", ONE_MINUTE, False, "Baseline", "Reference for everything else"),
    ("P0", "Baseline features on sampled training rows", ONE_MINUTE, False, "Control", "Within about 1 W of the baseline in each home, so sampling is safe"),
    ("E0_ridge", "Linear model on the baseline features", ONE_MINUTE, False, "Rejected", "Much worse than gradient boosting"),
    ("E0_hour_profile", "Average power by hour of day", ONE_MINUTE, False, "Rejected", "Knows nothing about the home's own signal"),
    ("E1_L30", "Add 30 minutes of future readings", ONE_MINUTE, True, "Rejected", "Real gain, but far behind cycle context"),
    ("E1_L60", "Add 60 minutes of future readings", ONE_MINUTE, True, "Rejected", "Real gain, but far behind cycle context"),
    ("E1_L120", "Add 120 minutes of future readings", ONE_MINUTE, True, "Rejected", "Almost no gain over 60 minutes"),
    ("E1_L60_ctx", "60 minutes of future readings plus cycle context", ONE_MINUTE, True, "Selected", "Lowest 1-minute RMSE; passes the keep rule"),
    ("E1_all", "All 148 past, future and context features", ONE_MINUTE, True, "Rejected", "No better than the smaller set"),
    ("C_ctx_shift7d", "Control: cycle context moved 7 days", ONE_MINUTE, True, "Control", "Gain vanishes, so the gain is tied to the current minute"),
    ("A_no_shape", "Selected model without 5 block-shape features", ONE_MINUTE, True, "Not used", "Almost no change; other features carry similar information"),
    ("E2_long", "Past windows up to 3 hours", ONE_MINUTE, False, "Rejected", "Worse than the control"),
    ("E2_jitter", "Add small past power variations", ONE_MINUTE, False, "Live option", "Best past-only 1-minute set, but fails the keep rule on MAE when ON"),
    ("E2_blocks", "Add past heating-block features", ONE_MINUTE, False, "Rejected", "No clear gain"),
    ("E2_clock", "Add hour and day of week", ONE_MINUTE, False, "Rejected", "No clear gain"),
    ("E2_history", "Add the home's own long-term habits", ONE_MINUTE, False, "Rejected", "Worse than the control"),
    ("E2_all", "All past-only feature groups", ONE_MINUTE, False, "Rejected", "No better than jitter alone"),
    ("E3_look_hurdle", "Selected features: ON/OFF, then power", ONE_MINUTE, True, "Rejected", "Same RMSE within noise; fails the keep rule"),
    ("E3_look_states", "Selected features: four power states", ONE_MINUTE, True, "Rejected", "Much worse RMSE"),
    ("E3_look_log", "Selected features: log target", ONE_MINUTE, True, "Rejected", "Heating is badly under-predicted"),
    ("E3_look_mae", "Selected features: absolute-error loss", ONE_MINUTE, True, "Rejected", "Collapses to always off"),
    ("E3_look_poisson", "Selected features: Poisson loss", ONE_MINUTE, True, "Failed run", "Predictions overflowed to infinity, so not a fair test"),
    ("E3_past_hurdle", "Jitter features: ON/OFF, then power", ONE_MINUTE, False, "Rejected", "No gain over jitter"),
    ("E3_past_states", "Jitter features: four power states", ONE_MINUTE, False, "Rejected", "Much worse RMSE"),
    ("E3_past_log", "Jitter features: log target", ONE_MINUTE, False, "Rejected", "Heating is badly under-predicted"),
    ("E3_past_mae", "Jitter features: absolute-error loss", ONE_MINUTE, False, "Rejected", "Collapses to always off"),
    ("E3_past_poisson", "Jitter features: Poisson loss", ONE_MINUTE, False, "Failed run", "Predictions overflowed to infinity, so not a fair test"),
    ("E2b_fine_past", "Add within-minute statistics from ~8 s data", EIGHT_SECOND, False, "Outside brief", "Large gain, reported as evidence for finer metering"),
    ("E2b_fine_all_past", "All past features plus ~8 s statistics", EIGHT_SECOND, False, "Outside brief", "No better than the ~8 s statistics alone"),
    ("E2b_fine_look", "Look-ahead features plus ~8 s statistics", EIGHT_SECOND, True, "Outside brief", "Lowest RMSE overall; no separate shift control was run"),
    ("C_fine_shift7d", "Control: ~8 s statistics moved 7 days", EIGHT_SECOND, False, "Control", "Gain vanishes, so the gain is tied to the current minute"),
]
COLUMNS = ["MAE", "RMSE", "R2", "MAE_ON", "off_energy_share", "energy_abs", "weekly_r", "weekly_abs"]


def one_home(name: str, house: int) -> dict:
    labels = pd.read_parquet(base_path(house), columns=list(NON_FEATURES))
    ok = labels["eval_ok"].to_numpy()
    pred = pd.read_parquet(prediction_path(name, house))["pred"].to_numpy(float)[ok]
    return home_metrics(labels["wm"].to_numpy(float)[ok], pred, labels.index[ok])


def main() -> None:
    rows = []
    for name, idea, source, future, decision, reason in ROWS:
        means = pd.read_csv(METRIC / f"{name}.csv").replace([np.inf, -np.inf], np.nan)[COLUMNS].mean()
        if decision == "Failed run":  # scores from overflowing predictions mean nothing, so none are reported
            means[:] = np.nan
        rows.append({"Experiment": name, "Idea": idea, "Input": source, "Needs future data": future, "Homes": 11, **means.round(3).to_dict(), "Decision": decision, "Reason": reason})
    for name, idea, future, reason in (("E5_look", "Small Seq2Point CNN, 60 minutes either side", True, "Behind E1_L60_ctx on the one home tested (RMSE 117.89); full run not pursued"),
                                       ("E5_past", "Small Seq2Point CNN, past 121 minutes", False, "No better than E2_jitter on the one home tested (RMSE 161.61); full run not pursued")):
        rows.append({"Experiment": name, "Idea": idea, "Input": ONE_MINUTE, "Needs future data": future, "Homes": 1, **{k: round(float(v), 3) for k, v in one_home(name, 16).items() if k in COLUMNS}, "Decision": "Rejected", "Reason": reason})
    table = pd.DataFrame(rows)
    path = ROOT / "outputs" / "metrics" / "experiments_summary.csv"
    table.to_csv(path, index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 44, "display.max_rows", 50)
    print(table[["Experiment", "Input", "Homes", "RMSE", "MAE", "R2", "Decision"]].to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
