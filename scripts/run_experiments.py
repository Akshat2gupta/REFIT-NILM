"""Train and evaluate everything. Finished folds are reused, so an interrupted run can resume.

Usage: ``python scripts/run_experiments.py``      (add ``--force`` to retrain from scratch)
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import eligible_houses  # noqa: E402
from src.feature_engineering import build_dataset  # noqa: E402
from src.modeling import (fit_final_model, run_leave_one_house_out, run_model_selection,  # noqa: E402
                          run_temporal, selected_methods)


def run_all(force: bool = False) -> None:
    houses = eligible_houses()

    data, features = build_dataset(houses)
    print(f"1-minute data: {len(data):,} rows, {len(features)} features, houses {houses}", flush=True)
    methods = ["always_off", "threshold_rule", "random_forest", "hist_gradient_boosting"]
    run_leave_one_house_out(data, features, methods, "loho", force)
    selection = run_model_selection(data, features, force)
    winners = most_selected(selected_methods(selection))
    print(f"Most often selected on validation houses: {winners}", flush=True)
    models = ["random_forest", "hist_gradient_boosting"]
    run_temporal(data, features, ["always_off", "threshold_rule", *models], "temporal", force)
    fit_final_model(data, features, winners)
    del data
    gc.collect()

    # What happens with a coarser meter: 15- and 30-minute readings.
    for step in (15, 30):
        data, features = build_dataset(houses, step_min=step)
        print(f"{step}-minute data: {len(data):,} rows, {len(features)} features", flush=True)
        run_leave_one_house_out(data, features, models, f"loho_{step}min", force)
        del data
        gc.collect()


def most_selected(choices: dict[int, str]) -> str:
    """Most frequent winner across folds. A tie goes to the Random Forest, the simpler model."""
    names = list(choices.values())
    return max(sorted(set(names), key=lambda n: n != "random_forest"), key=names.count)


if __name__ == "__main__":
    run_all(force="--force" in sys.argv)
    print("All experiments finished.")
