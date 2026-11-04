"""Train the final batch model by leave one house out. Finished folds are reused.

Run ``scripts/run_experiments.py`` first for the baselines. Usage: ``python scripts/run_final_model.py``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "10")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config as cfg  # noqa: E402
from src.data_loader import eligible_houses  # noqa: E402
from src.lookahead import build_house, feature_names, training_rows  # noqa: E402
from src.modeling import INNER_VALIDATION_HOUSES, _done, _save, best_threshold, make_hgb  # noqa: E402

NAME = "lookahead_ctx"


def fit(rows: dict, houses: list[int]):
    return make_hgb().fit(np.vstack([rows[h][0] for h in houses]), np.concatenate([rows[h][1] for h in houses]),
                          sample_weight=np.concatenate([rows[h][2] for h in houses]))


def main() -> None:
    houses = sorted(eligible_houses())
    rows, names = {}, None
    for house in houses:
        table = build_house(house)
        names = feature_names(table)
        rows[house] = training_rows(table, house)
        del table
    print(f"{len(names)} features, {sum(len(r[1]) for r in rows.values()):,} training rows", flush=True)
    for house in houses:
        if _done("loho", NAME, house):
            continue
        started = time.time()
        remaining = [h for h in houses if h != house]
        held = np.random.default_rng(cfg.RANDOM_SEED + house).choice(remaining, INNER_VALIDATION_HOUSES, replace=False).tolist()
        calibrator = fit(rows, [h for h in remaining if h not in held])
        scores, truth = [], []
        for other in held:
            table = build_house(other)
            scores.append(np.clip(calibrator.predict(table[names].to_numpy(dtype="float32")), 0, None))
            truth.append(table["wm"].to_numpy(dtype="float32"))
            del table
        threshold = best_threshold(np.concatenate(truth), np.concatenate(scores))
        del calibrator, scores, truth
        model = fit(rows, remaining)
        test = build_house(house)
        pred = np.clip(model.predict(test[names].to_numpy(dtype="float32")), 0, None)
        _save("loho", NAME, house, test, pred, threshold)
        print(f"House {house}: threshold {threshold:.0f} W, {time.time() - started:.0f}s", flush=True)
        del model, test, pred
    print("Final model finished.", flush=True)


if __name__ == "__main__":
    main()
