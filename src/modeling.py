"""Baselines, models and leakage-free validation.

No hyperparameters are searched: settings are fixed up front so test data cannot influence
them. The only fitted choice is the ON/OFF threshold, picked from training data alone. Under
leave-one-house-out it comes from two training houses held back for that purpose, and under
the temporal split from the last part of each home's training period.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as cfg
from .evaluation import conditional_mae, onoff_metrics, regression_metrics

PRED_DIR = cfg.PREDICTIONS
THRESHOLD_BASELINE_GRID = (100, 200, 300, 500, 750, 1000, 1500, 2000)
INNER_VALIDATION_HOUSES = 2


def make_rf(n_estimators: int = 100, max_samples: int = 400_000) -> RandomForestRegressor:
    """Random Forest. ``min_samples_leaf`` and ``max_samples`` keep memory and overfitting in check."""
    return RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=10, max_features=0.33, max_samples=max_samples,
        n_jobs=-1, random_state=cfg.RANDOM_SEED)


def make_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        early_stopping=False, random_state=cfg.RANDOM_SEED)


def best_threshold(y: np.ndarray, pred: np.ndarray) -> float:
    """Prediction level that best separates ON from OFF on validation data."""
    scores = {t: onoff_metrics(y, pred, t)["F1"] for t in cfg.PRED_THRESHOLD_GRID}
    return max(scores, key=scores.get)


def fit_threshold_baseline(train: pd.DataFrame) -> dict:
    """Rule: if the aggregate is T watts above its 60-minute low, predict typical ON power."""
    y = train["wm"].to_numpy()
    excess = train["above_min_60"].to_numpy()
    scores = {t: onoff_metrics(y, (excess > t).astype(float), 0.5)["F1"] for t in THRESHOLD_BASELINE_GRID}
    level = max(scores, key=scores.get)
    return {"excess_w": level, "on_power_w": float(y[y > cfg.ACTUAL_ON_W].mean())}


def predict_threshold_baseline(rule: dict, table: pd.DataFrame) -> np.ndarray:
    return np.where(table["above_min_60"].to_numpy() > rule["excess_w"], rule["on_power_w"], 0.0)


def _fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                 inner_train: np.ndarray, inner_val: np.ndarray):
    """Fit one method on training rows. Returns ``(test_predictions, threshold, model)``."""
    if name == "always_off":
        return np.zeros(len(test)), 0.5, None
    if name == "threshold_rule":
        rule = fit_threshold_baseline(train)
        return predict_threshold_baseline(rule, test), rule["on_power_w"] / 2, rule

    def build(small: bool, rows: int):
        if name == "random_forest":  # coarse meters can have fewer rows than the usual tree sample
            return make_rf(30, min(150_000, rows)) if small else make_rf(max_samples=min(400_000, rows))
        return make_hgb()

    X = train[features].to_numpy(dtype="float32")
    y = train["wm"].to_numpy(dtype="float32")
    calibrator = build(True, int(inner_train.sum())).fit(X[inner_train], y[inner_train])
    threshold = best_threshold(y[inner_val], np.clip(calibrator.predict(X[inner_val]), 0, None))
    del calibrator
    model = build(False, len(X)).fit(X, y)
    return np.clip(model.predict(test[features].to_numpy(dtype="float32")), 0, None), threshold, model


def _save(experiment: str, name: str, house: int, test: pd.DataFrame, pred: np.ndarray, threshold: float):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = test[["wm", "eval_ok", "issue_frac", "house"]].assign(pred=pred.astype("float32"))
    # Write the threshold first, then move the finished parquet into place, so that a run killed
    # mid-write cannot leave a file that a later resume mistakes for a completed fold.
    target = PRED_DIR / f"{experiment}__{name}__h{house}.parquet"
    (PRED_DIR / f"{experiment}__{name}__h{house}.json").write_text(json.dumps({"threshold": threshold}))
    partial = target.with_suffix(".parquet.tmp")
    out.to_parquet(partial)
    os.replace(partial, target)


def load_predictions(experiment: str, name: str) -> tuple[pd.DataFrame, dict]:
    """Saved test predictions and the calibrated threshold for each house."""
    frames, thresholds = [], {}
    for path in sorted(PRED_DIR.glob(f"{experiment}__{name}__h*.parquet")):
        house = int(path.stem.rsplit("__h", 1)[1])  # rsplit: method names may contain "__h"
        frames.append(pd.read_parquet(path))
        thresholds[house] = json.loads(path.with_suffix(".json").read_text())["threshold"]
    if not frames:
        raise FileNotFoundError(f"No saved predictions for {experiment}/{name}.")
    return pd.concat(frames), thresholds


def _done(experiment: str, name: str, house: int) -> bool:
    """A fold counts as finished only if its parquet file exists and is readable."""
    path = PRED_DIR / f"{experiment}__{name}__h{house}.parquet"
    if not path.exists():
        return False
    try:
        pq.read_metadata(path)
    except pa.ArrowInvalid:
        print(f"Unreadable prediction file, refitting this fold: {path.name}", flush=True)
        return False
    return True


def run_leave_one_house_out(data: pd.DataFrame, features: list[str], methods: list[str],
                            experiment: str, force: bool = False) -> None:
    """Primary validation: each house is tested by models that never saw any of its data."""
    houses = sorted(data["house"].unique())
    for method in methods:
        for house in houses:
            if not force and _done(experiment, method, house):
                continue
            started = time.time()
            is_test = (data["house"] == house).to_numpy()
            train, test = data[~is_test], data[is_test]
            remaining = [h for h in houses if h != house]
            rng = np.random.default_rng(cfg.RANDOM_SEED + house)
            held_back = rng.choice(remaining, INNER_VALIDATION_HOUSES, replace=False)
            inner_val = train["house"].isin(held_back).to_numpy()
            pred, threshold, _ = _fit_predict(method, train, test, features, ~inner_val, inner_val)
            _save(experiment, method, house, test, pred, threshold)
            print(f"[{experiment}] {method} house {house}: threshold {threshold:.0f} W, "
                  f"{time.time() - started:.0f}s", flush=True)


def run_model_selection(data: pd.DataFrame, features: list[str], force: bool = False) -> pd.DataFrame:
    """Choose Random Forest or gradient boosting for each fold without touching its test house.

    Both models train on the same 1.5M-row sample of the inner training houses and are scored on
    the two held-back inner houses. The lower validation RMSE wins, a criterion fixed in advance:
    overall MAE would favour predicting nothing. Sampling keeps this affordable and is the same
    for both models, so the comparison stays fair.
    """
    path = cfg.METRICS / "model_selection_inner_validation.csv"
    cfg.METRICS.mkdir(parents=True, exist_ok=True)
    houses = sorted(data["house"].unique())
    rows = pd.read_csv(path).to_dict("records") if path.exists() and not force else []
    finished = {row["test_house"] for row in rows}  # resume: each fold has its own seed
    for house in houses:
        if house in finished:
            continue
        remaining = [h for h in houses if h != house]
        rng = np.random.default_rng(cfg.RANDOM_SEED + house)
        held_back = rng.choice(remaining, INNER_VALIDATION_HOUSES, replace=False)
        inner_train = data[data["house"].isin(remaining) & ~data["house"].isin(held_back)]
        inner_val = data[data["house"].isin(held_back)]
        sample = rng.choice(len(inner_train), min(1_500_000, len(inner_train)), replace=False)
        sampled = inner_train.iloc[sample]
        X, y = sampled[features].to_numpy(dtype="float32"), sampled["wm"].to_numpy(dtype="float32")
        X_val = inner_val[features].to_numpy(dtype="float32")
        y_val = inner_val["wm"].to_numpy(dtype="float32")
        for method, model in (("random_forest", make_rf(50, 400_000)), ("hist_gradient_boosting", make_hgb())):
            pred = np.clip(model.fit(X, y).predict(X_val), 0, None)
            rows.append({"test_house": house, "validation_houses": ",".join(map(str, sorted(held_back))),
                         "method": method, **regression_metrics(y_val, pred), **conditional_mae(y_val, pred)})
        print(f"[selection] fold for house {house} done", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def selected_methods(selection: pd.DataFrame) -> dict[int, str]:
    """Winning method for each outer test house, by validation RMSE."""
    best = selection.loc[selection.groupby("test_house")["RMSE"].idxmin()]
    return dict(zip(best["test_house"], best["method"]))


def temporal_split(data: pd.DataFrame, fraction: float = cfg.TRAIN_FRACTION, purge_steps: int = 120):
    """Earlier ``fraction`` of each house trains and the rest tests. Rows are never shuffled.

    ``purge_steps`` test rows are discarded after the boundary, so test windows do not reach
    back into the training period.
    """
    train, test = [], []
    for _, group in data.groupby("house", sort=True):
        cut = int(len(group) * fraction)
        train.append(group.iloc[:cut])
        test.append(group.iloc[cut + purge_steps:])
    return pd.concat(train), pd.concat(test)


def run_temporal(data: pd.DataFrame, features: list[str], methods: list[str], experiment: str = "temporal",
                 force: bool = False) -> None:
    """Secondary validation: future readings from homes the model has already seen."""
    train, test = temporal_split(data)
    # Calibrate on the last 20% of each home's training period.
    inner_val = np.concatenate([
        (np.arange(n) >= int(n * 0.8)) for n in train.groupby("house", sort=True).size()])
    for method in methods:
        if not force and all(_done(experiment, method, h) for h in test["house"].unique()):
            continue
        started = time.time()
        pred, threshold, _ = _fit_predict(method, train, test, features, ~inner_val, inner_val)
        for house in test["house"].unique():
            mask = (test["house"] == house).to_numpy()
            _save(experiment, method, house, test[mask], pred[mask], threshold)
        print(f"[{experiment}] {method}: threshold {threshold:.0f} W, {time.time() - started:.0f}s", flush=True)


def fit_final_model(data: pd.DataFrame, features: list[str], method: str):
    """Refit the selected method on every eligible house, for persistence.

    This model has no held-out test data of its own. Its expected performance is the
    leave-one-house-out result.
    """
    model = make_rf() if method == "random_forest" else make_hgb()
    model.fit(data[features].to_numpy(dtype="float32"), data["wm"].to_numpy(dtype="float32"))
    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "method": method, "features": features}, cfg.MODELS / "final_model.joblib", compress=3)
    return model
