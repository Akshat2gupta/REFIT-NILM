"""Shared harness: leave-one-house-out runs, metrics and comparisons for washing-machine power.

Every run predicts the frozen baseline's exact rows, and every score uses its exact evaluation minutes.
Training keeps every informative row and a weighted 1-in-8 sample of quiet OFF minutes, because the full
table does not fit alongside extra features in this laptop's free memory. The P0 control measures any
effect of this against the saved full-data baseline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CACHE, FUTURE_GROUPS, FINE_GROUPS, HERE, HOUSES, ROOT, base_path, ensure_house

OUT = HERE / "outputs"
PRED = OUT / "predictions"
METRIC = OUT / "metrics"
LOG = OUT / "experiment_log.csv"
SAVED = ROOT / "outputs" / "predictions"
SEED = 42
KEEP_EVERY = 8
HARD_EXCESS_W = 300
NON_FEATURES = ("wm", "issue_frac", "eval_ok")


def base_columns() -> list[str]:
    import pyarrow.parquet as pq
    names = pq.read_schema(base_path(HOUSES[0])).names
    return [c for c in names if c not in NON_FEATURES and not c.startswith("__")]


def load_home(house: int, groups: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels on the baseline's rows."""
    ensure_house(house, groups)
    base = pd.read_parquet(base_path(house))
    labels = base[list(NON_FEATURES)]
    frames = [base.drop(columns=list(NON_FEATURES))]
    frames += [pd.read_parquet(CACHE / f"house{house}_{g}.parquet") for g in groups]
    return pd.concat(frames, axis=1), labels


def training_rows(x: pd.DataFrame, labels: pd.DataFrame, house: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ON minutes and all minutes above the home's recent low, plus a weighted sample of the rest."""
    y = labels["wm"].to_numpy(dtype="float32")
    hard = (y > 20) | (x["above_min_60"].to_numpy() >= HARD_EXCESS_W)
    rng = np.random.default_rng(SEED + house)
    keep = hard | (rng.random(len(y)) < 1 / KEEP_EVERY)
    weight = np.where(hard[keep], 1.0, float(KEEP_EVERY)).astype("float32")
    return x.to_numpy(dtype="float32")[keep], y[keep], weight


def _hgb(**extra) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**{"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50, "early_stopping": False, "random_state": SEED, **extra})


def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, early_stopping=False, random_state=SEED)


def fit_predict(kind: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if kind == "mse":
        return _hgb().fit(X, y, sample_weight=w).predict(X_test)
    if kind == "mae":
        return _hgb(loss="absolute_error").fit(X, y, sample_weight=w).predict(X_test)
    if kind == "poisson":
        return _hgb(loss="poisson").fit(X, y, sample_weight=w).predict(X_test)
    if kind == "log":
        return np.expm1(_hgb().fit(X, np.log1p(y), sample_weight=w).predict(X_test))
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model.fit(np.nan_to_num(X), y, ridge__sample_weight=w).predict(np.nan_to_num(X_test))
    if kind == "hurdle":      # probability ON x expected power when ON
        on = y > 20
        p = _clf().fit(X, on, sample_weight=w).predict_proba(X_test)[:, 1]
        return p * _hgb().fit(X[on], y[on]).predict(X_test)
    if kind == "states":      # probability of each state x its expected power
        band = np.digitize(y, [20, 200, 1000])
        p = _clf().fit(X, band, sample_weight=w).predict_proba(X_test)
        out = np.zeros(len(X_test))
        for column, value in enumerate(np.unique(band)):
            part = band == value
            level = float(np.average(y[part], weights=w[part])) if value == 0 else None
            out += p[:, column] * (level if value == 0 else _hgb(max_iter=150).fit(X[part], y[part]).predict(X_test))
        return out
    raise KeyError(kind)


def prediction_path(name: str, house: int) -> Path:
    return PRED / f"{name}__h{house}.parquet"


def run(name: str, groups: tuple[str, ...] = (), kind: str = "mse", drop: tuple[str, ...] = (), note: str = "") -> pd.DataFrame:
    """Leave one house out. Finished folds are reused."""
    PRED.mkdir(parents=True, exist_ok=True)
    METRIC.mkdir(parents=True, exist_ok=True)
    started = time.time()
    missing = [h for h in HOUSES if not prediction_path(name, h).exists()]
    if missing:
        rows = {}
        names = None
        for h in HOUSES:
            x, labels = load_home(h, groups)
            if drop:
                x = x.drop(columns=list(drop))
            names = list(x.columns)
            rows[h] = training_rows(x, labels, h)
            del x, labels
        print(f"[{name}] {len(names)} features, {sum(len(r[1]) for r in rows.values()):,} sampled rows, {time.time() - started:.0f}s to load", flush=True)
        for house in missing:
            t0 = time.time()
            X = np.vstack([rows[h][0] for h in HOUSES if h != house])
            y = np.concatenate([rows[h][1] for h in HOUSES if h != house])
            w = np.concatenate([rows[h][2] for h in HOUSES if h != house])
            x_test, labels = load_home(house, groups)
            if drop:
                x_test = x_test.drop(columns=list(drop))
            pred = np.clip(fit_predict(kind, X, y, w, x_test.to_numpy(dtype="float32")), 0, None)
            labels.assign(pred=pred.astype("float32"))[["pred"]].to_parquet(prediction_path(name, house))
            print(f"[{name}] house {house}: {time.time() - t0:.0f}s", flush=True)
            del X, y, w, x_test
    table = score(name)
    summary = table.mean(numeric_only=True).drop("house")
    entry = {"experiment": name, "groups": "+".join(groups), "model": kind, "requires_future": bool(set(groups) & FUTURE_GROUPS), "uses_8s_input": bool(set(groups) & FINE_GROUPS),
             "seconds": round(time.time() - started), "note": note, **summary.round(4).to_dict()}
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    log = pd.concat([log[log.get("experiment") != name] if len(log) else log, pd.DataFrame([entry])], ignore_index=True)
    log.to_csv(LOG, index=False)
    return table


def home_metrics(y: np.ndarray, p: np.ndarray, index: pd.DatetimeIndex) -> dict:
    on, heat = y > 20, y >= 1000
    medium, tumble = (y >= 200) & (y < 1000), (y > 20) & (y < 200)
    frame = pd.DataFrame({"y": y, "p": p}, index=index)
    week = frame.resample("7D").sum() / 60_000
    count = frame["y"].resample("7D").count()
    week = week[count >= 0.8 * 10_080]
    return {"MAE": np.abs(y - p).mean(), "RMSE": np.sqrt(((y - p) ** 2).mean()), "R2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "MAE_ON": np.abs(y - p)[on].mean(), "MAE_OFF": np.abs(y - p)[~on].mean(), "MAE_heat": np.abs(y - p)[heat].mean() if heat.any() else np.nan,
            "pred_heat": p[heat].mean() if heat.any() else np.nan, "pred_medium": p[medium].mean() if medium.any() else np.nan, "pred_tumble": p[tumble].mean() if tumble.any() else np.nan,
            "off_energy_share": 100 * p[~on].sum() / max(p.sum(), 1e-9), "energy_signed": 100 * (p.sum() - y.sum()) / y.sum(), "energy_abs": 100 * abs(p.sum() - y.sum()) / y.sum(),
            "weekly_r": week["y"].corr(week["p"]), "weekly_abs": 100 * (week["p"] - week["y"]).abs().mean() / week["y"].mean()}


def score(name: str) -> pd.DataFrame:
    rows = []
    for house in HOUSES:
        labels = pd.read_parquet(base_path(house), columns=list(NON_FEATURES))
        if name.startswith("saved:"):
            method = name.split(":", 1)[1]
            saved = pd.read_parquet(SAVED / f"loho__{method}__h{house}.parquet", columns=["pred"])
            pred = saved["pred"].reindex(labels.index).to_numpy(float)
        else:
            pred = pd.read_parquet(prediction_path(name, house))["pred"].to_numpy(float)
        ok = labels["eval_ok"].to_numpy()
        assert not np.isnan(pred[ok]).any(), (name, house)
        rows.append({"house": house, **home_metrics(labels["wm"].to_numpy(float)[ok], pred[ok], labels.index[ok])})
    table = pd.DataFrame(rows)
    table.to_csv(METRIC / f"{name.replace(':', '_')}.csv", index=False)
    return table


def compare(name: str, reference: str) -> dict:
    """Paired by home, against a reference, using the keep rule fixed in the roadmap."""
    new, old = score(name).set_index("house"), score(reference).set_index("house")
    out = {"experiment": name, "reference": reference}
    for metric in ("RMSE", "MAE_ON", "MAE", "R2", "off_energy_share", "weekly_r", "weekly_abs", "energy_abs"):
        out[metric] = new[metric].mean()
        out[f"{metric}_ref"] = old[metric].mean()
    keep = {}
    for metric in ("RMSE", "MAE_ON"):
        diff = (new[metric] - old[metric]).to_numpy()
        means = np.random.default_rng(SEED).choice(diff, (20_000, len(diff))).mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        wins = int((diff < 0).sum())
        out.update({f"{metric}_diff": diff.mean(), f"{metric}_ci_low": low, f"{metric}_ci_high": high, f"{metric}_wins": wins, f"{metric}_p": stats.wilcoxon(diff).pvalue if np.any(diff) else 1.0})
        keep[metric] = diff.mean() < 0 and wins >= 8 and high < 0
    off_ok = new["off_energy_share"].mean() <= old["off_energy_share"].mean()
    out["off_energy_not_higher"] = bool(off_ok)
    out["KEEP"] = bool(keep["RMSE"] and keep["MAE_ON"] and off_ok)
    return out


def report(names: list[str], reference: str, title: str) -> pd.DataFrame:
    rows = pd.DataFrame([compare(n, reference) for n in names])
    columns = ["experiment", "RMSE", "RMSE_diff", "RMSE_ci_low", "RMSE_ci_high", "RMSE_wins", "MAE_ON", "MAE_ON_diff", "MAE_ON_ci_high", "MAE_ON_wins", "MAE", "R2", "off_energy_share", "weekly_r", "weekly_abs", "energy_abs", "KEEP"]
    print(f"\n=== {title} (reference {reference}) ===", flush=True)
    print(rows[columns].round(3).to_string(index=False), flush=True)
    path = OUT / "comparisons.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(previous):
        previous = previous[~((previous["experiment"].isin(names)) & (previous["reference"] == reference))]
    pd.concat([previous, rows], ignore_index=True).to_csv(path, index=False)
    return rows


def describe(frame: pd.DataFrame) -> str:
    return json.dumps(frame.round(3).to_dict(orient="list"))
