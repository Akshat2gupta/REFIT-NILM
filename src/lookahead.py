"""Inputs and training for the final batch model: past features, future windows and cycle context.

Every input comes from the aggregate at 1-minute resolution. Unlike the past-only baseline in
``feature_engineering.py``, these features read minutes after t, so the model suits batch analysis such as
weekly reporting, not live prediction. Future aggregate values are not target leakage: they never contain
the washing machine's own meter, and the test home stays out of training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .feature_engineering import make_features
from .preprocessing import prepare_clean_house

CACHE = cfg.ROOT / "data" / "processed" / "final_model"
HIGH_W, MIN_BLOCK = 1000, 3   # a heating-like block: at least 1000 W above the recent low for 3 minutes
SMALL_W = 500                 # steps above this are large loads switching, not motor variation
QUIET_W, QUIET_MIN = 30, 15   # the home sits near its recent low for at least 15 minutes
KEEP_EVERY = 8                # share of quiet OFF minutes kept in training
HARD_EXCESS_W = 300
LABELS = ["wm", "issue_frac", "eval_ok", "house"]


def _runs(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


def future_windows(s: pd.Series, leads: tuple[int, ...], windows: tuple[int, ...]) -> pd.DataFrame:
    """Readings after t: single future values, and statistics over the next k minutes."""
    out = {}
    for k in leads:
        out[f"fut_lag_{k}"] = s.shift(-k)
        out[f"fut_diff_{k}"] = s - s.shift(-k)
    ahead = s.shift(-1)[::-1]
    for k in windows:
        r = ahead.rolling(k, min_periods=int(0.8 * k))
        low = r.min()[::-1]
        out.update({f"fut_mean_{k}": r.mean()[::-1], f"fut_std_{k}": r.std()[::-1], f"fut_max_{k}": r.max()[::-1],
                    f"fut_min_{k}": low, f"fut_above_min_{k}": s - low})
    return pd.DataFrame(out)


def cycle_context(s: pd.Series) -> pd.DataFrame:
    """What surrounds this minute: its heating-like block, nearby blocks, motor-like variation and the active stretch."""
    values = s.to_numpy(dtype="float64")
    n = len(values)
    excess = (s - s.rolling(60, min_periods=30).min()).to_numpy()
    high = np.nan_to_num(excess) >= HIGH_W
    # Bridge one-minute dips inside a block.
    s0, e0 = _runs(high)
    for a, b in zip(e0[:-1], s0[1:]):
        if b - a - 1 <= 1:
            high[a + 1:b] = True
    starts, ends = _runs(high)
    keep = ends - starts + 1 >= MIN_BLOCK
    starts, ends = starts[keep], ends[keep]

    d = np.abs(np.diff(values, prepend=np.nan))
    small = np.where(d < SMALL_W, d, np.nan)

    def med(a: int, b: int) -> float:
        part = small[max(a, 0):min(b, n)]
        part = part[~np.isnan(part)]
        return float(np.median(part)) if len(part) >= 5 else np.nan

    post = np.array([med(e + 1, e + 61) for e in ends], dtype="float32")
    pre = np.array([med(a - 30, a) for a in starts], dtype="float32")
    height = np.array([np.nanmedian(excess[a:b + 1]) for a, b in zip(starts, ends)], dtype="float32")
    length = (ends - starts + 1).astype("float32")

    position = np.arange(n)
    last = np.searchsorted(starts, position, side="right") - 1     # latest block starting at or before t
    upcoming = last + 1                                            # next block starting after t
    has_last = last >= 0
    has_next = upcoming < len(starts)
    in_block = np.zeros(n, dtype="float32")
    in_block[has_last] = (position[has_last] <= ends[last[has_last]]).astype("float32")

    def pick(array: np.ndarray, which: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype="float32")
        out[valid] = array[which[valid]]
        return out

    frame = pd.DataFrame({
        "ctx_in_block": in_block, "ctx_block_len": pick(length, last, has_last), "ctx_block_height": pick(height, last, has_last),
        "ctx_block_pos": np.where(has_last, position - pick(starts.astype("float32"), last, has_last), np.nan),
        "ctx_since_end": np.where(has_last, position - pick(ends.astype("float32"), last, has_last), np.nan),
        "ctx_post_jit": pick(post, last, has_last), "ctx_pre_jit": pick(pre, last, has_last),
        "ctx_to_next": np.where(has_next, pick(starts.astype("float32"), upcoming, has_next) - position, np.nan),
        "ctx_next_len": pick(length, upcoming, has_next), "ctx_next_post_jit": pick(post, upcoming, has_next), "ctx_next_pre_jit": pick(pre, upcoming, has_next),
    }, index=s.index, dtype="float32")

    event = np.zeros(n + 1, dtype="float64")
    event[starts + 1] = 1
    total = np.cumsum(event)
    for k in (60, 120, 180):
        frame[f"ctx_blocks_within_{k}"] = (total[np.minimum(position + k + 1, n)] - total[np.maximum(position - k, 0)]).astype("float32")

    quiet = np.nan_to_num(excess, nan=np.inf) < QUIET_W
    qs, qe = _runs(quiet)
    keep_q = qe - qs + 1 >= QUIET_MIN
    qs, qe = qs[keep_q], qe[keep_q]
    prior = np.searchsorted(qe, position, side="left") - 1
    following = np.searchsorted(qs, position, side="right")
    since_quiet = np.where(prior >= 0, position - qe[np.maximum(prior, 0)], np.nan)
    to_quiet = np.where(following < len(qs), qs[np.minimum(following, max(len(qs) - 1, 0))] - position, np.nan) if len(qs) else np.full(n, np.nan)
    frame["ctx_since_quiet"] = np.clip(since_quiet, 0, 720).astype("float32")
    frame["ctx_to_quiet"] = np.clip(to_quiet, 0, 720).astype("float32")
    frame["ctx_stretch"] = frame["ctx_since_quiet"] + frame["ctx_to_quiet"]

    ahead = pd.Series(small, index=s.index).shift(-1)[::-1]
    for k in (15, 30, 60):
        frame[f"fut_jit_med_{k}"] = ahead.rolling(k, min_periods=k // 2).median()[::-1].astype("float32")
    return frame


def build_house(house: int) -> pd.DataFrame:
    """One home's inputs and labels, on the same rows as the past-only baseline. The result is cached."""
    path = CACHE / f"house{house}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    minute = prepare_clean_house(house)
    s = minute["aggregate"]
    base = make_features(s)
    table = base.assign(wm=minute["wm"], issue_frac=minute["issue_frac"], eval_ok=~(minute["aggregate_imputed"].astype(bool) | minute["wm_imputed"].astype(bool))).dropna(subset=[*base.columns, "wm"])
    aggregate = s.astype("float32")
    extra = [future_windows(aggregate, (1, 2, 5, 10, 15, 30), (5, 15, 30)), future_windows(aggregate, (), (60,)), cycle_context(aggregate)]
    extra = pd.concat(extra, axis=1).reindex(table.index).astype("float32")
    table = pd.concat([table.astype({c: "float32" for c in [*base.columns, "wm", "issue_frac"]}), extra], axis=1).assign(house=house)
    table.to_parquet(path)
    return table


def feature_names(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in LABELS]


def training_rows(table: pd.DataFrame, house: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ON minutes and all minutes well above the home's recent low, plus a weighted 1-in-8 sample of the rest."""
    y = table["wm"].to_numpy(dtype="float32")
    hard = (y > 20) | (table["above_min_60"].to_numpy() >= HARD_EXCESS_W)
    rng = np.random.default_rng(cfg.RANDOM_SEED + house)
    keep = hard | (rng.random(len(y)) < 1 / KEEP_EVERY)
    weight = np.where(hard[keep], 1.0, float(KEEP_EVERY)).astype("float32")
    return table[feature_names(table)].to_numpy(dtype="float32")[keep], y[keep], weight
