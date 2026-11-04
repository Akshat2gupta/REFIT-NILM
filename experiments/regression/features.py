"""Feature groups for the regression experiments. Every feature comes from the aggregate only.

Groups starting with ``fut`` or named ``ctx`` read minutes after t, so they suit batch analysis only.
``fine`` reads within-minute statistics from the ~8 s aggregate, so it goes beyond 1-minute input.
Each group is cached per home, aligned to the rows the frozen baseline uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import read_clean_csv  # noqa: E402
from src.feature_engineering import make_features  # noqa: E402
from src.preprocessing import clean_file_true_utc, prepare_clean_house  # noqa: E402

CACHE = HERE / "cache"
FINE_CACHE = ROOT / "experiments" / "cycle_detection" / "cache"
if not FINE_CACHE.exists():
    FINE_CACHE = CACHE
HOUSES = (2, 5, 6, 7, 10, 13, 15, 16, 17, 19, 20)
EXPLORATION, VALIDATION = (2, 6, 16, 19, 20), (5, 7, 10, 13, 15, 17)

HIGH_W, MIN_BLOCK = 1000, 3   # the same block definition as the cycle experiment
SMALL_W = 500                 # steps above this are large loads switching, not motor jitter
QUIET_W, QUIET_MIN = 30, 15


def _runs(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


def _ffill(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).ffill().to_numpy()


def _excess(s: pd.Series) -> pd.Series:
    return s - s.rolling(60, min_periods=30).min()


def long_windows(s: pd.Series) -> pd.DataFrame:
    """Past context longer than the baseline's 60 minutes."""
    out = {}
    for k in (90, 120, 180):
        r = s.rolling(k, min_periods=int(0.8 * k))
        low = r.min()
        out.update({f"mean_{k}": r.mean(), f"std_{k}": r.std(), f"max_{k}": r.max(), f"min_{k}": low, f"above_min_{k}": s - low})
    return pd.DataFrame(out)


def jitter(s: pd.Series) -> pd.DataFrame:
    """Past variability that a single large switching step cannot dominate."""
    d = s.diff().abs()
    small = d.where(d < SMALL_W)
    excess = _excess(s)
    band = ((excess >= 30) & (excess <= SMALL_W)).astype("float32").where(excess.notna())
    steps = ((d >= 100) & (d < 1000)).astype("float32").where(d.notna())
    out = {}
    for k in (15, 30, 60):
        out[f"jit_med_{k}"] = small.rolling(k, min_periods=k // 2).median()
        out[f"jit_mean_{k}"] = small.rolling(k, min_periods=k // 2).mean()
        out[f"steps_{k}"] = steps.rolling(k, min_periods=k // 2).sum()
        out[f"band_{k}"] = band.rolling(k, min_periods=k // 2).mean()
    # Motor variation against the home's background level.
    out["jit_to_level_60"] = out["jit_med_60"] / (s.rolling(60, min_periods=30).min() + 50)
    return pd.DataFrame(out)


def blocks_past(s: pd.Series) -> pd.DataFrame:
    """Sustained high blocks seen so far. A block counts only once its third minute has arrived."""
    high = (_excess(s) >= HIGH_W).fillna(False).to_numpy()
    n = len(s)
    starts, ends = _runs(high)
    keep = ends - starts + 1 >= MIN_BLOCK
    starts, ends = starts[keep], ends[keep]
    confirmed = starts + MIN_BLOCK - 1
    position = np.arange(n, dtype="float32")

    run = np.zeros(n, dtype="float32")
    all_s, all_e = _runs(high)
    for a, b in zip(all_s, all_e):
        run[a:b + 1] = np.arange(1, b - a + 2)

    last_start, last_gap, last_end, last_len = (np.full(n, np.nan, dtype="float32") for _ in range(4))
    last_start[confirmed] = starts
    last_gap[confirmed[1:]] = np.diff(starts)
    after = ends + 1 < n
    last_end[ends[after] + 1] = ends[after]
    last_len[ends[after] + 1] = (ends - starts + 1)[after]
    event = np.zeros(n, dtype="float32")
    event[confirmed] = 1
    event_s, high_s = pd.Series(event, index=s.index), pd.Series(high.astype("float32"), index=s.index)
    out = {"run_high": run, "since_start": position - _ffill(last_start), "since_end": position - _ffill(last_end),
           "last_len": _ffill(last_len), "last_gap": _ffill(last_gap)}
    frame = pd.DataFrame(out, index=s.index)
    for k in (60, 120, 180):
        frame[f"blocks_{k}"] = event_s.rolling(k, min_periods=1).sum()
        frame[f"heat_{k}"] = high_s.rolling(k, min_periods=1).sum()
    return frame


def clock(local_time: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(local_time)
    hour = t.dt.hour + t.dt.minute / 60
    return pd.DataFrame({"hour": hour, "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                         "dow": t.dt.dayofweek, "weekend": (t.dt.dayofweek >= 5).astype("float32")}, index=local_time.index)


def history(s: pd.Series) -> pd.DataFrame:
    """A home's own past: its base load, how often it runs long blocks, and at which hours."""
    hourly = s.resample("1h").mean()
    level = hourly.rolling(168, min_periods=24).quantile(0.1).shift(1).reindex(s.index, method="ffill")
    frame = pd.DataFrame({"level_7d": level, "above_level": s - level}, index=s.index)
    high = (_excess(s) >= HIGH_W).fillna(False).to_numpy()
    starts, ends = _runs(high)
    starts = starts[ends - starts + 1 >= MIN_BLOCK]
    event = pd.Series(0.0, index=s.index, dtype="float32")
    event.iloc[np.minimum(starts + MIN_BLOCK - 1, len(s) - 1)] = 1
    frame["blocks_per_day_30d"] = event.rolling(43_200, min_periods=10_080).sum() * 1440 / event.rolling(43_200, min_periods=10_080).count()
    counts = event.resample("1h").sum()
    same_hour = counts.groupby(counts.index.hour, group_keys=False).apply(lambda g: g.rolling(30, min_periods=7).mean().shift(1))
    frame["block_rate_this_hour_30d"] = same_hour.sort_index().reindex(s.index, method="ffill")
    return frame


def future_windows(s: pd.Series, leads: tuple[int, ...], windows: tuple[int, ...]) -> pd.DataFrame:
    """Readings after t. Requires future data."""
    out = {}
    for k in leads:
        out[f"fut_lag_{k}"] = s.shift(-k)
        out[f"fut_diff_{k}"] = s - s.shift(-k)
    ahead = s.shift(-1)[::-1]
    for k in windows:
        r = ahead.rolling(k, min_periods=int(0.8 * k))
        low = r.min()[::-1]
        out.update({f"fut_mean_{k}": r.mean()[::-1], f"fut_std_{k}": r.std()[::-1], f"fut_max_{k}": r.max()[::-1], f"fut_min_{k}": low, f"fut_above_min_{k}": s - low})
    return pd.DataFrame(out)


def cycle_context(s: pd.Series) -> pd.DataFrame:
    """What surrounds this minute: the block it belongs to, jitter after it, and the active stretch.

    Requires future data.
    """
    values = s.to_numpy(dtype="float64")
    n = len(values)
    excess = _excess(s).to_numpy()
    high = np.nan_to_num(excess) >= HIGH_W
    # Bridge one-minute dips, as the cycle experiment does.
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

    # The active stretch: how long since the home last sat quietly at base load, and until it next does.
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

    d_series = pd.Series(small, index=s.index)
    ahead = d_series.shift(-1)[::-1]
    for k in (15, 30, 60):
        frame[f"fut_jit_med_{k}"] = ahead.rolling(k, min_periods=k // 2).median()[::-1].astype("float32")
    return frame


def minute_fine(house: int) -> pd.DataFrame:
    """Statistics of the ~8 s aggregate within each minute. Built once from the official cleaned file."""
    path = FINE_CACHE / f"house{house}_fine_every1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = read_clean_csv(house, ["Unix", "Aggregate"])
    raw["utc"] = clean_file_true_utc(raw["Unix"])
    raw = raw.dropna(subset=["utc"]).sort_values("utc")
    power = raw["Aggregate"].to_numpy("float32")
    step = np.diff(power, prepend=np.nan)
    step[raw["utc"].diff().dt.total_seconds().to_numpy() > 30] = np.nan
    size = np.abs(step)
    frame = pd.DataFrame({"minute": raw["utc"].dt.floor("min").to_numpy(), "size": size, "small": np.where(size < SMALL_W, size, 0.0),
                          "step": ((size >= 100) & (size < 1000)).astype("float32"), "power": power})
    g = frame.groupby("minute", sort=True)
    out = pd.DataFrame({"abs_med": g["size"].median(), "tv_small": g["small"].mean(), "steps": g["step"].mean(),
                        "range_small": g["power"].quantile(0.9) - g["power"].quantile(0.1)}).astype("float32")
    FINE_CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def fine(house: int, index: pd.DatetimeIndex, future: bool) -> pd.DataFrame:
    """Within-minute statistics from the ~8 s aggregate, then rolled over the past or the future."""
    raw = minute_fine(house).reindex(index)
    out = {}
    for name in raw.columns:
        if future:
            ahead = raw[name].shift(-1)[::-1]
            for k in (15, 30, 60):
                out[f"fut_fine_{name}_{k}"] = ahead.rolling(k, min_periods=k // 2).mean()[::-1]
        else:
            out[f"fine_{name}"] = raw[name]
            for k in (15, 30, 60):
                out[f"fine_{name}_{k}"] = raw[name].rolling(k, min_periods=k // 2).mean()
    return pd.DataFrame(out)


def build_group(group: str, house: int, minute: pd.DataFrame) -> pd.DataFrame:
    s = minute["aggregate"].astype("float32")
    if group == "long":
        return long_windows(s)
    if group == "jitter":
        return jitter(s)
    if group == "blocks":
        return blocks_past(s)
    if group == "clock":
        return clock(minute["local_time"])
    if group == "history":
        return history(s)
    if group == "fut30":
        return future_windows(s, (1, 2, 5, 10, 15, 30), (5, 15, 30))
    if group == "fut60":
        return future_windows(s, (), (60,))
    if group == "fut120":
        return future_windows(s, (45, 60), (90, 120))
    if group == "ctx":
        return cycle_context(s)
    if group == "fine":
        return fine(house, s.index, future=False)
    if group == "futfine":
        return fine(house, s.index, future=True)
    raise KeyError(group)


GROUPS = ("long", "jitter", "blocks", "clock", "history", "fut30", "fut60", "fut120", "ctx", "fine", "futfine")
FUTURE_GROUPS = {"fut30", "fut60", "fut120", "ctx", "futfine"}
FINE_GROUPS = {"fine", "futfine"}


def base_path(house: int) -> Path:
    return CACHE / f"house{house}_base.parquet"


def ensure_house(house: int, groups: tuple[str, ...] = GROUPS) -> None:
    """Write the base table and any missing groups for a home, on the baseline's exact rows."""
    CACHE.mkdir(parents=True, exist_ok=True)
    wanted = [g for g in groups if not (CACHE / f"house{house}_{g}.parquet").exists()]
    if base_path(house).exists() and not wanted:
        return
    minute = prepare_clean_house(house)
    s = minute["aggregate"]
    base = make_features(s)
    names = list(base.columns)
    table = base.assign(wm=minute["wm"].astype("float32"), issue_frac=minute["issue_frac"].astype("float32"),
                        eval_ok=~(minute["aggregate_imputed"].astype(bool) | minute["wm_imputed"].astype(bool))).dropna(subset=names + ["wm"])
    if not base_path(house).exists():
        table.to_parquet(base_path(house))
    for group in wanted:
        frame = build_group(group, house, minute).reindex(table.index).astype("float32")
        frame.to_parquet(CACHE / f"house{house}_{group}.parquet")
        print(f"House {house}: {group}, {frame.shape[1]} features", flush=True)
