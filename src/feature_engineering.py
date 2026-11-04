"""Features computed from the aggregate signal only.

The washing-machine signal and other appliance channels are never inputs. The frame is on a
regular time grid with gaps as NaN, and every window demands full coverage, so a lag or
rolling feature cannot silently cross an outage. Each house is processed separately, so no
window can cross a household boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import prepare_clean_house

JUMP_W = 200  # a step this large is an appliance switching, not background drift


def _causal_features(s: pd.Series, lags: tuple[int, ...], windows: tuple[int, ...], jump_window: int) -> dict:
    """Features at time t that use only readings at or before t. Sizes are in time steps."""
    features = {}
    for k in lags:
        features[f"lag_{k}"] = s.shift(k)
        features[f"diff_{k}"] = s - s.shift(k)
    for k in windows:
        rolling = s.rolling(k, min_periods=k)
        low = rolling.min()
        features[f"mean_{k}"] = rolling.mean()
        features[f"std_{k}"] = rolling.std()
        features[f"max_{k}"] = rolling.max()
        features[f"min_{k}"] = low
        features[f"above_min_{k}"] = s - low  # removes each home's own base load
    step = s.diff().abs()
    is_jump = (step > JUMP_W).astype("float32").where(step.notna())
    features[f"jumps_{jump_window}"] = is_jump.rolling(jump_window, min_periods=jump_window).sum()
    return features


def make_features(aggregate: pd.Series, step_min: int = 1) -> pd.DataFrame:
    """Build the feature table for one house, using only readings at or before each time step."""
    if step_min == 1:
        lags, windows, jump_window = (1, 2, 5, 10, 15, 30), (5, 15, 30, 60), 30
    else:  # coarse meters: the same idea, sized in meter readings
        lags, windows, jump_window = (1, 2, 3, 4), (2, 4, 8), 4
    s = aggregate.astype("float32")
    features = {"agg": s, **_causal_features(s, lags, windows, jump_window)}
    return pd.DataFrame(features).astype("float32")


def to_resolution(minute: pd.DataFrame, step_min: int) -> pd.DataFrame:
    """Average 1-minute data to a coarser meter interval, requiring 80% of the minutes present."""
    if step_min == 1:
        return minute
    rule = f"{step_min}min"
    required = int(np.ceil(0.8 * step_min))
    resampled = minute[["aggregate", "wm", "issue_frac"]].resample(rule)
    means = resampled.mean()
    counts = resampled.count()
    means["aggregate"] = means["aggregate"].where(counts["aggregate"] >= required)
    means["wm"] = means["wm"].where(counts["wm"] >= required)
    means["aggregate_imputed"] = False
    means["wm_imputed"] = False
    return means


def build_dataset(houses: list[int], step_min: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Stack complete rows from each house. Returns ``(table, feature_names)``.

    ``eval_ok`` marks rows whose aggregate and target at time t were measured rather than
    imputed. It checks time t only, so lags and rolling windows behind t can still contain
    interpolated aggregate values. Metrics are computed only on these rows.
    """
    frames = []
    for house in houses:
        minute = to_resolution(prepare_clean_house(house), step_min)
        features = make_features(minute["aggregate"], step_min)
        names = list(features.columns)
        table = features.assign(
            wm=minute["wm"].astype("float32"),
            issue_frac=minute["issue_frac"].astype("float32"),
            eval_ok=~(minute["aggregate_imputed"].astype(bool) | minute["wm_imputed"].astype(bool)),
            house=house,
        ).dropna(subset=names + ["wm"])
        frames.append(table)
    return pd.concat(frames), names
