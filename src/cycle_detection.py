"""Threshold-based washing-machine operating-event detection on 1-minute power."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def detect_events(power: pd.Series, active_w: float = cfg.ACTIVE_W,
                  max_gap_min: int = cfg.EVENT_MAX_GAP_MIN,
                  min_duration_min: int = cfg.EVENT_MIN_DURATION_MIN,
                  min_peak_w: float = cfg.EVENT_MIN_PEAK_W) -> pd.DataFrame:
    """Group active minutes into operating events.

    A minute is active when power exceeds ``active_w``. Active runs separated by at most
    ``max_gap_min`` measured low-power minutes join into one event. A missing minute always
    ends an event, so an event never spans an outage. Events shorter than
    ``min_duration_min`` or peaking below ``min_peak_w`` are discarded.

    These events are approximations: they are not verified physical wash-cycle boundaries.

    ``power`` must be on a regular 1-minute index. Returns one row per event with ``start``
    and ``end`` (index labels of the first and last active minute), ``duration_min``,
    ``peak_w``, ``mean_w``, ``energy_kwh`` and ``incomplete``. An event is incomplete when
    missing data or the end of the record lies within ``max_gap_min`` of either end, because
    the machine could have kept running unseen. Leave these out of per-cycle statistics.
    """
    columns = ["start", "end", "duration_min", "peak_w", "mean_w", "energy_kwh", "incomplete"]
    values = power.to_numpy(dtype=float)
    active_idx = np.flatnonzero(np.nan_to_num(values) > active_w)
    if len(active_idx) == 0:
        return pd.DataFrame(columns=columns)

    missing_before = np.cumsum(np.isnan(values))
    gap = np.diff(active_idx)
    has_missing = missing_before[active_idx[1:]] != missing_before[active_idx[:-1]]
    breaks = np.flatnonzero((gap - 1 > max_gap_min) | has_missing)
    starts = active_idx[np.r_[0, breaks + 1]]
    ends = active_idx[np.r_[breaks, len(active_idx) - 1]]

    rows = []
    for start, end in zip(starts, ends):
        duration = end - start + 1
        segment = values[start:end + 1]
        peak = np.nanmax(segment)
        if duration < min_duration_min or peak < min_peak_w:
            continue
        before = values[max(start - max_gap_min, 0):start]
        after = values[end + 1:end + 1 + max_gap_min]
        incomplete = (len(before) < max_gap_min or len(after) < max_gap_min
                      or bool(np.isnan(before).any()) or bool(np.isnan(after).any()))
        rows.append((power.index[start], power.index[end], duration, peak,
                     np.nanmean(segment), np.nansum(segment) / 60_000, incomplete))
    return pd.DataFrame(rows, columns=columns)


def sensitivity_grid(power: pd.Series, active=(10, 20, 50), gaps=(2, 5, 15, 30),
                     durations=(5, 10, 20)) -> pd.DataFrame:
    """Event counts and durations when one rule parameter moves away from the default."""
    settings = [dict(active_w=a) for a in active] + [dict(max_gap_min=g) for g in gaps] \
        + [dict(min_duration_min=d) for d in durations]
    rows = []
    for setting in settings:
        events = detect_events(power, **setting)
        name, value = next(iter(setting.items()))
        rows.append({
            "parameter": name, "value": value, "events": len(events),
            "median_duration_min": events["duration_min"].median() if len(events) else np.nan,
            "median_energy_kwh": events["energy_kwh"].median() if len(events) else np.nan,
        })
    return pd.DataFrame(rows)
