"""Regression, ON/OFF, energy and event-level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as cfg
from .cycle_detection import detect_events


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
    }


def onoff_metrics(y: np.ndarray, pred: np.ndarray, pred_threshold: float,
                  actual_threshold: float = cfg.ACTUAL_ON_W) -> dict:
    """Minute-level detection. Actual ON is fixed. Predicted ON uses a calibrated threshold."""
    actual, predicted = y > actual_threshold, pred > pred_threshold
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"Precision": precision, "Recall": recall, "F1": f1}


def conditional_mae(y: np.ndarray, pred: np.ndarray, actual_threshold: float = cfg.ACTUAL_ON_W) -> dict:
    """MAE separately for ON and OFF minutes, because OFF minutes dominate the overall value."""
    on = y > actual_threshold
    return {
        "MAE_ON": float(np.abs(y[on] - pred[on]).mean()) if on.any() else np.nan,
        "MAE_OFF": float(np.abs(y[~on] - pred[~on]).mean()) if (~on).any() else np.nan,
    }


def energy_metrics(y: np.ndarray, pred: np.ndarray, step_min: int = 1) -> dict:
    """Total energy over the evaluated period. Energy (kWh) = sum(W) x minutes / 60,000."""
    actual, estimated = y.sum() * step_min / 60_000, pred.sum() * step_min / 60_000
    return {
        "Energy_true_kWh": float(actual),
        "Energy_pred_kWh": float(estimated),
        "Energy_error_pct": float(100 * (estimated - actual) / actual) if actual else np.nan,
    }


def _span_mask(events: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """True on every minute inside any event's span."""
    mask = np.zeros(len(grid), dtype=bool)
    if not events.empty:
        for a, b in zip(grid.get_indexer(events["start"]), grid.get_indexer(events["end"])):
            mask[a:b + 1] = True
    return mask


def event_metrics(y: pd.Series, pred: pd.Series, pred_threshold: float, min_overlap: float = 0.25,
                  pred_min_peak_w: float | None = None) -> dict:
    """Could a utility count cycles from this? Works at 1-minute resolution on one house.

    ``pred_min_peak_w`` is the peak a predicted event must reach. ``None`` keeps the rule as
    originally reported, where a predicted event only has to pass ``pred_threshold``. True
    events always need ``cfg.EVENT_MIN_PEAK_W``.

    ``Event_recall`` and ``Event_precision`` use one matching rule: a true event is found if
    its span overlaps any predicted event, and a predicted event is correct if its span
    overlaps any true event.

    The ``_original`` columns are the metrics as originally reported, kept so the change is
    visible. Recall there uses predicted-ON minutes rather than predicted events: a true event
    is found if ``min_overlap`` of its minutes are predicted ON, and a predicted event is
    correct if ``min_overlap`` of its minutes are truly ON.
    """
    grid = pd.date_range(y.index.min(), y.index.max(), freq="1min")
    y_full, pred_full = y.reindex(grid), pred.reindex(grid)
    true_events = detect_events(y_full)
    peak_rule = pred_threshold if pred_min_peak_w is None else pred_min_peak_w
    pred_events = detect_events(pred_full, active_w=pred_threshold, min_peak_w=peak_rule)
    predicted_on = (pred_full > pred_threshold).to_numpy()
    actual_on = (y_full > cfg.ACTUAL_ON_W).to_numpy()

    def overlap_share(events: pd.DataFrame, flags: np.ndarray) -> np.ndarray:
        if events.empty:
            return np.array([])
        start = grid.get_indexer(events["start"])
        end = grid.get_indexer(events["end"])
        return np.array([flags[a:b + 1].mean() for a, b in zip(start, end)])

    def share(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else np.nan

    in_true, in_pred = _span_mask(true_events, grid), _span_mask(pred_events, grid)
    return {
        "Events_true": len(true_events),
        "Events_pred": len(pred_events),
        "Event_recall": share(overlap_share(true_events, in_pred) > 0),
        "Event_precision": share(overlap_share(pred_events, in_true) > 0),
        "Event_recall_original": share(overlap_share(true_events, predicted_on) >= min_overlap),
        "Event_precision_original": share(overlap_share(pred_events, actual_on) >= min_overlap),
    }


def chance_event_metrics(y: pd.Series, pred: pd.Series, pred_threshold: float, shift: int,
                         pred_min_peak_w: float | None = None) -> dict:
    """Event scores against the truth moved ``shift`` rows away, the same time shift used for
    the minute-level chance test. Timing is broken, while power levels and frequency are kept."""
    shifted = pd.Series(np.roll(y.to_numpy(), shift), index=y.index)
    return event_metrics(shifted, pred, pred_threshold, pred_min_peak_w=pred_min_peak_w)


def event_energy_table(y: pd.Series, pred: pd.Series, pred_threshold: float) -> pd.DataFrame:
    """One row per true event: its measured energy, the energy predicted over the same minutes,
    and the share of its minutes the model called ON. Works on one house at 1-minute resolution.
    """
    grid = pd.date_range(y.index.min(), y.index.max(), freq="1min")
    y_full, pred_full = y.reindex(grid), pred.reindex(grid)
    events = detect_events(y_full)
    if events.empty:
        return events
    first, last = grid.get_indexer(events["start"]), grid.get_indexer(events["end"])
    predicted = pred_full.to_numpy()
    actual = y_full.to_numpy()
    events["pred_kwh"] = [np.nansum(predicted[a:b + 1]) / 60_000 for a, b in zip(first, last)]
    events["on_share"] = [np.nanmean(predicted[a:b + 1] > pred_threshold) for a, b in zip(first, last)]
    events["heater_min"] = [int(np.nansum(actual[a:b + 1] >= 1000)) for a, b in zip(first, last)]
    return events


def evaluate(table: pd.DataFrame, pred_threshold: float, step_min: int = 1,
             exclude_issues: bool = False, with_events: bool = True) -> dict:
    """All metrics for rows with ``wm``, ``pred``, ``eval_ok`` and ``issue_frac`` columns."""
    rows = table[table["eval_ok"]]
    if exclude_issues:
        rows = rows[rows["issue_frac"] == 0]
    y, pred = rows["wm"].to_numpy(float), rows["pred"].to_numpy(float)
    result = {"Minutes": len(rows), "ON_pct": float(100 * (y > cfg.ACTUAL_ON_W).mean())}
    result.update(regression_metrics(y, pred))
    result.update(onoff_metrics(y, pred, pred_threshold))
    result.update(conditional_mae(y, pred))
    result.update(energy_metrics(y, pred, step_min))
    if with_events and step_min == 1:
        result.update(event_metrics(rows["wm"], rows["pred"], pred_threshold))
    return result


def evaluate_by_house(predictions: pd.DataFrame, thresholds: dict, step_min: int = 1, **kwargs) -> pd.DataFrame:
    """One metric row per house, each at the threshold calibrated without that house's labels."""
    rows = [{"house": house, **evaluate(group, thresholds[house], step_min, **kwargs)}
            for house, group in predictions.groupby("house")]
    return pd.DataFrame(rows).set_index("house")
