"""Evaluate the final batch model against the past-only baseline and always-off, and draw its figures.

Needs the predictions from ``scripts/run_experiments.py`` and ``scripts/run_final_model.py``.
Usage: ``python scripts/evaluate_final_model.py``
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from src import config as cfg  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.data_loader import parse_readme, read_clean_csv, wm_channel  # noqa: E402
from src.evaluation import evaluate_by_house, event_energy_table, event_metrics  # noqa: E402
from src.modeling import load_predictions  # noqa: E402
from src.preprocessing import clean_file_true_utc, prepare_clean_house  # noqa: E402

FINAL, BASELINE, OFF = "lookahead_ctx", "hist_gradient_boosting", "always_off"
LABELS = {FINAL: "Final model", BASELINE: "Past-only baseline", OFF: "Always off"}
EXAMPLE_HOUSE, FAILURES, ATTRIBUTION_HOUSES = 16, (19, 5), (6, 20)
BANDS = [("Heating, 1000+ W", 1000, np.inf), ("200-1000 W", 200, 1000), ("Tumbling, 20-200 W", 20, 200), ("OFF, 20 W or less", -np.inf, 20)]


def weekly_scores(table: pd.DataFrame) -> dict:
    rows = table[table["eval_ok"]]
    week = rows[["wm", "pred"]].resample("7D").sum() / 60_000
    week = week[rows["wm"].resample("7D").count() >= 0.8 * 10_080]
    return {"Weekly_r": week["wm"].corr(week["pred"]), "Weekly_abs_error_pct": 100 * (week["pred"] - week["wm"]).abs().mean() / week["wm"].mean()}


def score_method(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions, thresholds = load_predictions("loho", name)
    table = evaluate_by_house(predictions, thresholds)
    extra = []
    for house, group in predictions.groupby("house"):
        rows = group[group["eval_ok"]]
        off, heat = rows["wm"] <= cfg.ACTIVE_W, rows["wm"] >= 1000
        extra.append({"house": house, **weekly_scores(group), "Off_energy_share_pct": 100 * rows.loc[off, "pred"].sum() / max(rows["pred"].sum(), 1e-9),
                      "Mean_pred_heating_W": rows.loc[heat, "pred"].mean(), "Energy_abs_error_pct": abs(rows["pred"].sum() - rows["wm"].sum()) / rows["wm"].sum() * 100})
    return table.join(pd.DataFrame(extra).set_index("house")), predictions, thresholds


def paired_tests(by_method: dict) -> pd.DataFrame:
    final, rows = by_method[FINAL], []
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    for other in (BASELINE, OFF):
        for metric in ("RMSE", "MAE", "MAE_ON", "R2"):
            diff = (final[metric] - by_method[other][metric]).to_numpy()
            means = rng.choice(diff, (20_000, len(diff))).mean(axis=1)
            better = int(((diff > 0) if metric == "R2" else (diff < 0)).sum())
            rows.append({"Compared with": LABELS[other], "Metric": metric, "Final model": final[metric].mean(), "Other": by_method[other][metric].mean(),
                         "Mean difference": diff.mean(), "CI low": np.percentile(means, 2.5), "CI high": np.percentile(means, 97.5),
                         "Homes better": better, "Homes": len(diff), "Wilcoxon p": stats.wilcoxon(diff).pvalue})
    return pd.DataFrame(rows)


def energy_by_state(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[predictions["eval_ok"]]
    out = [{"Machine state": label, "Measured kWh": rows.loc[(rows["wm"] > low) & (rows["wm"] <= high) if high == 20 else (rows["wm"] >= low) & (rows["wm"] < high), "wm"].sum() / 60_000,
            "Predicted kWh": rows.loc[(rows["wm"] > low) & (rows["wm"] <= high) if high == 20 else (rows["wm"] >= low) & (rows["wm"] < high), "pred"].sum() / 60_000} for label, low, high in BANDS]
    return pd.DataFrame(out)


def attribute_false_alarms(house: int, predictions: pd.DataFrame, threshold: float, active_w: int = 100) -> pd.DataFrame:
    """Which monitored appliance was running while the model wrongly said the machine was on? Used for analysis only."""
    names = parse_readme()[house]
    raw = read_clean_csv(house, ["Unix", *cfg.APPLIANCES])
    raw["utc"] = clean_file_true_utc(raw["Unix"])
    raw = raw.dropna(subset=["utc"]).set_index("utc").drop(columns="Unix").astype("float32")
    channels = raw[cfg.APPLIANCES].clip(upper=cfg.APPLIANCE_MAX_W).resample("1min").mean()
    del raw
    group = predictions[(predictions["house"] == house) & predictions["eval_ok"]]
    channels = channels.reindex(group.index)
    off = group["wm"] <= cfg.ACTIVE_W
    weight = group.loc[off & (group["pred"] > threshold), "pred"]
    rows, any_running = [], pd.Series(False, index=weight.index)
    washer = f"Appliance{wm_channel(house)}"
    for channel in cfg.APPLIANCES:
        if channel == washer:
            continue
        running = channels[channel] > active_w
        here = running.reindex(weight.index).fillna(False).astype(bool)
        any_running |= here
        share = 100 * weight[here].sum() / weight.sum()
        chance = 100 * running[off].mean()
        rows.append({"House": house, "Appliance": names[int(channel[-1])].split(",")[0], "Share of false-alarm energy %": share, "Running in OFF minutes %": chance, "Above chance": share / max(chance, 0.01)})
    table = pd.DataFrame(rows).sort_values("Share of false-alarm energy %", ascending=False)
    print(f"House {house}: false-alarm energy {weight.sum() / 60_000:.0f} kWh, measured machine energy {group['wm'].sum() / 60_000:.0f} kWh, no monitored appliance above {active_w} W in {100 * weight[~any_running].sum() / weight.sum():.1f}% of false-alarm energy", flush=True)
    return table


def complete_events(predictions: pd.DataFrame, house: int, threshold: float) -> pd.DataFrame:
    group = predictions[(predictions["house"] == house) & predictions["eval_ok"]]
    events = event_energy_table(group["wm"], group["pred"], threshold)
    events = events[~events["incomplete"]].copy()
    events["error"] = (events["pred_kwh"] - events["energy_kwh"]).abs() / events["energy_kwh"]
    return events


def median_event(events: pd.DataFrame) -> pd.Series:
    return events.iloc[(events["error"] - events["error"].median()).abs().argsort().iloc[0]]


def plot_example(final: pd.DataFrame, base: pd.DataFrame, thresholds: dict) -> None:
    """The same day as the baseline's example figure, with both models."""
    middle = median_event(complete_events(base, EXAMPLE_HOUSE, thresholds[EXAMPLE_HOUSE]))
    start = middle["start"].floor("D")
    minute = prepare_clean_house(EXAMPLE_HOUSE)
    day = minute[(minute.index >= start) & (minute.index < start + pd.Timedelta(days=1))]
    pick = lambda table: table[table["house"] == EXAMPLE_HOUSE]["pred"].reindex(day.index)
    print(f"Example day: House {EXAMPLE_HOUSE}, {start:%d %b %Y}", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    axes[0].plot(day["local_time"], day["wm"], color=viz.ORANGE, label="Measured washing machine")
    axes[0].plot(day["local_time"], pick(final), color=viz.AQUA, label="Final model")
    axes[0].plot(day["local_time"], pick(base), color=viz.MUTED, linewidth=1, label="Past-only baseline")
    axes[0].set_title(f"House {EXAMPLE_HOUSE}, held out of training: the final model reaches more of the heating, but false alarms remain")
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(loc="upper right", ncols=3)
    axes[1].plot(day["local_time"], day["aggregate"], color=viz.BLUE, linewidth=1, label="Aggregate")
    axes[1].plot(day["local_time"], pick(final), color=viz.AQUA, label="Final model")
    axes[1].set_title("The same day with the aggregate the model sees")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("UK local time")
    axes[1].legend(loc="upper right", ncols=2)
    viz.format_time_axis(axes[1], "%d %b %H:%M")
    fig.tight_layout()
    viz.save(fig, "04_A_actual_vs_predicted_final_model")
    plt.close(fig)


def plot_scatter(final: pd.DataFrame, base: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    cmap = sns.light_palette(viz.BLUE, as_cmap=True)
    for ax, (table, title) in zip(axes, ((base, "Past-only baseline"), (final, "Final model"))):
        rows = table[table["eval_ok"]]
        rows = rows.sample(min(1_500_000, len(rows)), random_state=cfg.RANDOM_SEED)
        image = ax.hexbin(rows["wm"], rows["pred"], gridsize=45, bins="log", cmap=cmap, mincnt=1, extent=(0, 2700, 0, 2700))
        ax.plot([0, 2700], [0, 2700], color=viz.INK_2, linewidth=1, label="Perfect prediction")
        ax.set_title(f"{title}: measured versus predicted")
        ax.set_xlabel("Measured (W)")
        ax.legend(loc="upper left")
        ax.grid(False)
    axes[0].set_ylabel("Predicted (W)")
    fig.colorbar(image, ax=axes, label="Minutes (log scale)", fraction=0.025)
    viz.save(fig, "04_B_scatter_final_model")
    plt.close(fig)


def plot_by_home(by_method: dict) -> None:
    order = by_method[FINAL]["RMSE"].sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(order))
    for ax, metric, title in ((axes[0], "RMSE", "RMSE (W): lower is better"), (axes[1], "Weekly_r", "Weekly tracking r: higher is better")):
        base, final = by_method[BASELINE].loc[order, metric], by_method[FINAL].loc[order, metric]
        ax.hlines(y, base, final, color=viz.AXIS, linewidth=2)
        ax.scatter(base, y, color=viz.MUTED, s=42, label="Past-only baseline", zorder=3)
        ax.scatter(final, y, color=viz.AQUA, s=42, label="Final model", zorder=3)
        ax.set_title(title)
        ax.legend(loc="best")
    axes[0].set_yticks(y, [str(h) for h in order])
    axes[0].set_ylabel("House")
    fig.tight_layout()
    viz.save(fig, "04_C_results_by_home_final_model")
    plt.close(fig)


def plot_failures(final: pd.DataFrame, thresholds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, house in zip(axes, FAILURES):
        event = median_event(complete_events(final, house, thresholds[house]))
        minute = prepare_clean_house(house)
        start, end = event["start"] - pd.Timedelta(minutes=30), event["end"] + pd.Timedelta(minutes=30)
        part = minute[(minute.index >= start) & (minute.index <= end)]
        pred = final[final["house"] == house]["pred"].reindex(part.index)
        ax.plot(part["local_time"], part["aggregate"], color=viz.BLUE, linewidth=1, label="Aggregate")
        ax.plot(part["local_time"], part["wm"], color=viz.ORANGE, label="Measured")
        ax.plot(part["local_time"], pred, color=viz.AQUA, label="Final model")
        ax.set_title(f"House {house}, typical event: {event['energy_kwh']:.2f} kWh measured, {event['pred_kwh']:.2f} predicted")
        ax.set_ylabel("Power (W)")
        ax.set_xlabel("UK local time")
        ax.legend(loc="upper right")
        viz.format_time_axis(ax, "%H:%M")
        del minute
    fig.tight_layout()
    viz.save(fig, "04_D_failure_cases_final_model")
    plt.close(fig)


def event_rules(final: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Event counts when predicted events must reach a stated peak. True events always need 200 W."""
    rows = []
    for rule, peak in (("Calibrated threshold only", None), ("Peak at least 200 W, the same as true events", 200), ("Peak at least 500 W (diagnostic)", 500)):
        scores = pd.DataFrame([event_metrics(group.loc[group["eval_ok"], "wm"], group.loc[group["eval_ok"], "pred"], thresholds[house], pred_min_peak_w=peak) for house, group in final.groupby("house")])
        rows.append({"Rule": rule, "Predicted / true": scores["Events_pred"].mean() / scores["Events_true"].mean(), "Event_precision": scores["Event_precision"].mean(), "Event_recall": scores["Event_recall"].mean()})
    table = pd.DataFrame(rows)
    table.round(3).to_csv(cfg.METRICS / "final_model_event_rules.csv", index=False)
    return table


def reproduction_check(final: pd.DataFrame) -> None:
    folder = cfg.ROOT / "experiments" / "regression" / "outputs" / "predictions"
    diffs = {}
    for house, group in final.groupby("house"):
        path = folder / f"E1_L60_ctx__h{house}.parquet"
        if path.exists():
            diffs[house] = float(np.abs(pd.read_parquet(path)["pred"].to_numpy() - group["pred"].to_numpy()).max())
    if diffs:
        print("Largest difference from the experiment's predictions, by home (W):", {k: round(v, 4) for k, v in diffs.items()}, flush=True)


def main() -> None:
    viz.apply_style()
    by_method, kept, thresholds = {}, {}, {}
    for name in (FINAL, BASELINE, OFF):
        by_method[name], predictions, thresholds[name] = score_method(name)
        if name != OFF:
            kept[name] = predictions
        del predictions
        gc.collect()
        print(f"Scored {name}", flush=True)
    reproduction_check(kept[FINAL])
    by_method[FINAL].round(4).to_csv(cfg.METRICS / "final_model_by_house.csv")
    columns = ["MAE", "RMSE", "R2", "MAE_ON", "MAE_OFF", "Precision", "Recall", "F1", "Energy_error_pct", "Energy_abs_error_pct", "Off_energy_share_pct", "Mean_pred_heating_W", "Weekly_r", "Weekly_abs_error_pct", "Event_recall", "Event_precision", "Events_true", "Events_pred"]
    summary = pd.DataFrame({LABELS[n]: t[columns].mean() for n, t in by_method.items()}).T
    summary.index.name = "Method"
    summary.round(4).to_csv(cfg.METRICS / "final_model_summary.csv")
    tests = paired_tests(by_method)
    tests.round(4).to_csv(cfg.METRICS / "final_model_vs_baselines.csv", index=False)
    state = energy_by_state(kept[FINAL]).merge(energy_by_state(kept[BASELINE]), on="Machine state", suffixes=("", " (baseline)"))
    state.round(1).to_csv(cfg.METRICS / "final_model_energy_by_state.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summary.round(3).T.to_string(), "\n\n", tests.round(3).to_string(index=False), "\n\n", state.round(1).to_string(index=False), "\n", flush=True)
    print(by_method[FINAL][["RMSE", "R2", "MAE_ON", "F1", "Energy_error_pct", "Weekly_r"]].round(2).to_string(), flush=True)
    print(event_rules(kept[FINAL], thresholds[FINAL]).round(3).to_string(index=False), flush=True)
    plot_example(kept[FINAL], kept[BASELINE], thresholds[BASELINE])
    plot_scatter(kept[FINAL], kept[BASELINE])
    plot_by_home(by_method)
    plot_failures(kept[FINAL], thresholds[FINAL])
    del kept[BASELINE]
    gc.collect()
    attribution = [attribute_false_alarms(h, kept[FINAL], thresholds[FINAL][h]) for h in ATTRIBUTION_HOUSES]
    attribution = pd.concat(attribution)
    attribution.round(2).to_csv(cfg.METRICS / "final_model_false_alarm_attribution.csv", index=False)
    print(attribution.round(2).groupby("House").head(4).to_string(index=False), flush=True)
    print("Final model evaluation finished.", flush=True)


if __name__ == "__main__":
    main()
