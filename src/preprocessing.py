"""Cleaning pipelines: raw House 1, plus 1-minute preparation of the official cleaned files."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import load_raw_part, read_clean_csv, wm_channel

CHANNELS = ["Aggregate", *cfg.APPLIANCES]


# ---------------------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------------------
def fill_short_gaps(series: pd.Series, max_gap: int = cfg.MAX_INTERPOLATION_GAP_MIN):
    """Linearly interpolate NaN runs of at most ``max_gap`` samples.

    Unlike ``interpolate(limit=n)``, longer gaps stay completely empty rather than having
    their first ``n`` samples filled. Returns ``(filled, was_imputed)``.
    """
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    bounded = series.ffill().notna() & series.bfill().notna()
    fillable = missing & (run_length <= max_gap) & bounded
    interpolated = series.interpolate(method="linear", limit_area="inside")
    filled = series.where(~fillable, interpolated)
    return filled, fillable


def stuck_mask(values: np.ndarray, seconds: np.ndarray, minutes: float = cfg.STUCK_MINUTES,
               min_value: float | None = None) -> np.ndarray:
    """True where a reading stays exactly identical for at least ``minutes``.

    ``min_value`` exempts lower values, so a real OFF (0 W) or standby period is not flagged.
    NaNs break a run.
    """
    values = np.asarray(values, dtype=float)
    changed = np.flatnonzero((values[1:] != values[:-1]))
    starts = np.r_[0, changed + 1]
    ends = np.r_[changed + 1, len(values)]
    elapsed = seconds[ends - 1] - seconds[starts]
    flagged = elapsed >= minutes * 60
    if min_value is not None:
        flagged &= values[starts] > min_value
    mask = np.zeros(len(values), dtype=bool)
    for start, end in zip(starts[flagged], ends[flagged]):
        mask[start:end] = True
    return mask


def to_local_clock(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a naive UTC index to naive UK clock time."""
    return utc_index.tz_localize("UTC").tz_convert(cfg.LOCAL_TZ).tz_localize(None)


def as_published_clock(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Recreate the clock printed in the official cleaned files, for before/after checks."""
    local = to_local_clock(utc_index)
    return pd.DatetimeIndex(np.where(utc_index < cfg.CLEAN_CLOCK_SWITCH, utc_index, local))


# ---------------------------------------------------------------------------------------
# Raw House 1
# ---------------------------------------------------------------------------------------
def raw_quality_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Data-quality findings for one raw table, computed before anything is changed."""
    unix = df["Unix"].to_numpy()
    step = np.diff(unix)
    numeric = df[CHANNELS]
    findings = [
        ("Total records", len(df)),
        ("Unparseable Time values", int(pd.to_datetime(df["Time"], errors="coerce").isna().sum())),
        ("Start (raw clock)", df["Time"].iloc[0]),
        ("End (raw clock)", df["Time"].iloc[-1]),
        ("Backward time steps", int((step < 0).sum())),
        ("Duplicate Unix timestamps", int(df["Unix"].duplicated().sum())),
        ("Exact duplicate rows", int(df.duplicated().sum())),
        ("Duplicate timestamps with different aggregate",
         int((df.groupby("Unix")["Aggregate"].nunique() > 1).sum())),
        ("Missing values, all channels", int(numeric.isna().sum().sum())),
        ("Missing values, aggregate", int(numeric["Aggregate"].isna().sum())),
        ("Median sampling interval (s)", float(np.median(step[step > 0]))),
        ("99th percentile interval (s)", float(np.percentile(step[step > 0], 99))),
        ("Irregular intervals (over 30 s)", int((step > 30).sum())),
        ("Gaps over 2 min", int((step > 120).sum())),
        ("Gaps over 1 h", int((step > 3600).sum())),
        ("Negative readings", int((numeric < 0).sum().sum())),
        (f"Aggregate above {cfg.AGGREGATE_MAX_W} W", int((numeric["Aggregate"] > cfg.AGGREGATE_MAX_W).sum())),
        ("Aggregate equal to 0 W", int((numeric["Aggregate"] == 0).sum())),
        (f"Appliance readings above {cfg.APPLIANCE_MAX_W} W",
         int((numeric[cfg.APPLIANCES] > cfg.APPLIANCE_MAX_W).sum().sum())),
        ("Maximum aggregate (W)", float(numeric["Aggregate"].max())),
        ("Maximum appliance reading (W)", float(numeric[cfg.APPLIANCES].max().max())),
    ]
    return pd.DataFrame(findings, columns=["Check", label]).set_index("Check")


def raw_true_utc(df: pd.DataFrame, part: int) -> pd.Series:
    """Convert the raw ``Unix`` field to true UTC.

    Raw ``Unix`` is UK wall-clock time written as if it were UTC, so it jumps at clock changes.
    Part 1 keeps file order, so rows before the backward step belong to the earlier (BST)
    repeat of the autumn hour. Part 2 was exported sorted, which interleaves the repeated hour
    of 26 Oct 2014 beyond separation; those rows become NaT and are dropped.
    """
    wall = pd.to_datetime(df["Unix"], unit="s")
    if part == 1:
        back = np.flatnonzero(np.diff(df["Unix"].to_numpy()) < -60)
        split = back[0] + 1 if len(back) else len(df)
        ambiguous = np.arange(len(df)) < split
    else:
        ambiguous = "NaT"
    local = wall.dt.tz_localize(cfg.LOCAL_TZ, ambiguous=ambiguous, nonexistent="NaT")
    return local.dt.tz_convert("UTC").dt.tz_localize(None)


def clean_raw_house1(part1: pd.DataFrame | None = None, part2: pd.DataFrame | None = None,
                     save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the raw House 1 cleaning pipeline.

    Returns ``(minute_table, audit)``. ``minute_table`` is indexed by true UTC at 1-minute
    resolution. Each channel has a ``_imputed`` flag, and ``aggregate_outage`` marks minutes
    left empty because the gap was too long to fill. See DATA.md for the reasoning.
    """
    part1 = load_raw_part(1) if part1 is None else part1
    part2 = load_raw_part(2) if part2 is None else part2
    log: list[tuple[str, int]] = []

    # 1. Timestamps, then keep only Part 1 inside the Part 1 / Part 2 overlap.
    p1 = part1.assign(utc=raw_true_utc(part1, 1))
    p2 = part2.assign(utc=raw_true_utc(part2, 2))
    log.append(("Rows read (Part 1 + Part 2)", len(p1) + len(p2)))
    bad_clock = int(p1.utc.isna().sum() + p2.utc.isna().sum())
    p1, p2 = p1.dropna(subset=["utc"]), p2.dropna(subset=["utc"])
    log.append(("Dropped: unresolvable repeated clock hour", bad_clock))
    last_p1 = p1.utc.max()
    overlap = p2.utc <= last_p1
    log.append(("Dropped: Part 2 rows inside the overlap with Part 1", int(overlap.sum())))
    combined = pd.concat([p1, p2[~overlap]], ignore_index=True).sort_values("utc", kind="stable")

    # 2. Duplicates: same second, keep the last non-null reading of each channel.
    before = len(combined)
    combined = combined.groupby("utc", sort=True)[CHANNELS].last()
    log.append(("Merged: rows sharing a timestamp", before - len(combined)))
    seconds = (combined.index.astype("int64") // 10**9).to_numpy() if combined.index.dtype == "datetime64[ns]" \
        else (combined.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    seconds = np.asarray(seconds)

    # 3. Physically impossible readings become missing, never zero. No row is deleted.
    negative = combined[CHANNELS] < 0
    log.append(("Set missing: negative readings", int(negative.sum().sum())))
    combined[CHANNELS] = combined[CHANNELS].mask(negative)
    zero_agg = combined["Aggregate"] == 0
    log.append(("Set missing: aggregate exactly 0 W (a home always has some load)", int(zero_agg.sum())))
    high_agg = combined["Aggregate"] > cfg.AGGREGATE_MAX_W
    log.append((f"Set missing: aggregate above {cfg.AGGREGATE_MAX_W} W", int(high_agg.sum())))
    combined["Aggregate"] = combined["Aggregate"].mask(zero_agg | high_agg)
    high_iam = combined[cfg.APPLIANCES] > cfg.APPLIANCE_MAX_W
    log.append((f"Set missing: appliance above {cfg.APPLIANCE_MAX_W} W", int(high_iam.sum().sum())))
    combined[cfg.APPLIANCES] = combined[cfg.APPLIANCES].mask(high_iam)

    # 4. Change-only logging: hold each channel's last reading for at most 2 minutes.
    filled_by_hold = 0
    for channel in CHANNELS:
        values = combined[channel].to_numpy(dtype=float)
        has_value = ~np.isnan(values)
        last_index = np.maximum.accumulate(np.where(has_value, np.arange(len(values)), 0))
        age = seconds - seconds[last_index]
        hold = (~has_value) & has_value[last_index] & (age <= cfg.SENSOR_HOLD_LIMIT_S)
        values[hold] = values[last_index[hold]]
        filled_by_hold += int(hold.sum())
        combined[channel] = values
    log.append(("Sensor readings held forward (2 min limit)", filled_by_hold))

    # 5. Frozen aggregate (identical for 30+ min) is an outage in disguise.
    frozen = stuck_mask(combined["Aggregate"].to_numpy(), seconds)
    log.append(("Set missing: frozen aggregate samples", int(frozen.sum())))
    combined.loc[frozen, "Aggregate"] = np.nan

    # 6. 1-minute mean: mean power over a minute is what preserves its energy.
    minute = combined[CHANNELS].resample("1min").mean()
    log.append(("Minutes in the study window", len(minute)))
    log.append(("Minutes with no aggregate before gap filling", int(minute["Aggregate"].isna().sum())))

    # 7. Bridge only short gaps; flag every imputed value; leave outages empty.
    output = {}
    imputed_counts = {}
    for channel in CHANNELS:
        filled, flag = fill_short_gaps(minute[channel])
        output[channel] = filled
        output[f"{channel}_imputed"] = flag
        imputed_counts[channel] = int(flag.sum())
    table = pd.DataFrame(output)
    table["aggregate_outage"] = table["Aggregate"].isna()
    table["local_time"] = to_local_clock(table.index)
    table.index.name = "utc"
    log.append(("Minutes of aggregate interpolated", imputed_counts["Aggregate"]))
    log.append(("Minutes of aggregate left empty (outage)", int(table["aggregate_outage"].sum())))
    audit = pd.DataFrame(log, columns=["Step", "Count"]).set_index("Step")

    if save:
        cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cfg.DATA_PROCESSED / "house1_raw_cleaned_1min.parquet")
    return table, audit


def raw_to_minute_uncleaned(df: pd.DataFrame, channel: str) -> pd.Series:
    """Naive 1-minute mean of a raw channel on its raw clock, for 'before cleaning' plots."""
    wall = pd.to_datetime(df["Unix"], unit="s")
    return pd.Series(df[channel].to_numpy(dtype=float), index=wall).sort_index().resample("1min").mean()


# ---------------------------------------------------------------------------------------
# Official cleaned files
# ---------------------------------------------------------------------------------------
def clean_file_true_utc(unix: pd.Series) -> pd.Series:
    """Put an official cleaned file on one UTC clock.

    Before ``CLEAN_CLOCK_SWITCH`` the printed time is already UTC. From then on it is UK
    wall-clock time. The repeated hour of 26 Oct 2014 cannot be separated and becomes NaT.
    """
    wall = pd.to_datetime(unix, unit="s")
    after = wall >= cfg.CLEAN_CLOCK_SWITCH
    utc = wall.copy()
    utc[after] = (wall[after].dt.tz_localize(cfg.LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
                  .dt.tz_convert("UTC").dt.tz_localize(None))
    # The hour after the switch overlaps the last rows before it. Keep the earlier rows.
    last_before = utc[~after].max()
    if pd.notna(last_before):
        utc[after & (utc <= last_before)] = pd.NaT
    return utc


def prepare_clean_house(house: int, use_cache: bool = True) -> pd.DataFrame:
    """1-minute aggregate and washing-machine table for one official cleaned house.

    Columns: ``aggregate``, ``wm`` (W), ``issue_frac``, ``n_samples``, ``aggregate_imputed``,
    ``wm_imputed``, ``local_time``. Index: true UTC. Empty minutes stay NaN, never zero.
    """
    path = cfg.DATA_PROCESSED / f"house{house}_1min.parquet"
    audit_path = cfg.DATA_PROCESSED / f"house{house}_audit.json"
    if use_cache and path.exists() and audit_path.exists():
        return pd.read_parquet(path)

    channel = f"Appliance{wm_channel(house)}"
    raw = read_clean_csv(house, ["Unix", "Aggregate", channel, "Issues"]).rename(
        columns={"Aggregate": "aggregate", channel: "wm", "Issues": "issues"})
    audit = {"house": house, "wm_channel": channel, "rows_read": len(raw)}

    raw["utc"] = clean_file_true_utc(raw["Unix"])
    audit["rows_dropped_clock_ambiguous"] = int(raw["utc"].isna().sum())
    seconds = raw["Unix"].to_numpy()
    aggregate = raw["aggregate"].to_numpy(dtype=float)
    wm = raw["wm"].to_numpy(dtype=float)

    # Forward-filled outages show up as frozen values. Exempt OFF/standby on the appliance.
    agg_stuck = stuck_mask(aggregate, seconds)
    wm_stuck = stuck_mask(wm, seconds, min_value=cfg.ACTIVE_W)
    agg_out_of_range = (aggregate > cfg.AGGREGATE_MAX_W) | (aggregate <= 0)
    audit.update(
        rows_aggregate_frozen=int(agg_stuck.sum()),
        rows_wm_frozen=int(wm_stuck.sum()),
        rows_aggregate_out_of_range=int(agg_out_of_range.sum()),
        hours_aggregate_frozen=round(float(agg_stuck.sum()) * 7 / 3600, 1),
    )
    raw["aggregate"] = np.where(agg_stuck | agg_out_of_range, np.nan, aggregate)
    raw["wm"] = np.where(wm_stuck, np.nan, wm)

    frame = raw.dropna(subset=["utc"]).sort_values("utc", kind="stable").set_index("utc")
    grouped = frame[["aggregate", "wm", "issues"]].resample("1min")
    minute = grouped.mean().rename(columns={"issues": "issue_frac"})
    minute["n_samples"] = grouped["aggregate"].size().astype("int32")

    audit["minutes_total"] = len(minute)
    audit["minutes_no_aggregate"] = int(minute["aggregate"].isna().sum())
    audit["minutes_no_wm"] = int(minute["wm"].isna().sum())
    minute["aggregate"], minute["aggregate_imputed"] = fill_short_gaps(minute["aggregate"])
    minute["wm"], minute["wm_imputed"] = fill_short_gaps(minute["wm"])
    audit["minutes_aggregate_imputed"] = int(minute["aggregate_imputed"].sum())
    audit["minutes_wm_imputed"] = int(minute["wm_imputed"].sum())
    minute["local_time"] = to_local_clock(minute.index)
    minute.index.name = "utc"

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    minute.to_parquet(path)
    audit_path.write_text(json.dumps(audit, indent=1))
    return minute


def clean_house_audit(houses: list[int]) -> pd.DataFrame:
    """Collect the masking and imputation counts recorded by ``prepare_clean_house``."""
    rows = []
    for house in houses:
        prepare_clean_house(house)
        rows.append(json.loads((cfg.DATA_PROCESSED / f"house{house}_audit.json").read_text()))
    return pd.DataFrame(rows).set_index("house")
