"""Read REFIT files and identify channels from the official README."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config as cfg

EXPECTED_RAW = ["Time", "Unix", "Aggregate", *cfg.APPLIANCES]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return path


def parse_readme(path: Path = cfg.README_PATH) -> dict[int, dict[int, str]]:
    """Return ``{house: {channel: description}}`` parsed from the cleaned-data README."""
    houses: dict[int, dict[int, str]] = {}
    current = None
    for line in _require(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        header = re.fullmatch(r"House (\d+)", line)
        channel = re.match(r"^(\d)\.(.*)$", line)
        if header:
            current = int(header.group(1))
            houses[current] = {}
        elif channel and current is not None:
            houses[current][int(channel.group(1))] = channel.group(2).strip().rstrip(",")
    return houses


def household_channels(path: Path = cfg.README_PATH) -> pd.DataFrame:
    """Table of washing-machine channels, confounding appliances and eligibility per house.

    A home is *eligible* for the analysis when it has exactly one washing-machine channel,
    no solar contamination of the aggregate, and no separately listed washer-dryer: a second
    machine with a near-identical signature would act as unlabelled "wrong" washing load.
    """
    rows = []
    for house, channels in parse_readme(path).items():
        find = lambda pattern: [k for k, v in channels.items() if re.search(pattern, v, re.I)]
        washing = find(r"washing machine")
        washer_dryer = find(r"washer dryer")
        solar = house in cfg.SOLAR_HOUSES
        reasons = []
        if not washing:
            reasons.append("no washing-machine channel")
        if len(washing) > 1:
            reasons.append("two washing machines")
        if washer_dryer:
            reasons.append("also has a washer-dryer")
        if solar:
            reasons.append("solar on aggregate")
        rows.append({
            "house": house,
            "wm_channel": washing[0] if len(washing) == 1 else None,
            "wm_channels_all": ",".join(map(str, washing)),
            "wm_description": "; ".join(channels[k] for k in washing),
            "washer_dryer_channel": ",".join(map(str, washer_dryer)),
            "tumble_dryer_channel": ",".join(map(str, find(r"tumble dryer"))),
            "eligible": not reasons,
            "exclusion_reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows).sort_values("house").reset_index(drop=True)


def eligible_houses() -> list[int]:
    """House numbers used for the washing-machine EDA and modelling."""
    table = household_channels()
    return table.loc[table.eligible, "house"].tolist()


def wm_channel(house: int) -> int:
    """Washing-machine channel number for an eligible house."""
    table = household_channels().set_index("house")
    channel = table.loc[house, "wm_channel"]
    if pd.isna(channel):
        raise ValueError(f"House {house} has no single washing-machine channel.")
    return int(channel)


def load_raw_part(part: int, house: int = 1) -> pd.DataFrame:
    """Load one raw CSV unchanged, preserving file order (which encodes clock changes)."""
    path = _require(cfg.DATA_RAW / f"RAW_House{house}_Part{part}.csv")
    df = pd.read_csv(path)
    missing = set(EXPECTED_RAW) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def read_clean_csv(house: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Load an official cleaned CSV."""
    return pd.read_csv(_require(cfg.DATA_CLEANED / f"CLEAN_House{house}.csv"), usecols=columns)
