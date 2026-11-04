"""Leakage tests for the new feature groups, on one home. Run before trusting any experiment.

1. Past-only groups: changing every reading after a cut leaves all values at or before the cut identical.
2. Future groups must change, which shows the test can fail and confirms their label.
3. No feature name refers to the target, and each group reads the aggregate only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FUTURE_GROUPS, build_group
from src.preprocessing import prepare_clean_house

HOUSE = 16
GROUPS = ("long", "jitter", "blocks", "clock", "history", "fut30", "fut60", "fut120", "ctx")


def main() -> None:
    minute = prepare_clean_house(HOUSE)
    cut = len(minute) // 2
    # Cut on an hour boundary, because the home's hourly history uses only finished hours.
    while minute.index[cut].minute != 59:
        cut += 1
    changed = minute.copy()
    rng = np.random.default_rng(1)
    changed.iloc[cut + 1:, changed.columns.get_loc("aggregate")] = (minute["aggregate"].iloc[cut + 1:] + rng.normal(0, 800, len(minute) - cut - 1)).clip(lower=1).astype("float32")
    wm_broken = minute.copy()
    wm_broken["wm"] = 0.0
    failed = 0
    for group in GROUPS:
        a, b = build_group(group, HOUSE, minute), build_group(group, HOUSE, changed)
        early_a, early_b = a.iloc[:cut + 1].to_numpy(dtype=float), b.iloc[:cut + 1].to_numpy(dtype=float)
        same = np.allclose(early_a, early_b, equal_nan=True, atol=1e-3)
        expected_same = group not in FUTURE_GROUPS
        ok = same == expected_same
        failed += not ok
        target_ok = not any(("wm" in c.lower().split("_")) for c in a.columns)
        independent = np.allclose(a.to_numpy(dtype=float), build_group(group, HOUSE, wm_broken).to_numpy(dtype=float), equal_nan=True)
        failed += not (target_ok and independent)
        print(f"{group:8s} {a.shape[1]:3d} features | past unchanged by future: {same} (expected {expected_same}) | target-free: {target_ok and independent} | {'PASS' if ok and target_ok and independent else 'FAIL'}", flush=True)
    print("ALL PASS" if not failed else f"{failed} FAILURES")


if __name__ == "__main__":
    main()
