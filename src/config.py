"""Central configuration: paths, physical limits and analysis thresholds.

Every threshold that affects a result lives here, so it can be changed in one place.
DATA.md explains the evidence behind each value.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEANED = ROOT / "data" / "cleaned"
DATA_PROCESSED = ROOT / "data" / "processed"
README_PATH = ROOT / "data" / "archives" / "cleaned" / "CLEAN_READ_ME_081116.txt"
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"
PREDICTIONS = ROOT / "outputs" / "predictions"
MODELS = ROOT / "outputs" / "models"

RANDOM_SEED = 42
LOCAL_TZ = "Europe/London"
APPLIANCES = [f"Appliance{i}" for i in range(1, 10)]

# --- Sensor rules -----------------------------------------------------------------
# REFIT logs on change, so a missing raw reading means "unchanged", for a short while.
# 2 minutes is the REFIT authors' own forward-fill limit (Murray et al., 2017).
SENSOR_HOLD_LIMIT_S = 120
# Short gaps are bridged by linear interpolation only if the WHOLE gap fits this limit.
MAX_INTERPOLATION_GAP_MIN = 5
# 100 A (the largest common UK domestic fuse) x 230 V. Above this is physically implausible.
AGGREGATE_MAX_W = 23_000
# The REFIT authors' own appliance monitor spike limit.
APPLIANCE_MAX_W = 4_000
# An identical reading held this long is a frozen sensor, not a real load.
STUCK_MINUTES = 30

# --- Clock handling for the official cleaned files --------------------------------
# Rows before this wall-clock time are true UTC (BST hour removed by the publishers).
# Rows from it onwards keep UK local wall-clock time, uncorrected. See DATA.md.
CLEAN_CLOCK_SWITCH = pd.Timestamp("2014-10-01 00:00:00")

# --- Household eligibility --------------------------------------------------------
# Solar generation contaminates the aggregate in these homes (Murray et al., 2017).
SOLAR_HOUSES = (3, 11, 21)

# --- Washing-machine activity and event detection (see notebook 02 sensitivity) ---
ACTIVE_W = 20               # standby sits well below this
EVENT_MAX_GAP_MIN = 15      # pauses shorter than this stay inside the same event
EVENT_MIN_DURATION_MIN = 10  # shorter events are noise or brief activations
EVENT_MIN_PEAK_W = 200      # every real cycle has a drum motor or heater stage above this

# --- Evaluation -------------------------------------------------------------------
ACTUAL_ON_W = ACTIVE_W      # ground-truth ON threshold (fixed)
PRED_THRESHOLD_GRID = (20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750)
TRAIN_FRACTION = 0.70
