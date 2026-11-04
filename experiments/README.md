# Regression experiments

Experiments behind the final model. They are kept apart from the main pipeline in `src/`, and the selected model was moved into `src/lookahead.py` after they finished. Each predicts the baseline's exact rows, is scored on the same minutes, and uses leave one house out on the same 11 homes.

**Keep rule, fixed before the runs:** improve both RMSE and MAE when ON in at least 8 of 11 homes, with a 95% home-bootstrap interval below zero, without raising the share of predicted energy in OFF minutes.

The results for every run are in [`outputs/metrics/experiments_summary.csv`](../outputs/metrics/experiments_summary.csv).

## How to run

Run from `experiments/regression/`, after the data is downloaded and `scripts/run_experiments.py` has finished.

| Step | Command | Time on the laptop used |
|---|---|---|
| Baselines and control | `python e0_baselines.py` | About 5 min |
| Feature experiments | `python e1_e2_features.py` | About 1-2 h, more on first run because of feature caching |
| Leakage test | `python test_leakage.py` | Minutes |
| Controls | `python controls.py` | About 15 min |
| Losses and model structure | `python e3_losses.py` | About 2-3 h |
| Partial feature removal | `python e1b_ablation.py` | About 30 min |
| Summary table | `python summarise.py` | Under a minute |

**Seq2Point CNN:** it needs PyTorch, so it uses its own environment. Install `requirements-e5.txt` in a new virtual environment, run `e5_export_series.py` with the main environment, then `python e5_seq2point.py --track look --houses 16`. Each fold takes about 15-30 minutes.

## What was tested

| Group | Problem | Hypothesis | Result | Limit | Conclusion |
|---|---|---|---|---|---|
| **E0: control** | Sampling training rows must not change results | Sampled rows match the full-data baseline | RMSE 133.60 W against 133.41 W | One sampling rate | Safe reference |
| **E0: simple baselines** | Does the model beat simple alternatives? | Ridge and hour profile are weaker | RMSE 150.78 W and 160.54 W | Few alternatives | Rejected |
| **E1: future windows** | The past-only model cannot see what follows the heating | Later readings help | RMSE 125.44 W at 60 minutes; 125.24 W at 120 | Batch only | Real gain, flat beyond 60 minutes |
| **E1: cycle context** | A wash has a recognisable shape | Surrounding blocks and variation help further | RMSE 101.06 W, better in 11 of 11 homes | Batch only; chosen on these homes | **Selected** |
| **E2: past-only groups** | Live use has no future data | Longer history, jitter, blocks, clock or habits help | Best is jitter at 128.06 W; it fails the keep rule | Past-only is ambiguous at heating start | Best live option; no strong gain |
| **E3: losses and structure** | Squared error hedges under ambiguity | A different loss or a two-part model helps | No loss beat squared error on RMSE; hurdle had the lowest MAE of its set (15.87 W) | Poisson overflowed and was not a fair test | Rejected |
| **Controls** | Large gains need checking | Shifting features by 7 days removes real gains | Cycle context 126.12 W; ~8-second statistics 134.31 W | No control on `E2b_fine_look` | Gains tied to the current minute |
| **E5: Seq2Point CNN** | A network may learn the shape itself | A small CNN beats the trees | House 16: 136.54 W against 117.89 W | One home, untuned, 8 training homes | Not pursued |
| **E2b: ~8-second input** | Averaging to 1 minute may hide motor variation | Within-minute statistics help | RMSE 106.88 W past-only; 91.21 W with look-ahead | Outside the brief's 1-minute scope | Evidence for finer metering; not a final model |
| **Feature removal** | 86 features are a lot to explain | Some families are redundant | Without 5 block-shape features: 101.33 W | One of six runs finished | Not concluded |

## Not run

- **Stacking a cycle detector into the regression:** nested training adds complexity that would be hard to justify, and the brief favours a simple solution.
- **Remaining feature-removal runs:** stopped to prioritise submission.
- **Seq2Point CNN on all homes:** about 6 hours per version, with little sign it would change the decision.

## Files

| File | Purpose |
|---|---|
| `features.py` | All experimental feature groups, cached per home |
| `reglib.py` | Training, metrics, keep rule and experiment log |
| `test_leakage.py`, `controls.py` | Leakage tests, clock alignment and shift controls |
| `e0_*`, `e1_*`, `e1b_*`, `e3_*`, `e5_*` | One script per experiment group |
| `summarise.py` | Builds the summary table |

Predictions and caches are not committed, because the scripts rebuild them.
