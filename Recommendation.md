# Washing-machine insights from smart meters: recommendation

**Recommendation: pilot a weekly laundry-energy insight, checked against sub-meters first. The model is best suited to weekly reporting; live alerts, cycle counts, billing and claims about one home's change are outside its current scope.**

The model estimates washing-machine use from a home's total electricity, using 1-minute readings and looking at what happens before and after each minute. It follows weekly patterns well in most homes without any extra hardware. Results come from 11 UK homes (REFIT, 2013-2015), each held out of training in turn.

## What it can and cannot do

| Use | Verdict | Evidence |
|---|---|---|
| Weekly trend within a home | **Most promising** | Weekly r 0.78 (0.23-0.97 by home); past-only baseline 0.64 |
| Weekly energy estimate | Indicative only | 30% mean error |
| Power at each minute | Useful, not exact | RMSE 101 W (always off 161 W); R² 0.57; error while running 219 W |
| Finding when it runs | Moderate | Precision 0.49, recall 0.66, F1 0.53 |
| Counting wash cycles | Needs validation | 1.5 times the true count; event precision 0.57, recall 0.83 |
| Live use | Separate past-only model | The batch model needs later readings; past-only RMSE is 128-133 W |

## What to take from this

- **Do not judge by total energy.** The baseline's error (22%) beats the final model's (24%), yet 64% of its predicted energy falls when the machine is off, against 38%.
- **Target wash temperature, not timing.** Heating is 71% of washer energy in 15% of active minutes. Only 5 of 11 homes use more than a flat share at 17:00-21:00, so peak shifting is not a general opportunity.
- **Include low-heat cycles in the pilot.** They are the hardest to see, and may be common in homes already washing cooler.
- **Confirm the meter rate first.** It decides what is possible.

## The opportunity, and how to measure it

Show customers their weekly laundry trend, then test a lower-temperature tip. The pilot measures the saving, because the data does not record wash temperature.

| Step | What to do | Measure |
|---|---|---|
| 1. Validate | Sub-meter a sample of new homes | Weekly correlation and error, against targets the utility agrees in advance |
| 2. Test | Randomly assign homes to the insight and tip, or to a comparison group | Change in sub-metered energy per wash and per week, compared between groups |
| 3. Decide | Baseline 3-6 months; follow up across seasons | Roll out only if both steps pass |

## Limitations

- **Overlapping signatures:** similar loads cannot be fully separated. Dishwashers are the strongest false-alarm association.
- **Missing data:** availability is 75-93%, and some official zeros are really outages.
- **Household differences:** annual use varies 6 times, and energy error runs from -55% to +57% by home. Homes with a second washing load show more false alarms: precision 0.31 and 30% energy error (baseline 67%).
- **Evidence:** model development used 11 homes from one UK area, each held out in testing. On 4 other homes, not used to choose the model, RMSE was 102.5 W (baseline 134.5 W). New homes are the next test.
- **UK to India:** machines, cold-water washing, appliances and supply may differ, so collect local sub-metered data and retrain before use.

## With 15- or 30-minute data

Short peaks are averaged away, heating blocks become hard to see, cycle start and end times blur, and more appliances overlap in each reading. Power and cycle detail stop being usable: a past-only baseline falls to R² 0.03 and -0.07. Weekly tracking degrades more slowly (r 0.56 and 0.54, from 0.64), though this is unvalidated and the final model was not tested at these rates. Finer data helps in the other direction: ~8-second input, outside this brief, cut past-only RMSE from 128 W to 107 W.
