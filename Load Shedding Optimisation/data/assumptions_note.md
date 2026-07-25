# Dataset Assumptions Note (Role A)

## Zone roster
20 zones over a 48-hour (2-day) scheduling horizon, in five categories:

| Category | # Zones | Criticality | Justification |
|---|---|---|---|
| Hospital | 2 | 10 | Life-safety equipment; zero tolerance for outage |
| Cold storage / water pumping | 2 | 7 | Spoilage / public-health risk (vaccines, water supply) |
| School / institutional | 4 | 5 | Disrupts essential public services; low life-safety risk |
| Commercial | 6 | 3 | Economic loss only; no life-safety or spoilage risk |
| Residential (idle) | 6 | 1 | Lowest urgency; most tolerant of scheduled outages |

Counts are within the plan's recommended 15–25 zone range, weighted toward
lower-criticality categories to reflect a realistic grid mix (most zones
on any real feeder are ordinary residential/commercial load, not
hospitals).

## Demand curves
Each zone gets a per-hour demand curve shaped by its category (e.g.
hospitals run a flat high baseline around the clock; residential zones
show the classic 6–9am / 6–10pm double peak; schools peak during
daytime hours and drop to near-zero overnight), scaled by a randomly
sampled peak-demand value per zone and perturbed with light (~6%)
random noise so no two zones in the same category are identical.

## Backup capacity
Assigned per zone as none / partial / full, with probabilities that
vary by category (hospitals most likely to have some generator
capacity, idle residential almost never does). **This run intentionally
produced a mixed outcome for the two hospitals** — one is backed up to
~130% of peak demand (a generator with margin), the other to only ~55%
(a partial/undersized generator) — because a demo where every critical
zone happens to be fully self-sufficient doesn't exercise the part of
the model that matters most: correctly prioritizing a critical zone
that genuinely needs grid power over one that doesn't.

## Outage history
Each zone is seeded with a small random starting "consecutive hours
without power" value (0–6h) at hour 0, so the fairness constraint has
something real to act on from the first hour of the horizon rather than
starting every zone from a clean slate.

## What's simulated vs. real
Every value in this dataset (demand, criticality tier, backup capacity,
starting outage history) is synthetic — assigned by the generator
script, not sourced from a real utility. This is standard and expected
for a weekend prototype; the goal is to prove the allocation *logic*
works, not to source live data. See the real-world data-sourcing
discussion (self-reporting/registration, smart-meter detection of
near-zero demand during grid supply, tiered zone classification,
survey-based profiling) in the main project document for how this gap
would be closed for a real deployment.

## Known limitation to flag in Q&A
Two hospitals is a small sample — the specific mix of backup coverage
in this dataset (one well-covered, one not) was a deliberate seed
choice to make the demo informative, not something the model would
reliably produce on a re-run with a different seed. A real deployment
would need this data collected per-institution, not simulated.
