# Methodology Note (Role B)

## LP formulation
- **Decision variables:** `x[zone, hour] ∈ [0, 1]` — fraction of a zone's
  demand served that hour (continuous, not binary — see below).
- **Objective:** minimize `Σ effective_need[z,h] × (1 − x[z,h])` — total
  unserved-need-weighted harm across all zones and hours.
- **Constraint 1 (capacity):** total allocated kWh per hour ≤ available
  grid capacity that hour.
- **Constraint 2 (demand ceiling):** built into the `[0,1]` bound on
  `x[z,h]` — a zone can never be allocated more than its own demand.
- **Constraint 3 (fairness):** in every rolling window of
  `(MAX_CONSECUTIVE_OUTAGE_HOURS + 1)` hours, a zone's total served
  fraction across the window must sum to at least a minimum threshold —
  the LP-friendly way to bound how long any zone can go under-served
  without introducing binary variables (which would turn this into a
  slower MILP).

**Why continuous, not binary on/off:** the problem statement says "each
zone's power allocation cannot exceed its actual demand," which reads
naturally as a continuous fraction (partial curtailment), and it keeps
this a true LP — solves in well under a second at this scale, and stays
easy to reason about and defend live in front of judges.

## Fairness: hard constraint, not soft penalty
This implementation treats the consecutive-hours cap as a **hard**
constraint rather than folding a fairness-violation penalty into the
objective. Trade-off worth stating explicitly if asked: a hard
constraint can make the LP **infeasible** if the fairness cap is too
tight relative to available capacity (the model will report this
clearly rather than silently returning a degenerate solution) — a soft
penalty would never go infeasible, but would blur the "minimize harm"
story since fairness violations become just another cost term. We chose
hard constraints because "no zone denied power for more than N hours"
is a bright-line commitment worth being able to guarantee absolutely,
not merely discourage.

## Available capacity
Set as a fixed fraction (55% in the base scenario) of each hour's total
demand across all zones — this is what makes shedding necessary at all;
without a shortfall the LP trivially serves everyone and there's
nothing to compare against the baseline.

## Baseline (fixed rotation)
Zones are split into fixed groups purely by their order in the dataset
(no criticality or backup awareness — that's the entire point). One
group is "off" per rotation period on a round-robin cycle; the "on"
groups split available capacity proportionally by demand if capacity is
still short. The baseline respects the exact same hourly capacity cap
as the optimizer, so the comparison is about *allocation*, not about
one model getting more total electricity than the other.

## Harm scoring
Both schedules are scored by the identical formula:
`harm[z,h] = effective_need[z,h] × (1 − served_fraction[z,h])`
This is deliberately the same quantity the LP's objective minimizes, so
"total harm reduced" is a fair, apples-to-apples number. Because that
makes the headline metric somewhat circular (the optimizer is scored on
exactly what it optimized for), two **independent** metrics are also
tracked, since the optimizer wasn't directly solving for either:
- **Worst-case / mean consecutive outage hours per zone** — does the
  fairness constraint actually prevent long unbroken outages, not just
  lower the total?
- **Critical-zone (hospital) actual power shortfall hours** — hours
  where a hospital's *grid allocation + backup capacity* falls short of
  its demand. This must NOT be measured on grid allocation alone: a
  hospital whose generator fully covers its demand is *correctly* given
  near-zero grid power by the optimizer, and a grid-only metric would
  misreport that hospital as worse off under the optimized model than
  under the need-blind baseline — the opposite of reality. See the
  in-code comment in `evaluation.py::compute_metrics` for the concrete
  numbers this caught during a test run.

## Sensitivity / honesty check results
Re-running the LP as capacity shrinks (55% → 45% → 35% of demand) and as
the fairness cap loosens (4h → 6h → 8h) shows harm increasing roughly
monotonically as capacity tightens — expected — while a looser fairness
cap consistently *lowers* total harm at every capacity level (see
`sensitivity_results.csv`). That's an honest trade-off to be upfront
about in Q&A: tighter fairness guarantees cost some total-harm
efficiency, because the optimizer has less freedom to concentrate power
wherever need is highest. The team is protecting the fairness
constraint from being cut under time pressure for exactly this reason —
it's the part of the model that makes the "equity-aware" claim earn its
name, not just a knob to sacrifice for a better headline number.
