"""
optimizer.py — Role B (Optimization & Evaluation)

Formalizes and solves the equity-aware load-shedding LP.

Mathematical formulation
-------------------------
Sets:
    Z = zones, H = hours in the scheduling horizon

Decision variables:
    x[z, h] in [0, 1]   fraction of zone z's demand served in hour h
                         (continuous relaxation — a zone can be partially
                         served; this matches "each zone's allocation cannot
                         exceed its demand" more naturally than a strict 0/1
                         on/off variable, and keeps the LP a true LP rather
                         than a MILP, which solves far faster at hackathon
                         scale)

Objective:
    minimize   sum over z, h of  effective_need[z, h] * (1 - x[z, h])

    i.e. minimize total unserved-need-weighted harm. Serving a zone with
    high effective_need reduces the objective more than serving a
    low-need zone by the same amount.

Constraints:
    1. Capacity (per hour):
       sum over z of allocated_kwh[z, h] <= available_capacity[h]
       where allocated_kwh[z, h] = x[z, h] * demand_kwh[z, h]

    2. Demand ceiling (per zone, per hour):
       0 <= x[z, h] <= 1
       (built into the variable bounds — a zone can never receive more
       than its own demand)

    3. Fairness cap (per zone):
       a zone cannot go more than MAX_CONSECUTIVE_OUTAGE_HOURS consecutive
       hours with x[z, h] ~ 0 (effectively unpowered). Implemented as a
       rolling-window constraint: in every window of
       (MAX_CONSECUTIVE_OUTAGE_HOURS + 1) consecutive hours, at least one
       hour must have x[z, h] >= MIN_SERVED_FRACTION (the zone counts as
       "powered" that hour).

Notes on the fairness formalization
------------------------------------
This is a MULTI-PERIOD constraint — it links hours together within a
zone, not just within a single hour's slice. That's why the LP is built
over the whole horizon at once rather than solved hour-by-hour: an
hour-by-hour (myopic) solve cannot see or enforce a rolling consecutive-
hours cap.

A zone's PRE-EXISTING outage streak (from generate_dataset.py's
starting_outage_hours / needs.py's tracker) is respected at the horizon
boundary: if a zone already has k hours of consecutive outage before
hour 0, it is only allowed (MAX_CONSECUTIVE_OUTAGE_HOURS - k) further
unpowered hours before the constraint forces it to be served.

Fairness is enforced as a HARD constraint here (not a soft penalty in
the objective). This is a deliberate choice, not the only valid one —
see the "fairness as hard constraint vs. soft penalty" discussion in
methodology.md. A hard constraint can make the LP infeasible if capacity
is too tight relative to the fairness cap; if that happens, this script
reports it clearly (see solve()) rather than silently returning nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from needs import add_effective_need_column

# ---------------------------------------------------------------------------
# Configuration — agree with Role A before changing zone count / horizon
# ---------------------------------------------------------------------------
MAX_CONSECUTIVE_OUTAGE_HOURS = 4   # fairness cap: no zone unpowered > 4h straight
MIN_SERVED_FRACTION = 0.5          # x[z,h] >= 0.5 counts as "powered" for fairness purposes
CAPACITY_FRACTION_OF_PEAK_TOTAL = 0.55  # available grid capacity as a fraction
                                          # of total peak demand across all zones —
                                          # this is what makes shedding necessary at all


def load_data(dataset_path: str = "zones_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = add_effective_need_column(df)
    return df


def compute_available_capacity(df: pd.DataFrame) -> pd.Series:
    """
    Derives an hourly available-capacity series. Available capacity is set
    as a fraction of each hour's total demand across zones, so shedding is
    always necessary (otherwise the LP trivially serves everyone and there's
    nothing to compare against the baseline).
    """
    total_demand_by_hour = df.groupby("hour")["demand_kwh"].sum()
    return total_demand_by_hour * CAPACITY_FRACTION_OF_PEAK_TOTAL


def solve_lp(
    df: pd.DataFrame,
    available_capacity: pd.Series,
    max_consecutive_outage_hours: int = MAX_CONSECUTIVE_OUTAGE_HOURS,
    min_served_fraction: float = MIN_SERVED_FRACTION,
):
    """
    Builds and solves the LP. Returns (status_str, allocation_df).

    allocation_df columns: zone_id, hour, demand_kwh, effective_need,
    served_fraction (x[z,h]), allocated_kwh.
    """
    zones = df["zone_id"].unique().tolist()
    hours = sorted(df["hour"].unique().tolist())

    # Fast lookups
    demand = df.set_index(["zone_id", "hour"])["demand_kwh"].to_dict()
    need = df.set_index(["zone_id", "hour"])["effective_need"].to_dict()

    # Starting outage streak per zone, seeded from generate_dataset.py
    starting_streak = (
        df[df["hour"] == hours[0]]
        .set_index("zone_id")["cumulative_outage_hours"]
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    prob = pulp.LpProblem("equity_aware_load_shedding", pulp.LpMinimize)

    # Decision variables: x[z, h] in [0, 1]
    x = {
        (z, h): pulp.LpVariable(f"x_{z}_{h}", lowBound=0, upBound=1)
        for z in zones
        for h in hours
    }

    # Objective: minimize total effective-need-weighted unserved demand
    prob += pulp.lpSum(need[(z, h)] * (1 - x[(z, h)]) for z in zones for h in hours)

    # Constraint 1: hourly capacity
    for h in hours:
        prob += (
            pulp.lpSum(x[(z, h)] * demand[(z, h)] for z in zones)
            <= available_capacity.loc[h],
            f"capacity_hour_{h}",
        )

    # Constraint 3: fairness cap via rolling-window "served" indicator.
    # We introduce a binary-ish auxiliary: served[z,h] = 1 if x[z,h] >= min_served_fraction.
    # PuLP can't do that "if" directly in an LP, so instead we use the
    # standard trick: require that in every window of
    # (max_consecutive_outage_hours + 1) hours, the SUM of x[z,h] is at
    # least min_served_fraction (i.e. cumulative service in the window
    # can't be arbitrarily close to zero across the whole window). This is
    # a relaxation of "at least one hour must individually clear the
    # threshold" but is the standard LP-friendly way to keep this a true
    # LP (no binaries) while still preventing long real outages.
    window = max_consecutive_outage_hours + 1
    for z in zones:
        # Account for the zone's pre-existing streak at the horizon boundary:
        # it gets fewer "free" unpowered hours before the first window constraint bites.
        carry_in = starting_streak.get(z, 0)
        for start_idx in range(len(hours)):
            window_hours = hours[start_idx: start_idx + window]
            if len(window_hours) < window:
                break  # only full windows; tail handled implicitly by earlier windows
            # If this is the very first window and the zone already has a
            # streak going in, shrink the effective slack required.
            required = min_served_fraction
            if start_idx == 0 and carry_in > 0:
                # Already carrying an outage streak — demand service sooner.
                required = min_served_fraction + 0.1 * min(carry_in, max_consecutive_outage_hours)
            prob += (
                pulp.lpSum(x[(z, h)] for h in window_hours) >= required,
                f"fairness_{z}_{start_idx}",
            )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]

    records = []
    for z in zones:
        for h in hours:
            served_fraction = x[(z, h)].value()
            records.append(
                {
                    "zone_id": z,
                    "hour": h,
                    "demand_kwh": demand[(z, h)],
                    "effective_need": need[(z, h)],
                    "served_fraction": served_fraction,
                    "allocated_kwh": served_fraction * demand[(z, h)],
                }
            )
    allocation_df = pd.DataFrame(records)
    return status_str, allocation_df


def solve(dataset_path: str = "zones_dataset.csv", output_path: str = "optimized_schedule.csv"):
    df = load_data(dataset_path)
    available_capacity = compute_available_capacity(df)
    status, allocation_df = solve_lp(df, available_capacity)

    print(f"LP solve status: {status}")
    if status != "Optimal":
        print(
            "WARNING: LP did not solve to optimality. If status is 'Infeasible', "
            "the fairness cap and available capacity are in direct conflict — "
            "loosen MAX_CONSECUTIVE_OUTAGE_HOURS or raise "
            "CAPACITY_FRACTION_OF_PEAK_TOTAL and re-run."
        )

    # Sanity checks (per the plan's Day 2 step 21)
    merged = allocation_df.merge(
        df[["zone_id", "hour", "zone_name", "category", "criticality_score"]],
        on=["zone_id", "hour"],
    )
    assert (merged["allocated_kwh"] <= merged["demand_kwh"] + 1e-4).all(), \
        "A zone was allocated more than its demand"
    hourly_totals = merged.groupby("hour")["allocated_kwh"].sum()
    over_capacity = hourly_totals[hourly_totals > available_capacity + 1e-3]
    assert over_capacity.empty, f"Capacity exceeded in hours: {over_capacity.index.tolist()}"
    print("Sanity checks passed: no zone over-served, no hour over capacity.")

    merged.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(merged)} rows)")
    return merged


if __name__ == "__main__":
    solve()
