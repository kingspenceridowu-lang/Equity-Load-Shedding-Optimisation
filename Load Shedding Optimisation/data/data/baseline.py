"""
baseline.py — Role B (Optimization & Evaluation)

Builds the traditional fixed-rotation load-shedding schedule against the
SAME dataset and available capacity as optimizer.py, so the two can be
compared fairly by evaluation.py.

Logic (plain Python, no optimization):
    - Zones are split into ROTATION_GROUPS groups (a fixed assignment,
      independent of criticality/backup/need — this mirrors how real
      rotational load-shedding schedules work today: a zone's position
      in the rotation, not its urgency, determines when it's cut).
    - In each hour, one group is "off" (unpowered) on a round-robin
      cycle; all other groups are "on" (fully served) SUBJECT to the
      same total hourly capacity cap as the optimizer, split evenly
      across the "on" groups if capacity is still short.
    - This gives the baseline the same total-capacity constraint the
      optimizer had to respect, so neither model is allowed to use more
      electricity than the other — the comparison is about allocation,
      not total supply.
"""

from __future__ import annotations

import pandas as pd

ROTATION_GROUP_SIZE = 4  # zones per rotation group; groups take turns being cut
ROTATION_PERIOD_HOURS = 4  # each group is "off" for this many consecutive hours per cycle


def assign_rotation_groups(zone_ids: list[str]) -> dict[str, int]:
    """
    Fixed, need-blind group assignment — deliberately just cycles through
    zone_ids in the order they appear in the dataset (no criticality or
    backup awareness), which is the entire point of the baseline.
    """
    n_groups = max(1, len(zone_ids) // ROTATION_GROUP_SIZE)
    return {zid: i % n_groups for i, zid in enumerate(zone_ids)}


def build_baseline_schedule(
    df: pd.DataFrame, available_capacity: pd.Series
) -> pd.DataFrame:
    """
    Returns a DataFrame shaped like optimizer.py's allocation_df:
    zone_id, hour, demand_kwh, served_fraction, allocated_kwh
    (effective_need is left for evaluation.py to attach so both schedules
    go through the identical scoring path).
    """
    zone_ids = df["zone_id"].unique().tolist()
    hours = sorted(df["hour"].unique().tolist())
    groups = assign_rotation_groups(zone_ids)
    n_groups = max(groups.values()) + 1

    demand = df.set_index(["zone_id", "hour"])["demand_kwh"].to_dict()

    records = []
    for h in hours:
        # Which group is "off" this hour, on a fixed round-robin cycle.
        off_group = (h // ROTATION_PERIOD_HOURS) % n_groups

        on_zones = [z for z in zone_ids if groups[z] != off_group]
        off_zones = [z for z in zone_ids if groups[z] == off_group]

        # Off-group zones get nothing, regardless of criticality or backup —
        # that's the defining trait of fixed rotation.
        for z in off_zones:
            records.append(
                {
                    "zone_id": z,
                    "hour": h,
                    "demand_kwh": demand[(z, h)],
                    "served_fraction": 0.0,
                    "allocated_kwh": 0.0,
                }
            )

        # On-group zones split available capacity evenly by demand share —
        # still need-blind: no zone is prioritized over another within the
        # "on" group based on urgency.
        cap = available_capacity.loc[h]
        on_demand_total = sum(demand[(z, h)] for z in on_zones)
        if on_demand_total <= cap or on_demand_total == 0:
            served_fraction = 1.0
        else:
            served_fraction = cap / on_demand_total  # equal proportional curtailment

        for z in on_zones:
            records.append(
                {
                    "zone_id": z,
                    "hour": h,
                    "demand_kwh": demand[(z, h)],
                    "served_fraction": served_fraction,
                    "allocated_kwh": served_fraction * demand[(z, h)],
                }
            )

    baseline_df = pd.DataFrame(records)
    return baseline_df.sort_values(["zone_id", "hour"]).reset_index(drop=True)


def build_and_save(
    dataset_path: str = "zones_dataset.csv", output_path: str = "baseline_schedule.csv"
):
    from optimizer import load_data, compute_available_capacity  # reuse Role B's own helpers

    df = load_data(dataset_path)
    available_capacity = compute_available_capacity(df)
    baseline_df = build_baseline_schedule(df, available_capacity)

    merged = baseline_df.merge(
        df[["zone_id", "hour", "zone_name", "category", "criticality_score", "effective_need"]],
        on=["zone_id", "hour"],
    )

    # Same sanity checks as the optimizer, for a fair apples-to-apples build
    assert (merged["allocated_kwh"] <= merged["demand_kwh"] + 1e-4).all(), \
        "A zone was allocated more than its demand in the baseline"
    hourly_totals = merged.groupby("hour")["allocated_kwh"].sum()
    over_capacity = hourly_totals[hourly_totals > available_capacity + 1e-3]
    assert over_capacity.empty, f"Baseline exceeded capacity in hours: {over_capacity.index.tolist()}"

    merged.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(merged)} rows)")
    return merged


if __name__ == "__main__":
    build_and_save()
