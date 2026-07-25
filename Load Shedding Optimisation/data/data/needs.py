"""
needs.py — Role A (Data & Domain Modeling)

Two small, testable, importable functions that Role B's optimizer depends on:

  1. effective_need(criticality_score, demand_kwh, backup_capacity_kwh)
     -> effective_need = criticality_score * max(0, demand - backup_capacity)

  2. update_outage_tracker(prev_cumulative_outage_hours, was_powered)
     -> next cumulative_outage_hours, following the rule:
        - if the zone WAS powered this hour, its consecutive-outage streak resets to 0
        - if the zone was NOT powered, the streak increments by 1

Both are deliberately stateless / pure functions (no hidden globals) so
Role B can call them per zone per hour inside the LP loop or the
evaluation pipeline without needing to understand their internals —
just the signatures below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_need(
    criticality_score: float, demand_kwh: float, backup_capacity_kwh: float
) -> float:
    """
    effective_need = criticality_score x max(0, demand - backup_capacity)

    A zone whose backup fully covers its demand has effective_need == 0,
    so the optimizer has no incentive to allocate it scarce grid power.
    """
    unmet_demand = max(0.0, demand_kwh - backup_capacity_kwh)
    return criticality_score * unmet_demand


def effective_need_vectorized(
    criticality_score: pd.Series | np.ndarray,
    demand_kwh: pd.Series | np.ndarray,
    backup_capacity_kwh: pd.Series | np.ndarray,
):
    """Vectorized version for applying effective_need across a whole DataFrame column set."""
    unmet_demand = np.maximum(0.0, np.asarray(demand_kwh) - np.asarray(backup_capacity_kwh))
    return np.asarray(criticality_score) * unmet_demand


def update_outage_tracker(prev_cumulative_outage_hours: int, was_powered: bool) -> int:
    """
    Updates a zone's CONSECUTIVE hours-without-power streak for the fairness
    constraint. This tracks consecutive outage hours, not lifetime total —
    the streak resets to 0 the moment the zone receives power again.

    Args:
        prev_cumulative_outage_hours: streak going into this hour.
        was_powered: whether the zone received power (any allocation > 0)
                     during this hour.

    Returns:
        Updated consecutive-outage-hour count for this hour.
    """
    if was_powered:
        return 0
    return prev_cumulative_outage_hours + 1


def build_outage_streaks(
    df: pd.DataFrame, allocation_col: str = "allocated_kwh"
) -> pd.DataFrame:
    """
    Convenience helper: given a long-format DataFrame with columns
    [zone_id, hour, <allocation_col>] plus a starting outage value at hour 0
    (as produced by generate_dataset.py's cumulative_outage_hours seed),
    compute the full consecutive-outage-hour streak per zone per hour.

    This is what the fairness constraint's "max N consecutive hours without
    power" check should be run against. Note: this is a POST-HOC helper for
    scoring/validation. The optimizer itself must enforce the fairness cap
    as an LP constraint (see optimizer.py) — this function does not do that;
    it only recomputes streaks from a given allocation to verify the
    constraint held.
    """
    df = df.sort_values(["zone_id", "hour"]).copy()
    streaks = []
    for zone_id, group in df.groupby("zone_id", sort=False):
        streak = 0
        zone_streaks = []
        for _, row in group.iterrows():
            was_powered = row[allocation_col] > 1e-6
            streak = update_outage_tracker(streak, was_powered)
            zone_streaks.append(streak)
        streaks.extend(zone_streaks)
    df["consecutive_outage_hours"] = streaks
    return df


def add_effective_need_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'effective_need' column to a zones_dataset-shaped DataFrame.
    Expects columns: criticality_score, demand_kwh, backup_capacity_kwh.
    """
    df = df.copy()
    df["effective_need"] = effective_need_vectorized(
        df["criticality_score"], df["demand_kwh"], df["backup_capacity_kwh"]
    )
    return df


# ---------------------------------------------------------------------------
# Unit tests against hand-calculated edge cases (run directly: python needs.py)
# ---------------------------------------------------------------------------
def _run_tests():
    # --- effective_need edge cases ---
    # Zero backup: full demand counts as unmet need.
    assert effective_need(10, 50, 0) == 500, "zero backup case failed"

    # Full backup covering demand exactly: zero effective need.
    assert effective_need(10, 50, 50) == 0, "exact full backup case failed"

    # Backup exceeding demand: still zero (not negative).
    assert effective_need(10, 50, 80) == 0, "over-covered backup case failed"

    # Zero demand: zero need regardless of criticality.
    assert effective_need(10, 0, 0) == 0, "zero demand case failed"

    # Partial backup: only the shortfall counts.
    assert effective_need(5, 40, 15) == 5 * 25, "partial backup case failed"

    # --- outage tracker edge cases ---
    assert update_outage_tracker(0, was_powered=True) == 0, "powered-from-zero failed"
    assert update_outage_tracker(3, was_powered=True) == 0, "reset-on-power failed"
    assert update_outage_tracker(0, was_powered=False) == 1, "first-outage-hour failed"
    assert update_outage_tracker(5, was_powered=False) == 6, "streak-increment failed"

    print("All needs.py unit tests passed.")


if __name__ == "__main__":
    _run_tests()

    # Smoke test against the real dataset, if present
    import os

    if os.path.exists("zones_dataset.csv"):
        df = pd.read_csv("zones_dataset.csv")
        df = add_effective_need_column(df)
        print("\nSample effective_need values (first 5 rows):")
        print(
            df[
                [
                    "zone_id",
                    "zone_name",
                    "hour",
                    "demand_kwh",
                    "backup_capacity_kwh",
                    "effective_need",
                ]
            ].head()
        )
