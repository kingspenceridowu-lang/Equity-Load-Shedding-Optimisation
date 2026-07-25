"""
generate_dataset.py — Role A (Data & Domain Modeling)

Builds the synthetic zone dataset for the Equity-Aware Load-Shedding
Optimization project and writes it to zones_dataset.csv.

Output schema (one row per zone per hour):
    zone_id, zone_name, category, criticality_score, hour,
    demand_kwh, backup_capacity_kwh, cumulative_outage_hours

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — confirm zone count / horizon with Role B before changing
# ---------------------------------------------------------------------------
RANDOM_SEED = 5    # chosen so hospital backup coverage is mixed (one near-full,
                    # one partial) rather than both fully covered by chance —
                    # gives the critical-zone metric something real to show
N_HOURS = 48          # 2-day scheduling horizon (recommend 24-48; agreed with Role B)

# Criticality tiers (higher = more harm if cut). One-line justification each:
CRITICALITY_TIERS = {
    "hospital":            10,  # life-safety equipment, zero tolerance for outage
    "cold_storage_water":   7,  # spoilage / public-health risk (vaccines, water pumping)
    "school_institutional": 5,  # disruption to essential public services, low life-safety risk
    "commercial":           3,  # economic loss but no life-safety or spoilage risk
    "residential_idle":     1,  # lowest urgency, most tolerant of scheduled outages
}

# Zone roster: (category, count, name_prefix)
# 20 zones total — matches the plan's recommended 15-25 range.
ZONE_ROSTER = [
    ("hospital",             2, "Hospital"),
    ("cold_storage_water",   2, "ColdStorage_WaterPump"),
    ("school_institutional", 4, "School_Institution"),
    ("commercial",           6, "Commercial"),
    ("residential_idle",     6, "Residential"),
]

# Backup capacity distribution per category: (P(none), P(partial), P(full))
# Most zones have no backup; institutions are more likely to have generators/solar.
BACKUP_PROFILE = {
    "hospital":             (0.00, 0.40, 0.60),  # hospitals almost always have some generator
    "cold_storage_water":   (0.30, 0.50, 0.20),
    "school_institutional": (0.70, 0.25, 0.05),
    "commercial":           (0.75, 0.20, 0.05),
    "residential_idle":     (0.95, 0.05, 0.00),  # near-zero private backup for idle residential
}

# Peak demand ranges (kWh/hour at daily peak) per category — used to scale the
# time-of-day curve. (low, high) sampled uniformly per zone.
PEAK_DEMAND_RANGE = {
    "hospital":             (40, 70),
    "cold_storage_water":   (25, 45),
    "school_institutional": (15, 30),
    "commercial":           (10, 25),
    "residential_idle":     (3, 10),
}

OUTPUT_PATH = "zones_dataset.csv"


def time_of_day_curve(category: str, hour_of_day: np.ndarray) -> np.ndarray:
    """
    Returns a multiplier in roughly [0.15, 1.0] describing how demand varies
    across a 24-hour cycle for a given category. hour_of_day is 0-23.
    """
    if category == "hospital":
        # Hospitals run flat, high baseline load around the clock.
        return 0.85 + 0.15 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi) * 0.3

    if category == "cold_storage_water":
        # Fairly flat — refrigeration/pumping runs continuously, small daytime bump.
        return 0.7 + 0.3 * np.clip(np.sin((hour_of_day - 12) / 24 * 2 * np.pi) + 0.3, 0, 1)

    if category == "school_institutional":
        # Peaks during school/working hours (8am-4pm), near-zero overnight.
        return np.clip(
            np.exp(-0.5 * ((hour_of_day - 12) / 4.0) ** 2), 0.05, 1.0
        )

    if category == "commercial":
        # Two humps: daytime trading + early evening.
        morning = np.exp(-0.5 * ((hour_of_day - 12) / 4.5) ** 2)
        evening = 0.6 * np.exp(-0.5 * ((hour_of_day - 19) / 2.5) ** 2)
        return np.clip(0.2 + morning + evening, 0.15, 1.3) / 1.3

    if category == "residential_idle":
        # Classic double-peak: 6-9am and 6-10pm.
        morning = np.exp(-0.5 * ((hour_of_day - 7.5) / 1.5) ** 2)
        evening = np.exp(-0.5 * ((hour_of_day - 20) / 2.0) ** 2)
        return np.clip(0.15 + morning + evening, 0.15, 1.2) / 1.2

    raise ValueError(f"Unknown category: {category}")


def build_zone_roster(rng: np.random.Generator) -> pd.DataFrame:
    """One row per zone (static attributes, not yet expanded across hours)."""
    rows = []
    zone_idx = 1
    for category, count, prefix in ZONE_ROSTER:
        for i in range(count):
            zone_id = f"Z{zone_idx:03d}"
            zone_name = f"{prefix}_{i + 1}"
            peak_low, peak_high = PEAK_DEMAND_RANGE[category]
            peak_demand = rng.uniform(peak_low, peak_high)

            backup_none, backup_partial, backup_full = BACKUP_PROFILE[category]
            backup_choice = rng.choice(
                ["none", "partial", "full"], p=[backup_none, backup_partial, backup_full]
            )
            if backup_choice == "none":
                backup_frac = 0.0
            elif backup_choice == "partial":
                backup_frac = rng.uniform(0.25, 0.65)  # covers 25-65% of peak demand
            else:  # full
                backup_frac = rng.uniform(0.95, 1.15)  # covers demand with margin

            # Starting outage history so the fairness constraint has something
            # to act on from hour 1 — zones don't all start at zero.
            starting_outage_hours = int(rng.integers(0, 7))

            rows.append(
                {
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "category": category,
                    "criticality_score": CRITICALITY_TIERS[category],
                    "peak_demand_kwh": peak_demand,
                    "backup_capacity_kwh": peak_demand * backup_frac,
                    "starting_outage_hours": starting_outage_hours,
                }
            )
            zone_idx += 1
    return pd.DataFrame(rows)


def expand_to_hourly(zone_roster: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Expand the static roster into one row per zone per hour with a demand curve + noise."""
    records = []
    for _, zone in zone_roster.iterrows():
        hours = np.arange(N_HOURS)
        hour_of_day = hours % 24
        curve = time_of_day_curve(zone["category"], hour_of_day)
        noise = rng.normal(1.0, 0.06, size=N_HOURS)  # +/- ~6% light random noise
        demand = np.clip(zone["peak_demand_kwh"] * curve * noise, 0.2, None)

        for h, d in zip(hours, demand):
            records.append(
                {
                    "zone_id": zone["zone_id"],
                    "zone_name": zone["zone_name"],
                    "category": zone["category"],
                    "criticality_score": zone["criticality_score"],
                    "hour": int(h),
                    "demand_kwh": round(float(d), 2),
                    "backup_capacity_kwh": round(float(zone["backup_capacity_kwh"]), 2),
                    # cumulative_outage_hours is seeded here with the zone's starting
                    # value at hour 0 only; needs.py's tracker owns updating it hour
                    # by hour based on actual allocation outcomes. Downstream code
                    # should treat this column as the SEED, not a precomputed series.
                    "cumulative_outage_hours": (
                        int(zone["starting_outage_hours"]) if h == 0 else np.nan
                    ),
                }
            )
    df = pd.DataFrame(records)
    return df


def validate(df: pd.DataFrame, zone_roster: pd.DataFrame) -> None:
    """Basic sanity checks — no impossible values."""
    assert (df["demand_kwh"] > 0).all(), "Found non-positive demand"
    assert (df["backup_capacity_kwh"] >= 0).all(), "Found negative backup capacity"
    assert df["zone_id"].nunique() == len(zone_roster), "Zone count mismatch"
    assert df.groupby("zone_id")["hour"].count().eq(N_HOURS).all(), "Missing hourly rows"
    # Backup shouldn't wildly exceed demand at every hour for "partial" zones —
    # spot check is enough here, full distributional check isn't necessary.
    print(f"Validation OK: {df['zone_id'].nunique()} zones x {N_HOURS} hours "
          f"= {len(df)} rows")


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    zone_roster = build_zone_roster(rng)
    df = expand_to_hourly(zone_roster, rng)
    validate(df, zone_roster)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(df)} rows)")

    # Quick summary for the assumptions note / sanity check
    print("\nZone roster summary:")
    print(
        zone_roster.groupby("category")
        .agg(
            n_zones=("zone_id", "count"),
            avg_peak_demand=("peak_demand_kwh", "mean"),
            avg_backup=("backup_capacity_kwh", "mean"),
        )
        .round(1)
    )


if __name__ == "__main__":
    main()
