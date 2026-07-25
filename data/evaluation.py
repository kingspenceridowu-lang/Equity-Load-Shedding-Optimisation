"""
evaluation.py — Role B (Optimization & Evaluation)

Scores the optimized schedule and the baseline schedule with the IDENTICAL
harm formula and produces the comparison metrics that feed the dashboard
and pitch (Role C).

Harm formula (deliberately identical to the LP's own objective, so the
"total harm reduced" headline number is directly comparable):
    harm[z, h] = effective_need[z, h] * (1 - served_fraction[z, h])

Because "total harm reduced" is measured on the same quantity the
optimizer minimized, it is included alongside two INDEPENDENT metrics
that neither schedule was directly optimized for, so the improvement
isn't just circular:
    - max_consecutive_outage_hours per zone (does the optimizer actually
      protect zones from long unbroken outages, not just low totals?)
    - critical_zone_outage_hours (hospital-category zones specifically)
"""

from __future__ import annotations

import pandas as pd

from needs import build_outage_streaks


def score_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'harm' column to a schedule DataFrame that already has
    effective_need and served_fraction columns."""
    df = schedule_df.copy()
    df["harm"] = df["effective_need"] * (1 - df["served_fraction"])
    return df


def compute_metrics(schedule_df: pd.DataFrame, label: str) -> dict:
    scored = score_schedule(schedule_df)

    total_harm = scored["harm"].sum()

    # Independent metric 1: max consecutive outage hours, per zone, worst case
    streaked = build_outage_streaks(scored, allocation_col="allocated_kwh")
    max_consecutive_per_zone = streaked.groupby("zone_id")["consecutive_outage_hours"].max()
    worst_case_consecutive_outage = max_consecutive_per_zone.max()
    mean_max_consecutive_outage = max_consecutive_per_zone.mean()

    # Independent metric 2: critical-zone (hospital) outage hours.
    #
    # IMPORTANT: this must measure ACTUAL power shortfall (grid allocation +
    # backup vs. demand), not grid allocation alone. A hospital with backup
    # covering 100% of its demand is CORRECTLY given near-zero grid power by
    # the optimizer (that's the whole point of the backup-aware effective_need
    # design) — but if this metric only looked at served_fraction (grid
    # allocation), it would misreport that hospital as "worse off" under the
    # optimized model than under the need-blind baseline, which is the
    # opposite of what's actually happening. Requires 'backup_capacity_kwh'
    # to be present on the schedule DataFrame.
    critical = scored[scored["category"] == "hospital"].copy()
    if "backup_capacity_kwh" in critical.columns:
        actual_power = critical["allocated_kwh"] + critical["backup_capacity_kwh"]
        critical_unpowered_hours = (actual_power < critical["demand_kwh"] - 1e-6).sum()
    else:
        # Fallback only — grid-only view, will overstate hospital harm for
        # backed-up zones. Prefer merging backup_capacity_kwh before calling.
        critical_unpowered_hours = (critical["served_fraction"] < 0.5).sum()
    critical_total_hours = len(critical)

    # Per-zone outage hours, for the per-zone comparison chart.
    #
    # Same fix as the critical-zone metric above and for the same reason:
    # this must be ACTUAL power shortfall (grid + backup vs. demand), not
    # grid allocation alone — otherwise a well-backed-up zone (e.g. a
    # hospital the optimizer correctly deprioritizes for grid power because
    # its generator covers it) shows up as having MORE "outage hours" under
    # the optimized model than the baseline, which is backwards and would
    # be a very bad moment if it ended up on a chart unexamined.
    if "backup_capacity_kwh" in scored.columns:
        actual_power_all = scored["allocated_kwh"] + scored["backup_capacity_kwh"]
        is_outage_hour = actual_power_all < (scored["demand_kwh"] - 1e-6)
    else:
        is_outage_hour = scored["served_fraction"] < 0.5
    per_zone_outage_hours = (
        scored.assign(is_outage_hour=is_outage_hour)
        .groupby(["zone_id", "zone_name", "category"])["is_outage_hour"]
        .sum()
        .reset_index()
        .rename(columns={"is_outage_hour": "outage_hours"})
    )

    return {
        "label": label,
        "total_harm": round(float(total_harm), 1),
        "worst_case_consecutive_outage_hours": int(worst_case_consecutive_outage),
        "mean_max_consecutive_outage_hours": round(float(mean_max_consecutive_outage), 2),
        "critical_zone_unpowered_hours": int(critical_unpowered_hours),
        "critical_zone_total_hours": int(critical_total_hours),
        "per_zone_outage_hours": per_zone_outage_hours,  # DataFrame, not scalar
    }


def build_comparison(
    optimized_path: str = "optimized_schedule.csv",
    baseline_path: str = "baseline_schedule.csv",
    output_path: str = "results_comparison.csv",
    per_zone_output_path: str = "results_per_zone.csv",
):
    optimized = pd.read_csv(optimized_path)
    baseline = pd.read_csv(baseline_path)

    # Both schedules need backup_capacity_kwh attached for the corrected
    # critical-zone metric (see compute_metrics docstring note).
    dataset = pd.read_csv("zones_dataset.csv")[["zone_id", "hour", "backup_capacity_kwh"]]
    optimized = optimized.merge(dataset, on=["zone_id", "hour"], how="left")
    baseline = baseline.merge(dataset, on=["zone_id", "hour"], how="left")

    opt_metrics = compute_metrics(optimized, "optimized")
    base_metrics = compute_metrics(baseline, "baseline_fixed_rotation")

    summary_rows = []
    for m in (base_metrics, opt_metrics):
        summary_rows.append(
            {
                "schedule": m["label"],
                "total_harm": m["total_harm"],
                "worst_case_consecutive_outage_hours": m["worst_case_consecutive_outage_hours"],
                "mean_max_consecutive_outage_hours": m["mean_max_consecutive_outage_hours"],
                "critical_zone_unpowered_hours": m["critical_zone_unpowered_hours"],
                "critical_zone_total_hours": m["critical_zone_total_hours"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    harm_reduction_pct = (
        (base_metrics["total_harm"] - opt_metrics["total_harm"]) / base_metrics["total_harm"] * 100
        if base_metrics["total_harm"] > 0
        else 0.0
    )
    summary_df["harm_reduction_vs_baseline_pct"] = [None, round(harm_reduction_pct, 1)]

    summary_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(summary_df.to_string(index=False))

    # Per-zone outage-hours comparison, for the grouped bar chart
    per_zone = base_metrics["per_zone_outage_hours"].merge(
        opt_metrics["per_zone_outage_hours"],
        on=["zone_id", "zone_name", "category"],
        suffixes=("_baseline", "_optimized"),
    )
    per_zone.to_csv(per_zone_output_path, index=False)
    print(f"\nWrote {per_zone_output_path}")

    return summary_df, per_zone


def run_sensitivity_check(
    capacity_fractions=(0.55, 0.45, 0.35),
    fairness_caps=(4, 6, 8),
    dataset_path: str = "zones_dataset.csv",
    output_path: str = "sensitivity_results.csv",
):
    """
    Day 3 honesty/sensitivity check: reruns the optimizer under shrinking
    capacity and loosening/tightening fairness caps, records how total
    harm and worst-case outages respond. Feeds Role C's 'what happens when
    things get worse' narrative.
    """
    from optimizer import load_data, compute_available_capacity, solve_lp

    df = load_data(dataset_path)
    base_available = compute_available_capacity(df)
    peak_total_by_hour = df.groupby("hour")["demand_kwh"].sum()

    rows = []
    for cap_frac in capacity_fractions:
        available = peak_total_by_hour * cap_frac
        for fairness_cap in fairness_caps:
            status, allocation_df = solve_lp(
                df, available, max_consecutive_outage_hours=fairness_cap
            )
            merged = allocation_df.merge(
                df[
                    [
                        "zone_id",
                        "hour",
                        "zone_name",
                        "category",
                        "criticality_score",
                        "backup_capacity_kwh",
                    ]
                ],
                on=["zone_id", "hour"],
            )
            merged["effective_need"] = allocation_df["effective_need"]
            metrics = compute_metrics(merged, f"cap={cap_frac}_fair={fairness_cap}")
            rows.append(
                {
                    "capacity_fraction": cap_frac,
                    "fairness_cap_hours": fairness_cap,
                    "lp_status": status,
                    "total_harm": metrics["total_harm"],
                    "worst_case_consecutive_outage_hours": metrics[
                        "worst_case_consecutive_outage_hours"
                    ],
                    "critical_zone_unpowered_hours": metrics["critical_zone_unpowered_hours"],
                }
            )
            print(
                f"capacity={cap_frac:.2f} fairness_cap={fairness_cap}h -> "
                f"status={status}, total_harm={metrics['total_harm']}, "
                f"worst_case_outage={metrics['worst_case_consecutive_outage_hours']}h"
            )

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
    return sens_df


if __name__ == "__main__":
    build_comparison()
    print("\n--- Running sensitivity check (this takes a bit longer) ---")
    run_sensitivity_check()
