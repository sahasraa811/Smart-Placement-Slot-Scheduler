"""
Scalability experiments for the optimizer (stage-by-stage timings).

Runs the baseline and the optimizer on several dataset sizes and writes one row
per run to results/experiments.csv. It does NOT touch data/ or the saved
schedules, so it is safe to run at any time.

Run from the project root:

    python -m scheduler.experiments                 # 20/5, 40/8, 60/12, 80/12
    python -m scheduler.experiments --limit 120     # 120 s solver budget per run
"""

from __future__ import annotations

import argparse

import pandas as pd

from scheduler import config
from scheduler.baseline import baseline_schedule
from scheduler.data_generator import generate_sample_data
from scheduler.metrics import compute_metrics
from scheduler.optimizer import optimize_schedule

# (students, companies, label)
SCENARIOS = [
    (20, 5, "Small"),
    (40, 8, "Medium"),
    (60, 12, "Larger"),
    (80, 12, "Scalability"),
]


def run_experiment(students: int, companies: int, seed: int, time_limit: int) -> dict:
    data = generate_sample_data(students, companies, seed)
    baseline = baseline_schedule(data)
    result = optimize_schedule(data, time_limit=time_limit, warm_start=baseline)

    mb, mo = compute_metrics(baseline), compute_metrics(result.schedule)
    row = {
        "students": students,
        "companies": companies,
        "baseline_total_excess_min": mb["total_excess_wait_minutes"],
        "optimized_total_excess_min": mo["total_excess_wait_minutes"],
        "baseline_longest_excess_min": mb["max_excess_wait_minutes"],
        "optimized_longest_excess_min": mo["max_excess_wait_minutes"],
        "baseline_day_min": mb["makespan_minutes"],
        "optimized_day_min": mo["makespan_minutes"],
        "solver_status": result.status,
        "total_seconds": result.solve_seconds,
        "fallback_used": result.fallback_used,
        "variables": result.num_variables,
        "binaries": result.num_binaries,
        "constraints": result.num_constraints,
    }
    for snap in result.trajectory:
        row[f"{snap.key}_longest_min"] = snap.longest * config.SLOT_MINUTES
        row[f"{snap.key}_total_min"] = snap.total * config.SLOT_MINUTES
        row[f"{snap.key}_day_min"] = snap.day * config.SLOT_MINUTES
    for i, stage in enumerate(result.stages, start=1):
        row[f"stage{i}_status"] = stage.status
        row[f"stage{i}_seconds"] = stage.seconds
        row[f"stage{i}_limit"] = stage.time_limit
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--limit", type=int, default=config.SOLVER_TIME_LIMIT_SEC,
                        help="solver time limit per run, in seconds")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    rows = []
    for students, companies, label in SCENARIOS:
        print(f"{label}: {students} students, {companies} companies ...", flush=True)
        try:
            rows.append(run_experiment(students, companies, args.seed, args.limit))
        except (ValueError, RuntimeError) as err:
            print(f"  failed: {err}")

    table = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "experiments.csv"
    table.to_csv(out, index=False)
    print(table.T.to_string(header=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
