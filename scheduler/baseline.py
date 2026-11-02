"""
Baseline schedule: greedy first-come-first-served (FCFS).

Interviews are taken in a fixed order (by company, then student ID) and each is
placed at the EARLIEST start slot that is free for both the student and the
company. The result is clash-free and respects the same required student buffer
(config.BUFFER_MINUTES) as the optimizer, but it never looks at waiting time,
which makes it a fair "what you get without optimization" reference.

Run from the project root:

    python -m scheduler.baseline
"""

from __future__ import annotations

import pandas as pd

from scheduler import config
from scheduler.data_generator import load_sample_data
from scheduler.metrics import build_schedule, compute_metrics, assert_valid


def baseline_schedule(data: pd.DataFrame) -> pd.DataFrame:
    """Build the greedy FCFS schedule for the given problem data."""
    panels_of = data.groupby("company_id")["panels"].first().to_dict()

    company_load = {c: [0] * config.TOTAL_SLOTS for c in panels_of}   # interviews running per slot
    student_busy: dict[str, list[bool]] = {}                          # slot taken for this student?
    starts: dict[tuple[str, str], int] = {}

    ordered = data.sort_values(["company_id", "student_id"])
    for r in ordered.itertuples(index=False):
        dur = int(r.duration_slots)
        busy = student_busy.setdefault(r.student_id, [False] * config.TOTAL_SLOTS)
        load = company_load[r.company_id]
        cap = panels_of[r.company_id]

        placed = None
        for start in config.get_valid_start_slots(dur):
            if any(load[t] >= cap for t in range(start, start + dur)):
                continue  # company has no free panel
            if any(busy[t] for t in config.student_occupied_slots(start, dur)):
                continue  # student is busy (or in the buffer)
            placed = start
            break

        if placed is None:
            raise RuntimeError(
                f"Baseline could not place {r.student_id}/{r.company_id}. "
                "Add days or reduce the number of shortlists."
            )

        starts[(r.student_id, r.company_id)] = placed
        for t in range(placed, placed + dur):
            load[t] += 1
        for t in config.student_occupied_slots(placed, dur):
            busy[t] = True

    schedule = build_schedule(data, starts)
    assert_valid(schedule, data)
    return schedule


def main() -> None:
    data = load_sample_data()
    schedule = baseline_schedule(data)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(config.BASELINE_SCHEDULE_PATH, index=False)

    print(f"Baseline schedule saved to {config.BASELINE_SCHEDULE_PATH}")
    for key, value in compute_metrics(schedule).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
