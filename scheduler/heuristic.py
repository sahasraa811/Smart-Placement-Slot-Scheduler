"""
Starting-timetable heuristic and the priority comparison (no solver needed).

The optimizer (optimizer.py) starts CBC from an existing valid schedule. The
baseline is deliberately naive, so this module tries to improve it first with a
simple local search, giving CBC a stronger starting point and a better
"never worse than this" reference.

Local search
------------
Repeatedly take ONE interview and try every other start slot it could have. A
move is only considered when it keeps the timetable valid (no student clash, the
student's buffer, no company running more interviews than it has panels). It is
accepted only if it is strictly better in the project's priority order:

    (longest excess wait, total excess wait, placement-day length)   - lexicographic

so fairness to the worst-off student is never traded for anything lower. The
search stops when no single move helps or the time budget is used up. The result
is re-checked with validate_schedule(); if anything is off the original schedule
is returned unchanged.
"""

from __future__ import annotations

import time

import pandas as pd

from scheduler import config
from scheduler.metrics import (
    build_schedule,
    schedule_to_starts,
    validate_schedule,
    waiting_times,
)


def priority_key(schedule: pd.DataFrame) -> tuple[int, int, int]:
    """
    (longest excess wait, total excess wait, day duration), all in slots.

    Tuples compare in exactly the project's priority order, so
    priority_key(a) < priority_key(b) means "a is better than b".
    """
    w = waiting_times(schedule)
    return (
        int(w["excess_wait_slots"].max()),
        int(w["excess_wait_slots"].sum()),
        int(schedule["end_slot"].max()),
    )


def improve_schedule(
    data: pd.DataFrame, schedule: pd.DataFrame, time_budget: float = 5.0
) -> pd.DataFrame:
    """
    Return a valid schedule that is never worse than `schedule` in priority
    order (the same schedule if no improving move was found).
    """
    deadline = time.time() + max(time_budget, 0.0)
    spd = config.SLOTS_PER_DAY
    blocked = config.get_blocked_slots()

    dur = {(r.student_id, r.company_id): int(r.duration_slots) for r in data.itertuples(index=False)}
    panels = {c: int(p) for c, p in data.groupby("company_id")["panels"].first().items()}
    valid_starts = {d: config.get_valid_start_slots(d) for d in set(dur.values())}
    by_student: dict[str, list[tuple[str, str]]] = {}
    for key in dur:
        by_student.setdefault(key[0], []).append(key)

    starts = schedule_to_starts(schedule)
    load = {c: [0] * config.TOTAL_SLOTS for c in panels}  # interviews running per company and slot
    for (s, c), t in starts.items():
        for u in range(t, t + dur[(s, c)]):
            load[c][u] += 1

    def excess_of(student: str, moved: tuple[tuple[str, str], int] | None = None) -> int:
        """Excess waiting slots of one student (same definition as metrics._idle_slots)."""
        per_day: dict[int, list[tuple[int, int]]] = {}
        for key in by_student[student]:
            t = moved[1] if moved is not None and moved[0] == key else starts[key]
            per_day.setdefault(t // spd, []).append((t, dur[key]))
        total = 0
        for items in per_day.values():
            busy: set[int] = set()
            rest: set[int] = set()
            for t, d in items:
                busy.update(range(t, t + d))
                rest.update(config.rest_slots(t, d))
            for u in range(min(busy), max(busy) + 1):
                if u not in busy and u not in blocked and u not in rest:
                    total += 1
        return total

    excess = {s: excess_of(s) for s in by_student}
    ends = {key: starts[key] + dur[key] for key in starts}

    def key_of(ex: dict[str, int], en: dict[tuple[str, str], int]) -> tuple[int, int, int]:
        return (max(ex.values()), sum(ex.values()), max(en.values()))

    current = key_of(excess, ends)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for key in sorted(starts):
            if time.time() >= deadline:
                break
            student, company = key
            d, old = dur[key], starts[key]

            taken: set[int] = set()  # slots the student's OTHER interviews occupy (incl. buffer)
            for other in by_student[student]:
                if other != key:
                    taken.update(config.student_occupied_slots(starts[other], dur[other]))
            for u in range(old, old + d):
                load[company][u] -= 1

            best_t, best_key = old, current
            for t in valid_starts[d]:
                if t == old:
                    continue
                if any(load[company][u] >= panels[company] for u in range(t, t + d)):
                    continue  # company has no free panel
                if taken.intersection(config.student_occupied_slots(t, d)):
                    continue  # clash or buffer broken for the student
                ex2 = dict(excess)
                ex2[student] = excess_of(student, (key, t))
                en2 = dict(ends)
                en2[key] = t + d
                candidate = key_of(ex2, en2)
                if candidate < best_key:
                    best_t, best_key = t, candidate

            for u in range(best_t, best_t + d):
                load[company][u] += 1
            if best_t != old:
                starts[key] = best_t
                ends[key] = best_t + d
                excess[student] = excess_of(student)
                current = best_key
                improved = True

    candidate_schedule = build_schedule(data, starts)
    if validate_schedule(candidate_schedule, data):
        return schedule  # should never happen; never hand back an invalid timetable
    if priority_key(candidate_schedule) < priority_key(schedule):
        return candidate_schedule
    return schedule
