"""
Schedule format, validation and metrics for the Smart Placement Slot Scheduler.

Everything downstream (baseline.py, optimizer.py, tests, app.py) shares ONE
schedule format: a DataFrame with one row per interview and these columns.

    student_id, student_name, company_id, company_name, panel,
    day, start_slot, end_slot, duration_slots, start_time, end_time

* start_slot / end_slot are global slot indices (see config.py). end_slot is
  EXCLUSIVE: an interview occupies slots start_slot .. end_slot - 1.
* panel is the room number (1..panels) the interview runs in.

Waiting time and the buffer
---------------------------
Zero waiting is unrealistic, so every student gets a required buffer
(config.BUFFER_MINUTES, 15 min by default) after each interview. For one
student on one day:

    gap time      = idle slots between their first start and last end
                    (lunch slots are not counted)
    excess wait   = gap time that is NOT inside the buffer that follows an
                    interview

i.e. for every pair of consecutive interviews, excess = max(0, gap - buffer).
A 15-minute gap is therefore "free", a 45-minute gap has 30 minutes of excess
waiting. Time before the first interview and after the last one is never
waiting, and nothing spans overnight (each day is measured separately).

The optimizer minimizes EXCESS waiting, which is the same quantity
waiting_times() reports, so the optimizer and the dashboard always agree.
Gap time is still reported (total_gap_minutes) as supporting information.
"""

from __future__ import annotations

import pandas as pd

from scheduler import config

SCHEDULE_COLUMNS = [
    "student_id",
    "student_name",
    "company_id",
    "company_name",
    "panel",
    "day",
    "start_slot",
    "end_slot",
    "duration_slots",
    "start_time",
    "end_time",
]


# ---------------------------------------------------------------------------
# Building a schedule
# ---------------------------------------------------------------------------
def build_schedule(data: pd.DataFrame, starts: dict[tuple[str, str], int]) -> pd.DataFrame:
    """
    Turn {(student_id, company_id): start_slot} into a full schedule DataFrame,
    and assign each interview to a panel (room).

    Panels are assigned greedily by start time. For interval scheduling this
    always succeeds when no more than `panels` interviews overlap at any
    moment, which is exactly the company-clash rule.
    """
    rows = []
    for r in data.itertuples(index=False):
        key = (r.student_id, r.company_id)
        if key not in starts:
            raise KeyError(f"No start slot for {key}")
        start = int(starts[key])
        dur = int(r.duration_slots)
        rows.append(
            {
                "student_id": r.student_id,
                "student_name": r.student_name,
                "company_id": r.company_id,
                "company_name": r.company_name,
                "panel": 0,
                "day": config.slot_to_day(start),
                "start_slot": start,
                "end_slot": start + dur,
                "duration_slots": dur,
                "start_time": config.slot_to_clock(start),
                "end_time": config.slot_to_clock(start, dur),
            }
        )
    sched = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)

    panels_of = data.groupby("company_id")["panels"].first().to_dict()
    for company, grp in sched.groupby("company_id"):
        free_at = [0] * int(panels_of[company])  # slot at which each panel becomes free
        for idx in grp.sort_values(["start_slot", "student_id"]).index:
            start = sched.at[idx, "start_slot"]
            p = next((k for k, t in enumerate(free_at) if t <= start), None)
            if p is None:  # only possible if the schedule has a company clash
                p = min(range(len(free_at)), key=free_at.__getitem__)
            free_at[p] = sched.at[idx, "end_slot"]
            sched.at[idx, "panel"] = p + 1

    return (
        sched.sort_values(["start_slot", "company_id", "student_id"])
        .reset_index(drop=True)
    )


def schedule_to_starts(schedule: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Inverse of build_schedule: {(student_id, company_id): start_slot}."""
    return {
        (r.student_id, r.company_id): int(r.start_slot)
        for r in schedule.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schedule(schedule: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    """
    Check a schedule against the problem data. Returns a list of human-readable
    violations; an EMPTY list means the schedule is valid.

    Checks:
      1. every shortlisted (student, company) pair is scheduled exactly once,
         and nothing else is scheduled;
      2. each interview has the right length and a legal start (inside the
         working day, not overlapping lunch);
      3. no student has two interviews at once, or fewer than BUFFER_SLOTS
         free slots between two interviews on the same day;
      4. no company runs more interviews at once than it has panels;
      5. no two interviews share a panel at the same time.
    """
    errors: list[str] = []

    # 1. coverage -----------------------------------------------------------
    expected = set(zip(data["student_id"], data["company_id"]))
    if schedule.duplicated(["student_id", "company_id"]).any():
        dup = schedule[schedule.duplicated(["student_id", "company_id"], keep=False)]
        errors.append(f"Interviews scheduled more than once: {sorted(set(zip(dup.student_id, dup.company_id)))}")
    got = set(zip(schedule["student_id"], schedule["company_id"]))
    if expected - got:
        errors.append(f"Missing interviews: {sorted(expected - got)}")
    if got - expected:
        errors.append(f"Interviews that were never shortlisted: {sorted(got - expected)}")

    duration_of = data.groupby("company_id")["duration_slots"].first().to_dict()
    panels_of = data.groupby("company_id")["panels"].first().to_dict()
    blocked = config.get_blocked_slots()

    # 2. length and legal start ---------------------------------------------
    for r in schedule.itertuples(index=False):
        label = f"{r.student_id}/{r.company_id}"
        want = duration_of.get(r.company_id)
        if want is not None and (r.end_slot - r.start_slot != want or r.duration_slots != want):
            errors.append(f"{label}: wrong duration (expected {want} slots)")
        if r.start_slot < 0 or r.end_slot > config.TOTAL_SLOTS:
            errors.append(f"{label}: outside the scheduling horizon")
        elif r.start_slot % config.SLOTS_PER_DAY + (r.end_slot - r.start_slot) > config.SLOTS_PER_DAY:
            errors.append(f"{label}: runs past the end of the day")
        elif any(s in blocked for s in range(r.start_slot, r.end_slot)):
            errors.append(f"{label}: overlaps the lunch break")

    # 3. student clashes ----------------------------------------------------
    buffer = config.BUFFER_SLOTS
    for student, grp in schedule.groupby("student_id"):
        ivs = sorted(zip(grp.start_slot, grp.end_slot, grp.company_id))
        for (s1, e1, c1), (s2, e2, c2) in zip(ivs, ivs[1:]):
            if s2 < e1:
                errors.append(f"Student clash: {student} is in {c1} and {c2} at the same time")
            elif buffer and s2 - e1 < buffer and config.slot_to_day(s1) == config.slot_to_day(s2):
                errors.append(
                    f"Student {student}: less than the required {config.BUFFER_MINUTES}-minute "
                    f"buffer between {c1} and {c2}"
                )

    # 4. company clashes (sweep over slots) ---------------------------------
    for company, grp in schedule.groupby("company_id"):
        load: dict[int, int] = {}
        for r in grp.itertuples(index=False):
            for t in range(r.start_slot, r.end_slot):
                load[t] = load.get(t, 0) + 1
        cap = panels_of.get(company, 1)
        bad = sorted(t for t, n in load.items() if n > cap)
        if bad:
            errors.append(
                f"Company clash: {company} has more than {cap} interview(s) at once "
                f"(first at slot {bad[0]})"
            )

    # 5. panel consistency --------------------------------------------------
    for (company, panel), grp in schedule.groupby(["company_id", "panel"]):
        ivs = sorted(zip(grp.start_slot, grp.end_slot))
        if any(s2 < e1 for (_, e1), (s2, _) in zip(ivs, ivs[1:])):
            errors.append(f"Panel clash: {company} panel {panel} has overlapping interviews")
        if panel < 1 or panel > panels_of.get(company, 1):
            errors.append(f"{company}: panel {panel} does not exist")

    return errors


def assert_valid(schedule: pd.DataFrame, data: pd.DataFrame) -> None:
    """Raise ValueError listing every violation (handy in tests and scripts)."""
    errors = validate_schedule(schedule, data)
    if errors:
        raise ValueError("Invalid schedule:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Waiting time and summary metrics
# ---------------------------------------------------------------------------
def _idle_slots(day_grp: pd.DataFrame, blocked: set[int]) -> list[tuple[int, str]]:
    """
    Idle slots of ONE student on ONE day, between their first start and last end,
    as (slot, kind). kind is "buffer" when the slot lies inside the buffer that
    follows an interview, otherwise "excess". Lunch slots are left out.

    This is the single place where waiting is defined: waiting_times() counts
    these slots and waiting_segments() draws them.
    """
    busy: set[int] = set()
    buffer: set[int] = set()
    for r in day_grp.itertuples(index=False):
        busy.update(range(r.start_slot, r.end_slot))
        buffer.update(config.rest_slots(r.start_slot, r.end_slot - r.start_slot))
    first, last = min(busy), max(busy)
    return [
        (t, "buffer" if t in buffer else "excess")
        for t in range(first, last + 1)
        if t not in busy and t not in blocked
    ]


def waiting_times(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Per-student waiting (see the module docstring for the definitions).

    Returns one row per student: student_id, student_name, num_interviews,
    interview_slots, gap_slots, gap_minutes, excess_wait_slots,
    excess_wait_minutes, first_start, last_end.
    """
    blocked = config.get_blocked_slots()
    rows = []
    for (student_id, student_name), grp in schedule.groupby(["student_id", "student_name"]):
        gap = 0
        excess = 0
        for _, day_grp in grp.groupby("day"):
            idle = _idle_slots(day_grp, blocked)
            gap += len(idle)
            excess += sum(1 for _, kind in idle if kind == "excess")
        rows.append(
            {
                "student_id": student_id,
                "student_name": student_name,
                "num_interviews": len(grp),
                "interview_slots": int(grp["duration_slots"].sum()),
                "gap_slots": gap,
                "gap_minutes": gap * config.SLOT_MINUTES,
                "excess_wait_slots": excess,
                "excess_wait_minutes": excess * config.SLOT_MINUTES,
                "first_start": int(grp["start_slot"].min()),
                "last_end": int(grp["end_slot"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("student_id").reset_index(drop=True)


SEGMENT_COLUMNS = [
    "student_id", "student_name", "day", "kind",
    "start_slot", "end_slot", "slots", "minutes", "start_time", "end_time",
]


def waiting_segments(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    The idle stretches between each student's interviews, for drawing and for
    per-student detail. One row per stretch; neighbouring slots of the same kind
    are merged.

    kind = "buffer": the required buffer after an interview (not counted as waiting)
    kind = "excess": waiting beyond the buffer (what the optimizer minimizes)

    Lunch slots are not included. The "excess" rows add up to exactly
    waiting_times()["excess_wait_slots"], and buffer + excess to gap_slots.
    """
    blocked = config.get_blocked_slots()
    rows = []
    for (student_id, student_name), grp in schedule.groupby(["student_id", "student_name"]):
        for day, day_grp in grp.groupby("day"):
            runs: list[list] = []  # [kind, first_slot, end_slot_exclusive]
            for t, kind in _idle_slots(day_grp, blocked):
                if runs and runs[-1][0] == kind and runs[-1][2] == t:
                    runs[-1][2] = t + 1
                else:
                    runs.append([kind, t, t + 1])
            for kind, start, end in runs:
                rows.append(
                    {
                        "student_id": student_id,
                        "student_name": student_name,
                        "day": int(day),
                        "kind": kind,
                        "start_slot": start,
                        "end_slot": end,
                        "slots": end - start,
                        "minutes": (end - start) * config.SLOT_MINUTES,
                        "start_time": config.slot_to_clock(start),
                        "end_time": config.slot_to_clock(start, end - start),
                    }
                )
    out = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
    return out.sort_values(["student_id", "start_slot"]).reset_index(drop=True)


def compute_metrics(schedule: pd.DataFrame) -> dict[str, float]:
    """Headline numbers used by the dashboard and the comparison."""
    w = waiting_times(schedule)
    multi = w[w["num_interviews"] > 1]  # a single interview can never involve waiting
    return {
        "total_excess_wait_minutes": int(w["excess_wait_minutes"].sum()),
        "avg_excess_wait_minutes": round(w["excess_wait_minutes"].mean(), 2),
        "max_excess_wait_minutes": int(w["excess_wait_minutes"].max()),
        "students_with_excess_wait": int((w["excess_wait_slots"] > 0).sum()),
        "students_with_multiple_interviews": int(len(multi)),
        "total_gap_minutes": int(w["gap_minutes"].sum()),
        "makespan_minutes": int(
            (schedule["end_slot"].max() - schedule["start_slot"].min()) * config.SLOT_MINUTES
        ),
        "last_finish_time": config.slot_to_clock(int(schedule["end_slot"].max()) - 1, 1),
    }


def compare_schedules(
    baseline: pd.DataFrame, optimized: pd.DataFrame, labels: tuple[str, str] = ("Baseline", "Optimized")
) -> pd.DataFrame:
    """
    Side-by-side metrics table with the improvement of `optimized` over
    `baseline` (positive % = optimized is lower, i.e. better).
    """
    m_base, m_opt = compute_metrics(baseline), compute_metrics(optimized)
    rows = []
    for key in m_base:
        b, o = m_base[key], m_opt[key]
        numeric = isinstance(b, (int, float)) and isinstance(o, (int, float))
        improvement = round(100 * (b - o) / b, 1) if numeric and b else (0.0 if numeric else None)
        rows.append({"metric": key, labels[0]: b, labels[1]: o, "improvement_%": improvement})
    return pd.DataFrame(rows)
