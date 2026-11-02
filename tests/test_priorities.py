"""
Priority-order tests for the optimizer (needs PuLP + CBC).

The optimizer must follow this order, strictly:

    1. no clashes            (hard)
    2. 15-minute buffer      (hard)
    3. smallest possible LONGEST excess wait
    4. then smallest possible TOTAL excess wait
    5. then the SHORTEST placement day

"Strictly" means a lower priority is never bought with a worse higher one. The
tests check that against an exhaustive search: on tiny problems every valid
schedule can be listed, so the true best (longest wait, total wait, day length)
is known and the optimizer has to hit it exactly.

Run from the project root:

    python -m pytest tests/test_priorities.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from scheduler import config
from scheduler.baseline import baseline_schedule
from scheduler.metrics import compute_metrics, validate_schedule, waiting_times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_data(shortlists: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Hand-made problem data. Each item is (student_id, company_id, duration_slots, panels)."""
    return pd.DataFrame(
        [
            {
                "student_id": s,
                "student_name": f"Student {s}",
                "company_id": c,
                "company_name": f"Company {c}",
                "duration_slots": d,
                "panels": p,
            }
            for s, c, d, p in shortlists
        ]
    )


def short_day(monkeypatch, slots: int, buffer_slots: int = 1) -> None:
    """A one-day timetable with `slots` slots and a buffer of `buffer_slots` (derived values are set at import)."""
    monkeypatch.setattr(config, "NUM_DAYS", 1)
    monkeypatch.setattr(config, "SLOTS_PER_DAY", slots)
    monkeypatch.setattr(config, "TOTAL_SLOTS", slots)
    monkeypatch.setattr(config, "BUFFER_SLOTS", buffer_slots)
    monkeypatch.setattr(config, "BUFFER_MINUTES", buffer_slots * config.SLOT_MINUTES)


def exhaustive_best(data: pd.DataFrame) -> tuple[int, int, int]:
    """
    Try EVERY valid schedule (one-panel or multi-panel companies, one day) and return the best
    (longest excess wait, total excess wait, day duration) in slots, the day starting at slot 0, compared in that order.
    Written from the rules, independently of metrics.py and the optimizer.
    """
    slots, buffer = config.TOTAL_SLOTS, config.BUFFER_SLOTS
    items = [(r.student_id, r.company_id, int(r.duration_slots), int(r.panels)) for r in data.itertuples()]
    best: list[tuple[int, int, int]] = []
    load: dict[tuple[str, int], int] = {}
    mine: dict[str, list[tuple[int, int]]] = {}
    spans: list[tuple[int, int]] = []

    def place(i: int) -> None:
        if i == len(items):
            excess = []
            for ivs in mine.values():
                ivs = sorted(ivs)
                excess.append(sum(b - a_end - buffer for (_, a_end), (b, _) in zip(ivs, ivs[1:])))
            span = max(e for _, e in spans)  # day start -> last interview ends
            cand = (max(excess), sum(excess), span)
            if not best or cand < best[0]:
                best[:] = [cand]
            return
        student, company, dur, panels = items[i]
        for start in range(slots - dur + 1):
            end = start + dur
            if any(load.get((company, t), 0) >= panels for t in range(start, end)):
                continue  # company has no free panel
            if any(not (end + buffer <= a or b + buffer <= start) for a, b in mine.get(student, [])):
                continue  # clash, or less than the buffer
            for t in range(start, end):
                load[(company, t)] = load.get((company, t), 0) + 1
            mine.setdefault(student, []).append((start, end))
            spans.append((start, end))
            place(i + 1)
            spans.pop()
            mine[student].pop()
            for t in range(start, end):
                load[(company, t)] -= 1

    place(0)
    assert best, "the test instance has no valid schedule"
    return best[0]


def achieved(result) -> tuple[int, int, int]:
    return (
        result.max_excess_wait_slots,
        result.total_excess_wait_slots,
        result.day_duration_slots,
    )


# A 10-slot day where the priorities pull in different directions (checked by exhaustive search):
#   * the lowest TOTAL excess wait is 2, but it needs one student to wait 2 slots;
#   * the lowest LONGEST wait is 1, and then the total can be no lower than 4.
# Priority 3 (longest wait) beats priority 4 (total), so the right answer is (1, 4, 10).
TRADEOFF_INSTANCE = [
    ("S1", "C2", 2, 1), ("S1", "C1", 2, 1),
    ("S2", "C2", 2, 1), ("S2", "C1", 2, 1),
    ("S3", "C2", 2, 1), ("S3", "C3", 3, 1),
    ("S4", "C1", 2, 1), ("S4", "C2", 2, 1),
    ("S5", "C2", 2, 1), ("S5", "C1", 2, 1),
]


# ---------------------------------------------------------------------------
# Strict priority order
# ---------------------------------------------------------------------------
def test_longest_wait_beats_total_wait(monkeypatch):
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    short_day(monkeypatch, 10)
    data = make_data(TRADEOFF_INSTANCE)
    assert exhaustive_best(data) == (1, 4, 10)  # the premise of this test

    result = optimize_schedule(data, time_limit=30)
    assert result.status == "Optimal"
    assert achieved(result) == (1, 4, 10)  # NOT (2, 2, 10): a lower total may not cost a longer worst wait
    assert validate_schedule(result.schedule, data) == []


def test_optimizer_matches_exhaustive_search_on_small_problems(monkeypatch):
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    short_day(monkeypatch, 10)
    instances = [
        TRADEOFF_INSTANCE,
        # nobody has to wait: only the day length is left to optimize
        [("S1", "A", 2, 1), ("S1", "B", 3, 1), ("S2", "A", 2, 1), ("S2", "B", 3, 1)],
        # a two-panel company
        [("S1", "A", 3, 2), ("S2", "A", 3, 2), ("S3", "A", 3, 2), ("S3", "B", 2, 1), ("S1", "B", 2, 1)],
        # three interviews for one student, with a shared bottleneck company
        [("S1", "A", 2, 1), ("S1", "B", 2, 1), ("S1", "C", 2, 1), ("S2", "A", 2, 1), ("S3", "A", 2, 1), ("S2", "B", 2, 1)],
    ]
    for shortlists in instances:
        data = make_data(shortlists)
        result = optimize_schedule(data, time_limit=30)
        assert result.status == "Optimal", shortlists
        assert achieved(result) == exhaustive_best(data), shortlists
        assert validate_schedule(result.schedule, data) == [], shortlists


def test_reported_numbers_are_measured_on_the_returned_schedule():
    pytest.importorskip("pulp")
    from scheduler.data_generator import generate_sample_data
    from scheduler.optimizer import optimize_schedule

    result = optimize_schedule(generate_sample_data(num_students=6, num_companies=3, seed=2), time_limit=30)
    w = waiting_times(result.schedule)
    m = compute_metrics(result.schedule)
    assert result.max_excess_wait_slots == int(w["excess_wait_slots"].max())
    assert result.total_excess_wait_slots == int(w["excess_wait_slots"].sum())
    assert result.day_duration_slots == int(result.schedule["end_slot"].max())
    assert result.schedule["start_slot"].min() == 0  # default setup: the day is packed to its start
    assert result.day_duration_slots * config.SLOT_MINUTES == m["makespan_minutes"]


def test_stages_run_in_priority_order():
    pytest.importorskip("pulp")
    from scheduler.data_generator import generate_sample_data
    from scheduler.optimizer import optimize_schedule

    result = optimize_schedule(generate_sample_data(num_students=6, num_companies=3, seed=2), time_limit=30)
    assert [s.name for s in result.stages] == ["Longest excess wait", "Total excess wait", "Day duration"]
    if result.status == "Optimal":
        # every stage was proven optimal, so each locked value is exactly what the final schedule has
        assert [s.value for s in result.stages] == [
            result.max_excess_wait_slots,
            result.total_excess_wait_slots,
            result.day_duration_slots,
        ]


# ---------------------------------------------------------------------------
# The hard rules (priorities 1 and 2) are never traded away
# ---------------------------------------------------------------------------
def test_hard_rules_hold_even_when_they_force_waiting(monkeypatch):
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    short_day(monkeypatch, 10)
    data = make_data(TRADEOFF_INSTANCE)
    result = optimize_schedule(data, time_limit=30)
    errors = validate_schedule(result.schedule, data)
    assert errors == []  # no student/company clash, buffer kept between every pair of interviews
    for _, grp in result.schedule.groupby("student_id"):
        rows = sorted(zip(grp.start_slot, grp.end_slot))
        for (_, end), (start, _) in zip(rows, rows[1:]):
            assert start - end >= config.BUFFER_SLOTS


# ---------------------------------------------------------------------------
# The optimizer is never worse than the baseline, in priority order
# ---------------------------------------------------------------------------
def test_optimized_is_not_worse_than_baseline_in_priority_order():
    pytest.importorskip("pulp")
    from scheduler.data_generator import generate_sample_data
    from scheduler.optimizer import optimize_schedule

    for seed in (1, 2, 3):
        data = generate_sample_data(num_students=6, num_companies=3, seed=seed)
        base = baseline_schedule(data)
        opt = optimize_schedule(data, time_limit=30, warm_start=base)
        mb, mo = compute_metrics(base), compute_metrics(opt.schedule)
        assert (mo["max_excess_wait_minutes"], mo["total_excess_wait_minutes"]) <= (
            mb["max_excess_wait_minutes"], mb["total_excess_wait_minutes"]
        ), f"seed {seed}"


# ---------------------------------------------------------------------------
# Day-length lower bound (used to cut the search)
# ---------------------------------------------------------------------------
def test_day_duration_lower_bound_is_the_busiest_company_or_student(monkeypatch):
    pytest.importorskip("pulp")
    from scheduler.optimizer import _day_duration_lower_bound

    short_day(monkeypatch, 12, buffer_slots=1)
    # Company A: 3 interviews x 2 slots on one panel = 6 slots; each student needs far less.
    assert _day_duration_lower_bound(make_data([("S1", "A", 2, 1), ("S2", "A", 2, 1), ("S3", "A", 2, 1)])) == 6
    # Two panels halve that: 3 x 2 / 2 = 3 slots.
    assert _day_duration_lower_bound(make_data([("S1", "A", 2, 2), ("S2", "A", 2, 2), ("S3", "A", 2, 2)])) == 3
    # One student with 3 interviews of 2 slots needs 6 + 2 buffers = 8 slots, more than any company.
    assert _day_duration_lower_bound(make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1), ("S1", "C", 2, 1)])) == 8
