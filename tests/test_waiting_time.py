"""
Waiting-time tests.

Definitions being tested (see scheduler/metrics.py):

    gap time    = idle slots between a student's first start and last end
    excess wait = gap time that is NOT inside the required buffer (15 min)
                  that follows an interview, i.e. max(0, gap - buffer) for
                  every pair of consecutive interviews

Slots below are 15 minutes long, so "1 slot" = 15 min. Most tests pin the
buffer to 1 slot so they do not depend on what config.py currently says.

Run from the project root:

    python -m pytest tests/test_waiting_time.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from scheduler import config
from scheduler.baseline import baseline_schedule
from scheduler.data_generator import generate_sample_data
from scheduler.metrics import build_schedule, compute_metrics, waiting_segments, waiting_times


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


def set_buffer(monkeypatch, minutes: int) -> None:
    monkeypatch.setattr(config, "BUFFER_MINUTES", minutes)
    monkeypatch.setattr(config, "BUFFER_SLOTS", minutes // config.SLOT_MINUTES)


def student_waiting(starts: dict[tuple[str, str], int], durations: dict[str, int]) -> pd.Series:
    """
    Build a schedule for ONE student (S1) from {(S1, company): start_slot} and
    return that student's row from waiting_times().
    """
    data = make_data([("S1", c, durations[c], 1) for (_, c) in starts])
    schedule = build_schedule(data, starts)
    return waiting_times(schedule).iloc[0]


D2 = {"A": 2, "B": 2, "C": 2}  # three companies with 2-slot (30 min) interviews


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------
def test_single_interview_never_waits(monkeypatch):
    set_buffer(monkeypatch, 15)
    row = student_waiting({("S1", "A"): 5}, D2)
    assert row["gap_slots"] == 0
    assert row["excess_wait_slots"] == 0


def test_gap_equal_to_the_buffer_is_free(monkeypatch):
    set_buffer(monkeypatch, 15)
    # A occupies slots 0-1, slot 2 is idle, B occupies slots 3-4
    row = student_waiting({("S1", "A"): 0, ("S1", "B"): 3}, D2)
    assert row["gap_slots"] == 1
    assert row["excess_wait_slots"] == 0
    assert row["excess_wait_minutes"] == 0


def test_only_the_part_beyond_the_buffer_counts(monkeypatch):
    set_buffer(monkeypatch, 15)
    # A: 0-1, idle 2,3,4,5 (60 min), B: 6-7  -> 15 min buffer + 45 min excess
    row = student_waiting({("S1", "A"): 0, ("S1", "B"): 6}, D2)
    assert row["gap_slots"] == 4
    assert row["gap_minutes"] == 60
    assert row["excess_wait_slots"] == 3
    assert row["excess_wait_minutes"] == 45


def test_excess_adds_up_over_several_gaps(monkeypatch):
    set_buffer(monkeypatch, 15)
    # A: 0-1 | idle 2,3 (gap 2, excess 1) | B: 4-5 | idle 6,7,8 (gap 3, excess 2) | C: 9-10
    row = student_waiting({("S1", "A"): 0, ("S1", "B"): 4, ("S1", "C"): 9}, D2)
    assert row["gap_slots"] == 5
    assert row["excess_wait_slots"] == 3


def test_time_before_first_and_after_last_interview_is_not_waiting(monkeypatch):
    set_buffer(monkeypatch, 15)
    early = student_waiting({("S1", "A"): 0, ("S1", "B"): 6}, D2)
    late = student_waiting({("S1", "A"): 20, ("S1", "B"): 26}, D2)  # same gap, much later in the day
    assert early["excess_wait_slots"] == late["excess_wait_slots"] == 3
    assert late["first_start"] == 20  # idle time before slot 20 was not counted


def test_waiting_measures_gaps_not_slots_used(monkeypatch):
    """The project's key point: a late-finishing schedule is not automatically a waiting one."""
    set_buffer(monkeypatch, 15)
    tight_but_late = student_waiting({("S1", "A"): 28, ("S1", "B"): 31}, D2)   # ends at slot 33
    spread_but_early = student_waiting({("S1", "A"): 0, ("S1", "B"): 8}, D2)  # ends at slot 10
    assert tight_but_late["excess_wait_slots"] == 0
    assert spread_but_early["excess_wait_slots"] > 0
    assert tight_but_late["last_end"] > spread_but_early["last_end"]


def test_zero_buffer_makes_every_gap_excess(monkeypatch):
    set_buffer(monkeypatch, 0)
    back_to_back = student_waiting({("S1", "A"): 0, ("S1", "B"): 2}, D2)
    assert back_to_back["gap_slots"] == 0 and back_to_back["excess_wait_slots"] == 0

    two_slot_gap = student_waiting({("S1", "A"): 0, ("S1", "B"): 4}, D2)
    assert two_slot_gap["gap_slots"] == 2
    assert two_slot_gap["excess_wait_slots"] == 2


def test_bigger_buffer_absorbs_more_of_the_gap(monkeypatch):
    set_buffer(monkeypatch, 30)  # 2 slots
    row = student_waiting({("S1", "A"): 0, ("S1", "B"): 6}, D2)  # gap of 4 slots
    assert row["gap_slots"] == 4
    assert row["excess_wait_slots"] == 2


# ---------------------------------------------------------------------------
# Lunch is not waiting
# ---------------------------------------------------------------------------
def test_gap_made_only_of_lunch_is_not_waiting(monkeypatch):
    set_buffer(monkeypatch, 15)
    monkeypatch.setattr(config, "LUNCH_BREAK", ("12:00", "13:00"))  # slots 12-15
    # A ends exactly when lunch starts (slot 12), B starts when lunch ends (slot 16)
    row = student_waiting({("S1", "A"): 10, ("S1", "B"): 16}, D2)
    assert row["gap_slots"] == 0
    assert row["excess_wait_slots"] == 0


def test_lunch_slots_are_skipped_but_other_idle_time_still_counts(monkeypatch):
    set_buffer(monkeypatch, 15)
    monkeypatch.setattr(config, "LUNCH_BREAK", ("12:00", "13:00"))
    # A: 8-9, idle 10 and 11, lunch 12-15 (ignored), B: 16-17
    row = student_waiting({("S1", "A"): 8, ("S1", "B"): 16}, D2)
    assert row["gap_slots"] == 2
    assert row["excess_wait_slots"] == 1  # slot 10 is the buffer, slot 11 is excess


# ---------------------------------------------------------------------------
# Waiting segments (what the dashboard draws)
# ---------------------------------------------------------------------------
def test_segments_split_a_gap_into_buffer_and_excess(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 6})  # idle slots 2,3,4,5
    seg = waiting_segments(schedule)
    assert list(seg["kind"]) == ["buffer", "excess"]
    assert list(seg["start_slot"]) == [2, 3]
    assert list(seg["end_slot"]) == [3, 6]
    assert list(seg["minutes"]) == [15, 45]


def test_a_gap_equal_to_the_buffer_has_no_excess_segment(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    seg = waiting_segments(build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 3}))
    assert list(seg["kind"]) == ["buffer"]


def test_segments_add_up_to_waiting_times_and_never_overlap_interviews():
    data = generate_sample_data()
    schedule = baseline_schedule(data)
    w = waiting_times(schedule).set_index("student_id")
    seg = waiting_segments(schedule)

    excess = seg[seg["kind"] == "excess"].groupby("student_id")["slots"].sum()
    total = seg.groupby("student_id")["slots"].sum()
    for student_id, row in w.iterrows():
        assert excess.get(student_id, 0) == row["excess_wait_slots"], student_id
        assert total.get(student_id, 0) == row["gap_slots"], student_id

    for r in seg.itertuples(index=False):  # a waiting stretch is never on top of an interview
        busy = schedule[schedule["student_id"] == r.student_id]
        assert not ((busy["start_slot"] < r.end_slot) & (busy["end_slot"] > r.start_slot)).any()


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
def test_compute_metrics_totals(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data(
        [
            ("S1", "A", 2, 1), ("S1", "B", 2, 1),   # S1 will have 45 min excess
            ("S2", "A", 2, 1), ("S2", "C", 2, 1),   # S2 will have 0 excess
            ("S3", "C", 2, 1),                      # S3 has one interview, cannot wait
        ]
    )
    starts = {
        ("S1", "A"): 0, ("S1", "B"): 6,
        ("S2", "A"): 3, ("S2", "C"): 6,
        ("S3", "C"): 10,
    }
    m = compute_metrics(build_schedule(data, starts))
    assert m["total_excess_wait_minutes"] == 45
    assert m["students_with_excess_wait"] == 1
    assert m["max_excess_wait_minutes"] == 45
    assert m["students_with_multiple_interviews"] == 2
    assert m["total_gap_minutes"] == 60 + 15  # S1 gap 60, S2 gap 15 (just the buffer)


def test_excess_is_never_more_than_gap_and_gap_minus_excess_is_at_most_the_buffers():
    data = generate_sample_data()
    schedule = baseline_schedule(data)
    w = waiting_times(schedule)

    assert (w["excess_wait_slots"] >= 0).all()
    assert (w["excess_wait_slots"] <= w["gap_slots"]).all()
    # every student can have at most (interviews - 1) buffers inside their gaps
    assert (w["gap_slots"] - w["excess_wait_slots"] <= config.BUFFER_SLOTS * (w["num_interviews"] - 1)).all()
    assert (w.loc[w["num_interviews"] == 1, ["gap_slots", "excess_wait_slots"]] == 0).all().all()


# ---------------------------------------------------------------------------
# Optimizer vs metrics vs baseline (need PuLP + CBC)
# ---------------------------------------------------------------------------
def small_instance() -> pd.DataFrame:
    return generate_sample_data(num_students=6, num_companies=3, seed=2)


def test_optimizer_finds_zero_excess_when_it_is_possible(monkeypatch):
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 3, 1)])  # nobody else competes for the companies
    result = optimize_schedule(data, time_limit=30)
    row = waiting_times(result.schedule).iloc[0]
    assert result.status == "Optimal"
    assert row["excess_wait_slots"] == 0
    assert row["gap_slots"] == 1  # exactly the 15-minute buffer, never more
    # The reported number must agree: the free buffer must not be priced as waiting.
    assert result.total_excess_wait_slots == 0


def test_optimizer_waiting_matches_the_metric():
    """The MILP and the dashboard must measure the same thing."""
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    result = optimize_schedule(small_instance(), time_limit=30)
    measured = int(waiting_times(result.schedule)["excess_wait_slots"].sum())
    assert result.total_excess_wait_slots == measured
    # (tests/test_priorities.py checks the optimizer's numbers against an exhaustive search)


def test_optimized_is_not_worse_than_baseline():
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    data = small_instance()
    base = compute_metrics(baseline_schedule(data))
    opt = compute_metrics(optimize_schedule(data, time_limit=30, warm_start=baseline_schedule(data)).schedule)
    # Priority order: the longest wait comes first, the total second.
    assert (opt["max_excess_wait_minutes"], opt["total_excess_wait_minutes"]) <= (
        base["max_excess_wait_minutes"], base["total_excess_wait_minutes"]
    )
