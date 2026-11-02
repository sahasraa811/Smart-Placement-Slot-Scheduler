"""
Clash tests: a schedule must never double-book a student or a company panel,
must keep the required buffer between a student's interviews, and must cover
every shortlisted (student, company) pair exactly once.

Run from the project root:

    python -m pytest tests/test_clashes.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from scheduler import config
from scheduler.baseline import baseline_schedule
from scheduler.data_generator import generate_sample_data
from scheduler.metrics import build_schedule, validate_schedule


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
    """Change the required buffer for one test (BUFFER_SLOTS is derived at import time)."""
    monkeypatch.setattr(config, "BUFFER_MINUTES", minutes)
    monkeypatch.setattr(config, "BUFFER_SLOTS", minutes // config.SLOT_MINUTES)


def set_days(monkeypatch, days: int) -> None:
    """Change the number of interview days for one test (TOTAL_SLOTS is derived at import time)."""
    monkeypatch.setattr(config, "NUM_DAYS", days)
    monkeypatch.setattr(config, "TOTAL_SLOTS", config.SLOTS_PER_DAY * days)


def has_error(errors: list[str], text: str) -> bool:
    return any(text in e for e in errors)


# ---------------------------------------------------------------------------
# The validator itself: does it accept good schedules and catch bad ones?
# ---------------------------------------------------------------------------
def test_validator_accepts_a_clean_schedule(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    # A: slots 0-1, one free slot (the 15-min buffer), B: slots 3-4
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 3})
    assert validate_schedule(schedule, data) == []


def test_validator_catches_student_double_booking(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 1})  # overlap on slot 1
    assert has_error(validate_schedule(schedule, data), "Student clash")


def test_validator_catches_missing_buffer(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 2})  # back-to-back, no buffer
    errors = validate_schedule(schedule, data)
    assert has_error(errors, "buffer")
    assert not has_error(errors, "Student clash")  # not an overlap, only a missing buffer


def test_back_to_back_is_allowed_when_buffer_is_zero(monkeypatch):
    set_buffer(monkeypatch, 0)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 2})
    assert validate_schedule(schedule, data) == []


def test_buffer_does_not_apply_across_different_days(monkeypatch):
    set_buffer(monkeypatch, 15)
    set_days(monkeypatch, 2)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    last_slot_of_day_1 = config.SLOTS_PER_DAY - 2           # A ends exactly at the end of day 1
    first_slot_of_day_2 = config.SLOTS_PER_DAY               # B starts first thing on day 2
    schedule = build_schedule(data, {("S1", "A"): last_slot_of_day_1, ("S1", "B"): first_slot_of_day_2})
    assert validate_schedule(schedule, data) == []


def test_validator_catches_company_clash_with_one_panel(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S2", "A", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S2", "A"): 0})  # same time, one panel
    assert has_error(validate_schedule(schedule, data), "Company clash")


def test_two_panels_allow_two_parallel_interviews(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 2), ("S2", "A", 2, 2)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S2", "A"): 0})
    assert validate_schedule(schedule, data) == []
    assert sorted(schedule["panel"]) == [1, 2]  # each interview got its own room


def test_two_panels_do_not_allow_three_parallel_interviews(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 2), ("S2", "A", 2, 2), ("S3", "A", 2, 2)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S2", "A"): 0, ("S3", "A"): 0})
    assert has_error(validate_schedule(schedule, data), "Company clash")


def test_validator_catches_missing_and_unexpected_interviews():
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1)])
    only_a = build_schedule(data.iloc[[0]], {("S1", "A"): 0})
    assert has_error(validate_schedule(only_a, data), "Missing interviews")

    extra = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1), ("S2", "A", 2, 1)])
    schedule = build_schedule(extra, {("S1", "A"): 0, ("S1", "B"): 5, ("S2", "A"): 10})
    assert has_error(validate_schedule(schedule, data), "never shortlisted")


def test_validator_catches_interview_in_lunch_break(monkeypatch):
    monkeypatch.setattr(config, "LUNCH_BREAK", ("12:00", "13:00"))  # slots 12-15
    data = make_data([("S1", "A", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 11})              # 11:45-12:15
    assert has_error(validate_schedule(schedule, data), "lunch")


def test_validator_catches_interview_past_end_of_day(monkeypatch):
    set_days(monkeypatch, 2)  # so the overrun lands inside the horizon, on day 2
    data = make_data([("S1", "A", 4, 1)])
    schedule = build_schedule(data, {("S1", "A"): config.SLOTS_PER_DAY - 2})
    assert has_error(validate_schedule(schedule, data), "end of the day")


def test_validator_catches_interview_outside_horizon():
    data = make_data([("S1", "A", 4, 1)])
    schedule = build_schedule(data, {("S1", "A"): config.TOTAL_SLOTS - 2})
    assert has_error(validate_schedule(schedule, data), "horizon")


# ---------------------------------------------------------------------------
# The real schedulers must produce clash-free timetables
# ---------------------------------------------------------------------------
def test_baseline_is_clash_free_on_default_sample_data():
    data = generate_sample_data()
    schedule = baseline_schedule(data)
    assert validate_schedule(schedule, data) == []
    assert len(schedule) == len(data)  # every shortlist scheduled exactly once


def test_baseline_is_clash_free_for_several_seeds():
    for seed in range(1, 6):
        data = generate_sample_data(seed=seed)
        schedule = baseline_schedule(data)
        assert validate_schedule(schedule, data) == [], f"seed {seed}"


def test_optimized_schedule_is_clash_free():
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    data = generate_sample_data(num_students=6, num_companies=3, seed=2)
    result = optimize_schedule(data, time_limit=30)
    assert validate_schedule(result.schedule, data) == []
    assert len(result.schedule) == len(data)


def test_optimized_schedule_respects_two_panels():
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    # Four students all shortlisted by one company with two panels.
    data = make_data([(f"S{i}", "A", 3, 2) for i in range(1, 5)])
    result = optimize_schedule(data, time_limit=30)
    assert validate_schedule(result.schedule, data) == []
    assert result.schedule["panel"].isin([1, 2]).all()
