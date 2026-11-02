"""
Tests for the starting-timetable heuristic and the "never worse than the start" safety net.

The heuristic tests need no solver. The optimizer tests need PuLP + CBC and are
skipped when it is not installed.

    python -m pytest tests/test_start_and_fallback.py
"""

from __future__ import annotations

import pytest

from scheduler.baseline import baseline_schedule
from scheduler.data_generator import generate_sample_data
from scheduler.heuristic import improve_schedule, priority_key
from scheduler.metrics import validate_schedule


@pytest.mark.parametrize("students,companies,seed", [(8, 3, 1), (20, 5, 4), (40, 8, 4), (30, 6, 7)])
def test_heuristic_returns_valid_schedule_never_worse_than_baseline(students, companies, seed):
    data = generate_sample_data(students, companies, seed)
    base = baseline_schedule(data)
    improved = improve_schedule(data, base, time_budget=5)
    assert validate_schedule(improved, data) == []        # clashes and buffer still respected
    assert priority_key(improved) <= priority_key(base)   # lexicographic: longest, total, day length


def test_heuristic_improves_the_baseline_on_the_default_scenario():
    data = generate_sample_data(20, 5, 4)
    base = baseline_schedule(data)
    assert priority_key(improve_schedule(data, base, time_budget=5)) < priority_key(base)


def test_heuristic_with_no_time_returns_the_same_schedule():
    data = generate_sample_data(20, 5, 4)
    base = baseline_schedule(data)
    assert improve_schedule(data, base, time_budget=0).equals(base)


def test_safety_net_returns_start_only_when_it_is_strictly_better():
    pytest.importorskip("pulp")
    from scheduler.optimizer import FALLBACK, StageResult, _keep_start_if_better

    data = generate_sample_data(20, 5, 4)
    base = baseline_schedule(data)
    better = improve_schedule(data, base, time_budget=5)
    assert priority_key(better) < priority_key(base)

    stages = [StageResult("a", 99, "Optimal", 1.0, 10), StageResult("b", 99, "Optimal", 1.0, 10),
              StageResult("c", 99, "Optimal", 1.0, 10)]

    # a worse solver result is replaced by the better starting timetable, and nothing claims "Optimal"
    schedule, new_stages, used = _keep_start_if_better(base, stages, better)
    assert used and schedule is better
    assert [s.value for s in new_stages] == list(priority_key(better))
    assert all(s.status == FALLBACK for s in new_stages)

    # a solver result that is better or equal is kept as it is
    for solver_result, start in ((better, base), (better, better), (base, None)):
        schedule, new_stages, used = _keep_start_if_better(solver_result, stages, start)
        assert not used and schedule is solver_result and new_stages is stages


def test_optimizer_reports_model_size_and_is_never_worse_than_baseline():
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    data = generate_sample_data(8, 3, 2)
    base = baseline_schedule(data)
    result = optimize_schedule(data, time_limit=30, warm_start=base)
    assert result.num_variables > result.num_binaries > 0
    assert result.num_constraints > 0
    assert priority_key(result.schedule) <= priority_key(base)
    assert validate_schedule(result.schedule, data) == []
    if result.status == "Optimal":
        assert not result.fallback_used


def test_a_stage_result_that_worsens_an_earlier_priority_is_rejected(monkeypatch):
    """Force the stage-2 and stage-3 candidates to look worse; they must be rejected, not accepted or labelled optimal."""
    pytest.importorskip("pulp")
    import scheduler.optimizer as optimizer

    real_measure = optimizer._measure
    calls = {"n": 0}

    def inflated_after_stage_1(schedule):
        calls["n"] += 1
        key = real_measure(schedule)
        return key if calls["n"] == 1 else (key[0], key[1] + 5, key[2])  # total looks 5 slots worse

    monkeypatch.setattr(optimizer, "_measure", inflated_after_stage_1)
    data = generate_sample_data(8, 3, 2)
    result = optimizer.optimize_schedule(data, time_limit=30, warm_start=baseline_schedule(data))

    if result.stages[1].status.startswith("Skipped"):
        pytest.skip("nobody waits in this instance, so stage 2 has nothing to reject")
    assert result.stages[1].status == optimizer.REJECTED
    assert result.status != "Optimal"
    assert validate_schedule(result.schedule, data) == []


def test_trajectory_records_every_step_and_never_ends_worse_than_the_start():
    pytest.importorskip("pulp")
    from scheduler.optimizer import optimize_schedule

    data = generate_sample_data(8, 3, 2)
    base = baseline_schedule(data)
    result = optimize_schedule(data, time_limit=30, warm_start=base)

    keys = [snap.key for snap in result.trajectory]
    assert keys == ["baseline", "improved_start", "stage1", "stage2", "stage3", "final"]

    def key(snap):
        return (snap.longest, snap.total, snap.day)

    by_key = {snap.key: snap for snap in result.trajectory}
    assert key(by_key["baseline"]) == priority_key(base)
    assert key(by_key["final"]) == priority_key(result.schedule)  # the last row describes the returned schedule
    assert key(by_key["final"]) <= key(by_key["improved_start"]) <= key(by_key["baseline"])
    # from stage 1 on, a later stage never worsens the longest or the total wait
    chain = [by_key[k] for k in ("stage1", "stage2", "stage3")]
    for earlier, later in zip(chain, chain[1:]):
        assert (later.longest, later.total) <= (earlier.longest, earlier.total)
