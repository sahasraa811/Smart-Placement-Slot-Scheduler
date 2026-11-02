"""
Tests for the dashboard's helper functions (tables, wording, charts). They do
not start Streamlit; app.py just has to be importable.

Run from the project root:

    python -m pytest tests/test_dashboard.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")

import app  # noqa: E402
from scheduler import config  # noqa: E402
from scheduler.baseline import baseline_schedule  # noqa: E402
from scheduler.data_generator import generate_sample_data  # noqa: E402
from scheduler.metrics import build_schedule  # noqa: E402


def make_data(shortlists: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Hand-made problem data. Each item is (student_id, company_id, duration_slots, panels)."""
    return pd.DataFrame(
        [
            {"student_id": s, "student_name": f"Student {s}", "company_id": c,
             "company_name": f"Company {c}", "duration_slots": d, "panels": p}
            for s, c, d, p in shortlists
        ]
    )


def set_buffer(monkeypatch, minutes: int) -> None:
    monkeypatch.setattr(config, "BUFFER_MINUTES", minutes)
    monkeypatch.setattr(config, "BUFFER_SLOTS", minutes // config.SLOT_MINUTES)


# ---------------------------------------------------------------------------
# Wording and tables
# ---------------------------------------------------------------------------
def test_headline_says_whether_waiting_went_down():
    assert "645 to 75" in app.headline(645, 75) and "88%" in app.headline(645, 75)
    assert "no excess waiting" in app.headline(0, 0)
    assert "same" in app.headline(100, 100)
    assert "more excess waiting" in app.headline(50, 80)


def test_numbers_keep_their_decimals_but_whole_numbers_stay_whole():
    assert app._fmt(75) == "75" and app._fmt(75.0) == "75"
    assert app._fmt(3.75) == "3.75" and app._fmt(2.5) == "2.5"
    assert app._fmt("16:00") == "16:00"


def test_comparison_table_is_all_text_and_shows_the_change():
    base = baseline_schedule(generate_sample_data())
    same = app.comparison_table(base, base)
    assert list(same.columns) == ["Metric", "Baseline", "Optimized", "Change"]
    assert all(isinstance(v, str) for v in same.to_numpy().ravel())  # mixed types would break st.dataframe
    assert set(same["Change"]) <= {"0%", "-"}                  # identical schedules: nothing changed


def test_student_itinerary_shows_wait_and_excess_before_each_interview(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 2, 1), ("S1", "C", 2, 1)])
    # A 0-1 | idle 2-5 | B 6-7 | idle 8 | C 9-10
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 6, ("S1", "C"): 9})
    rows = app.student_itinerary(schedule, "S1")
    assert list(rows["Wait since previous (min)"]) == ["-", "60", "15"]
    assert list(rows["Of which excess (min)"]) == ["-", "45", "0"]
    assert list(rows["Company"]) == ["Company A", "Company B", "Company C"]


# ---------------------------------------------------------------------------
# Charts build without errors
# ---------------------------------------------------------------------------
def test_charts_are_built_for_both_views():
    data = generate_sample_data()
    schedule = baseline_schedule(data)
    colors = app.company_colors(data, schedule)
    assert len(set(colors.values())) == data["company_id"].nunique()  # one colour per company

    for draw in (app.student_timetable_figure, app.company_timetable_figure):
        fig = draw(schedule, colors, 1, "Baseline")
        assert fig.axes and fig.axes[0].get_title(loc="left") == "Baseline"
    assert app.excess_by_student_figure(schedule, schedule) is not None


def test_no_comparison_chart_when_nobody_has_two_interviews():
    data = make_data([("S1", "A", 2, 1), ("S2", "B", 2, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S2", "B"): 0})
    assert app.excess_by_student_figure(schedule, schedule) is None


# ---------------------------------------------------------------------------
# Overview numbers and the solver-status wording
# ---------------------------------------------------------------------------
def test_overview_rows_mark_improved_unchanged_and_worse():
    base = {"total_excess_wait_minutes": 645, "max_excess_wait_minutes": 210,
            "makespan_minutes": 420, "students_with_excess_wait": 9}
    opt = {"total_excess_wait_minutes": 0, "max_excess_wait_minutes": 210,
           "makespan_minutes": 435, "students_with_excess_wait": 0}
    rows = {r["label"]: r for r in app.overview_rows(base, opt)}
    assert len(rows) == 4
    assert rows["Total excess waiting"]["state"] == "improved"
    assert rows["Total excess waiting"]["baseline"] == "645 min" and rows["Total excess waiting"]["optimized"] == "0 min"
    assert rows["Longest excess waiting"]["state"] == "unchanged" and rows["Longest excess waiting"]["change"] == "No change"
    assert rows["Placement duration"]["state"] == "worse"        # longer day is not an improvement
    assert rows["Students with excess wait"]["optimized"] == "0"  # a count has no unit

    html = app.comparison_html(list(rows.values()))
    assert "class='improved'" in html and "class='unchanged'" in html and "class='worse'" in html


def test_metric_card_colours_follow_the_change():
    assert "card-delta improved" in app.metric_card_html("Total", "0 min", app.delta_text(-645), "improved")
    assert "No change vs baseline" in app.metric_card_html("Total", "0 min", app.delta_text(0), "unchanged")
    assert "card-delta worse" in app.metric_card_html("Total", "9 min", app.delta_text(9), "worse")
    assert "&lt;b&gt;" in app.metric_card_html("<b>x</b>", "1")   # text is escaped, never raw HTML


def test_only_a_proven_optimum_is_called_optimal():
    assert app.solver_status_info("Optimal")["label"] == "Optimal"

    feasible = app.solver_status_info("Feasible (not proven optimal)")
    assert feasible["label"] == "Feasible" and feasible["tone"] == "warn"
    assert "time limit" in feasible["body"] and "better timetable may exist" in feasible["note"]
    assert "Optimal" not in feasible["label"] and "Optimal" not in feasible["title"]

    assert app.solver_status_info("Infeasible")["tone"] == "error"
    assert app.solver_status_info("Time limit reached")["label"] == "Time limit reached"
    assert app.solver_status_info(None)["label"] == "Not recorded"
    assert "starting timetable" in app.solver_status_info("Optimal", fallback_used=True)["note"]


def test_solver_error_text_is_mapped_to_a_status():
    assert app.solver_error_status("Solver found no complete schedule (status: Infeasible). Try ...") == "Infeasible"
    assert app.solver_error_status("Solver found no complete schedule (status: Not Solved). Try ...") == "Time limit reached"
    assert app.solver_error_status("Something else went wrong") is None


# ---------------------------------------------------------------------------
# Student Schedule and Input Data pages
# ---------------------------------------------------------------------------
def test_itinerary_display_has_one_time_column(monkeypatch):
    set_buffer(monkeypatch, 15)
    data = make_data([("S1", "A", 2, 1), ("S1", "B", 3, 1)])
    schedule = build_schedule(data, {("S1", "A"): 0, ("S1", "B"): 3})   # A 09:00-09:30, gap, B 09:45-10:30
    table = app.itinerary_display(schedule, "S1")
    assert list(table.columns) == ["Time", "Company", "Wait before (min)", "Excess wait (min)"]  # one day: no Day column
    assert list(table["Time"]) == ["09:00\u201309:30", "09:45\u201310:30"]
    assert list(table["Company"]) == ["Company A", "Company B"]


def test_input_data_tables_show_minutes_and_the_three_views():
    data = generate_sample_data()
    companies, students, shortlists = (app.company_details_table(data), app.student_summary_table(data),
                                       app.shortlist_table(data))
    assert list(companies.columns) == ["ID", "Company", "Interview duration (min)", "Panels", "Shortlisted students"]
    assert list(students.columns) == ["ID", "Student", "Interviews", "Total interview duration (min)"]
    assert list(shortlists.columns) == ["Student ID", "Company ID", "Company", "Interview duration (min)"]
    assert len(shortlists) == len(data)                                   # one row per interview to schedule
    assert set(companies["Interview duration (min)"]) <= {d * config.SLOT_MINUTES for d in config.DURATION_CHOICES_SLOTS}
    assert students["Total interview duration (min)"].sum() == shortlists["Interview duration (min)"].sum()


# ---------------------------------------------------------------------------
# The saved record of the last solver run
# ---------------------------------------------------------------------------
def fake_result(schedule):
    """Just the fields app.summarize_run reads from optimizer.OptimizationResult."""
    snap = SimpleNamespace(label="After stage 1", longest=0, total=0, day=28, note="Optimal")
    stage = SimpleNamespace(name="Longest excess wait", status="Optimal", seconds=1.5, time_limit=20, value=0)
    return SimpleNamespace(schedule=schedule, status="Feasible (not proven optimal)", solve_seconds=3.2,
                           fallback_used=False, start_note="", start_improved=False, num_variables=120,
                           num_binaries=100, num_constraints=300, trajectory=[snap], stages=[stage])


def test_run_summary_survives_a_save_and_load_and_only_matches_its_own_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    data = generate_sample_data()
    schedule = baseline_schedule(data)

    summary = app.summarize_run(fake_result(schedule))
    assert summary["status"] == "Feasible (not proven optimal)" and summary["model"] == [120, 100, 300]
    assert summary["trajectory"][0]["day"] == 28 * config.SLOT_MINUTES     # slots become minutes

    app.save_run_summary(summary)
    assert app.load_run_summary() == summary
    assert app.load_run_summary()["fingerprint"] == app.schedule_fingerprint(schedule)

    shifted = schedule.copy()
    shifted.loc[shifted.index[0], "start_slot"] += 1                       # a different schedule
    assert app.schedule_fingerprint(shifted) != summary["fingerprint"]


def test_missing_or_broken_run_summary_is_just_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert app.load_run_summary() is None
    (tmp_path / "run_summary.json").write_text("{not json", encoding="utf-8")
    assert app.load_run_summary() is None
