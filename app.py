"""
Streamlit dashboard for the Smart Placement Scheduler.

Shows the baseline (first-come-first-served) timetable next to the optimized
one, so you can see whether waiting really went down. Four pages:

    Overview            headline numbers, baseline vs optimized, waiting per student,
                        solver status, and (collapsed) the Optimization Details
    Interview Schedule  Gantt charts by student or company, one schedule at a time
    Student Schedule    one student's itinerary and waiting-time comparison
    Input Data          company details, student summary, interview shortlists

Waiting follows scheduler/metrics.py: every student gets a required buffer
(config.BUFFER_MINUTES) between interviews, and only waiting BEYOND that buffer
counts as "excess waiting".

Run from the project root:

    streamlit run app.py

The app reads data/sample_data.csv and results/*.csv. Create them with

    python -m scheduler.data_generator
    python -m scheduler.baseline
    python -m scheduler.optimizer

or use "Generate & Optimize" in the sidebar. A run from the sidebar also saves
results/run_summary.json, so the solver status and Optimization Details are
still there after a page reload.
"""

from __future__ import annotations

import json
from html import escape as esc
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scheduler import config
from scheduler.data_generator import get_companies, get_students, load_sample_data
from scheduler.metrics import (
    compare_schedules,
    compute_metrics,
    validate_schedule,
    waiting_segments,
    waiting_times,
)

# ---------------------------------------------------------------------------
# Look: navy and white
# ---------------------------------------------------------------------------
NAVY = "#14284B"
BASELINE_COLOR = "#A7B0BE"    # muted: the thing we are improving on
OPTIMIZED_COLOR = "#1F3A68"   # the accent of the whole dashboard
BUFFER_COLOR = "#CBD2DA"      # required buffer, not counted as waiting
EXCESS_COLOR = "#E15759"      # waiting beyond the buffer
INK = "#1F2933"

# One colour per company, reused in every chart. No red, which means "excess waiting".
COMPANY_COLORS = [
    "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#9C755F", "#EDC948",
    "#76B7B2", "#FF9DA7", "#6F4E7C", "#A0CBE8", "#FFBE7D", "#8CD17D",
]

# One font everywhere: the page uses the theme font (.streamlit/config.toml), the charts this stack.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

CSS = """
.block-container { padding-top: 4.5rem; max-width: 1200px; }  /* clears Streamlit's fixed header bar */
.app-eyebrow { font-size: .72rem; letter-spacing: .16em; font-weight: 600; color: #4A6FA5; margin: 0; line-height: 1.6; }
.app-title { font-size: 2rem; font-weight: 700; color: #14284B; line-height: 1.15; margin: .2rem 0 .25rem; }
.app-sub { color: #5B6573; font-size: 1rem; margin: 0 0 1.1rem; }
.section-title { font-size: 1.15rem; font-weight: 600; color: #14284B; margin: 1.5rem 0 .4rem; }
.lead { font-size: 1.05rem; color: #1F2933; margin: .2rem 0 1rem; }
.student-head { color: #5B6573; margin: .2rem 0 .8rem; }
.student-head strong { color: #14284B; }
.side-title { font-size: 1.05rem; font-weight: 600; color: #14284B; margin-bottom: .1rem; }

[data-testid="stSidebar"] { background: #F4F6FA; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] { margin-bottom: .5rem; }
.card { background: #F4F6FA; border-radius: 10px; padding: .9rem 1.1rem; }
.card-label { font-size: .8rem; font-weight: 600; color: #5B6573; }
.card-value { font-size: 1.9rem; font-weight: 700; color: #14284B; line-height: 1.25; margin: .1rem 0; }
.card-delta { font-size: .85rem; font-weight: 600; min-height: 1.2rem; }
.card-delta.improved { color: #1E7F4F; }
.card-delta.unchanged { color: #6B7280; }
.card-delta.worse { color: #B42318; }
[data-testid="stExpander"] { border-color: #E3E8F0; border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }

table.cmp { width: 100%; border-collapse: collapse; font-size: .95rem; margin: .3rem 0 .6rem; }
table.cmp th { text-align: left; color: #5B6573; font-weight: 600; font-size: .75rem; letter-spacing: .06em;
               text-transform: uppercase; padding: .5rem .75rem; border-bottom: 2px solid #14284B; }
table.cmp td { padding: .65rem .75rem; border-bottom: 1px solid #E8ECF2; }
table.cmp td.improved { color: #1E7F4F; font-weight: 600; }
table.cmp td.unchanged { color: #6B7280; }
table.cmp td.worse { color: #B42318; font-weight: 600; }

.status { display: flex; gap: .8rem; align-items: flex-start; padding: .75rem 1rem; border-radius: 10px;
          margin: .4rem 0; font-size: .93rem; line-height: 1.5; }
.status-text strong { margin-right: .35rem; }
.pill { flex: none; font-size: .7rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
        padding: .2rem .65rem; border-radius: 999px; margin-top: .1rem; color: #fff; }
.status-ok { background: #EAF5EE; }       .pill-ok { background: #1E7F4F; }
.status-warn { background: #FFF6E0; }     .pill-warn { background: #B7791F; }
.status-error { background: #FDECEA; }    .pill-error { background: #B42318; }
.status-neutral { background: #EEF1F5; }  .pill-neutral { background: #6B7280; }
.status-plain { background: none; padding-left: 0; padding-right: 0; }
"""

METRIC_LABELS = {
    "total_excess_wait_minutes": "Total excess waiting (min)",
    "avg_excess_wait_minutes": "Average excess waiting per student (min)",
    "max_excess_wait_minutes": "Longest excess wait (min)",
    "students_with_excess_wait": "Students waiting beyond the buffer",
    "total_gap_minutes": "All time between interviews, buffers included (min)",
    "makespan_minutes": "Placement-day length, first start to last end (min)",
    "last_finish_time": "Last interview ends",
}

# The four numbers on the Overview: (label, key in compute_metrics, unit). Lower is better for all of them.
OVERVIEW_METRICS = [
    ("Total excess waiting", "total_excess_wait_minutes", " min"),
    ("Longest excess waiting", "max_excess_wait_minutes", " min"),
    ("Placement duration", "makespan_minutes", " min"),
    ("Students with excess wait", "students_with_excess_wait", ""),
]


# ---------------------------------------------------------------------------
# Loading and running
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)  # `modified` is only here so a changed file is re-read


def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a results csv, or None if it does not exist yet."""
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def load_problem() -> pd.DataFrame | None:
    """Read and validate data/sample_data.csv, or None if it does not exist yet."""
    if not config.SAMPLE_DATA_PATH.exists():
        return None
    return load_sample_data()


def run_pipeline(num_students: int, num_companies: int, seed: int, time_limit: int):
    """Generate data, run the baseline and the optimizer, save all three csv files."""
    from scheduler.baseline import baseline_schedule
    from scheduler.data_generator import generate_sample_data, save_sample_data
    from scheduler.optimizer import optimize_schedule  # imports PuLP; loaded only when needed

    data = generate_sample_data(num_students, num_companies, seed)
    baseline = baseline_schedule(data)
    result = optimize_schedule(data, time_limit=time_limit, warm_start=baseline)

    save_sample_data(data)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(config.BASELINE_SCHEDULE_PATH, index=False)
    result.schedule.to_csv(config.OPTIMIZED_SCHEDULE_PATH, index=False)
    return result


# ---------------------------------------------------------------------------
# The record of the last solver run (so status and details survive a reload)
# ---------------------------------------------------------------------------
def schedule_fingerprint(schedule: pd.DataFrame) -> list[int]:
    """A few numbers that identify a schedule, to tell whether a saved run summary still belongs to it."""
    m = compute_metrics(schedule)
    return [
        int(m["total_excess_wait_minutes"]),
        int(m["max_excess_wait_minutes"]),
        int(m["makespan_minutes"]),
        int(len(schedule)),
        int(schedule["start_slot"].sum()),
    ]


def summarize_run(result) -> dict:
    """Everything the page shows about one solver run, as plain numbers and text (minutes, not slots)."""
    minutes = config.SLOT_MINUTES
    return {
        "status": result.status,
        "seconds": result.solve_seconds,
        "fallback_used": result.fallback_used,
        "start_note": result.start_note,
        "start_improved": result.start_improved,
        "model": [result.num_variables, result.num_binaries, result.num_constraints],
        "trajectory": [
            {
                "step": snap.label,
                "longest": snap.longest * minutes,
                "total": snap.total * minutes,
                "day": snap.day * minutes,
                "note": snap.note,
            }
            for snap in result.trajectory
        ],
        "stages": [
            {
                "name": stage.name,
                "status": stage.status,
                "seconds": stage.seconds,
                "limit": stage.time_limit,
                "value": stage.value * minutes,
            }
            for stage in result.stages
        ],
        "fingerprint": schedule_fingerprint(result.schedule),
    }


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def summary_path() -> Path:
    return config.RESULTS_DIR / "run_summary.json"


def save_run_summary(summary: dict) -> None:
    """Best effort: a read-only folder must not break the run."""
    try:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path().write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    except OSError:
        pass


def load_run_summary() -> dict | None:
    try:
        return json.loads(summary_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def current_run(optimized: pd.DataFrame) -> dict | None:
    """The run summary that belongs to these results, or None (never a summary from another run)."""
    for candidate in (st.session_state.get("last_run"), load_run_summary()):
        if candidate and candidate.get("fingerprint") == schedule_fingerprint(optimized):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Tables and text (no Streamlit in here, so they are easy to test)
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    """Number or text as a short string; whole numbers without decimals."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}".rstrip("0").rstrip(".")  # 3.75 stays 3.75, 2.50 becomes 2.5
    return str(value)


def comparison_table(baseline: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side metrics with readable names. Every cell is text, so the table always displays."""
    rows = []
    for rec in compare_schedules(baseline, optimized).to_dict("records"):
        if rec["metric"] not in METRIC_LABELS:
            continue
        improvement = rec["improvement_%"]
        if improvement is None or pd.isna(improvement):
            change = "-"
        else:
            pct = -float(improvement)  # improvement is "% lower"; show the change itself
            change = "0%" if round(pct) == 0 else f"{pct:+.0f}%"
        rows.append(
            {
                "Metric": METRIC_LABELS[rec["metric"]],
                "Baseline": _fmt(rec["Baseline"]),
                "Optimized": _fmt(rec["Optimized"]),
                "Change": change,
            }
        )
    return pd.DataFrame(rows)


def change_state(diff: float) -> str:
    """Lower is better everywhere in this dashboard: a drop is an improvement."""
    return "unchanged" if diff == 0 else ("improved" if diff < 0 else "worse")


def signed(diff: float, unit: str = " min") -> str:
    """'\u2212645 min' or '+30 min' (a real minus sign)."""
    return f"{diff:+g}{unit}".replace("-", "\u2212")


def delta_text(diff: float, unit: str = " min") -> str:
    """'\u2212645 min vs baseline', or 'No change vs baseline'."""
    return "No change vs baseline" if diff == 0 else f"{signed(diff, unit)} vs baseline"


def metric_card_html(label: str, value: str, delta: str = "", state: str = "unchanged") -> str:
    """A small summary card. The delta line is green for an improvement, grey if unchanged, red if worse."""
    arrow = {"improved": "\u25BC ", "worse": "\u25B2 "}.get(state, "") if delta else ""
    return (
        f"<div class='card'><div class='card-label'>{esc(label)}</div><div class='card-value'>{esc(value)}</div>"
        f"<div class='card-delta {state}'>{arrow}{esc(delta) or '&nbsp;'}</div></div>"
    )


def overview_rows(mb: dict, mo: dict) -> list[dict]:
    """The four Overview metrics as baseline / optimized / change, plus whether that is better, equal or worse."""
    rows = []
    for label, key, unit in OVERVIEW_METRICS:
        base, opt = float(mb[key]), float(mo[key])
        diff = opt - base
        state = change_state(diff)
        rows.append(
            {
                "label": label,
                "key": key,
                "unit": unit,
                "baseline": f"{_fmt(base)}{unit}",
                "optimized": f"{_fmt(opt)}{unit}",
                "diff": diff,
                "change": "No change" if diff == 0 else signed(diff, unit),
                "state": state,
            }
        )
    return rows


def comparison_html(rows: list[dict]) -> str:
    """Baseline vs optimized: green for an improvement, grey when unchanged, red if it got worse."""
    body = "".join(
        f"<tr><td>{esc(r['label'])}</td><td>{esc(r['baseline'])}</td>"
        f"<td class='{r['state']}'>{esc(r['optimized'])}</td><td class='{r['state']}'>{esc(r['change'])}</td></tr>"
        for r in rows
    )
    return (
        "<table class='cmp'><thead><tr><th>Metric</th><th>Baseline</th><th>Optimized</th><th>Change</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def headline(baseline_total: float, optimized_total: float) -> str:
    """One plain sentence that answers 'did waiting actually go down?'."""
    if baseline_total == 0:
        return "The baseline schedule already has no excess waiting."
    change = (baseline_total - optimized_total) / baseline_total
    if optimized_total < baseline_total:
        return (
            f"The optimized schedule cuts excess waiting from {_fmt(baseline_total)} to "
            f"{_fmt(optimized_total)} minutes ({change:.0%} less)."
        )
    if optimized_total == baseline_total:
        return f"The optimized schedule has the same excess waiting as the baseline ({_fmt(baseline_total)} minutes)."
    return (
        f"The optimized schedule has more excess waiting than the baseline "
        f"({_fmt(optimized_total)} vs {_fmt(baseline_total)} minutes). The optimizer shortens the longest "
        "single wait first, so this can be the price of a better worst case; otherwise re-run it with a longer time limit."
    )


def solver_status_info(status: str | None, fallback_used: bool = False) -> dict:
    """
    Plain-language wording for a solver status: {tone, label, title, body, note}.
    Only a proven-optimal run is ever called "Optimal".
    """
    s = (status or "").strip().lower()
    if not s:
        return {
            "tone": "neutral", "label": "Not recorded", "title": "Solver status not recorded",
            "body": "These timetables were loaded from saved files, so no solver run is attached to them.",
            "note": "Use Generate & Optimize in the sidebar to run the solver and see its status.",
        }
    if s.startswith("optimal"):
        info = {
            "tone": "ok", "label": "Optimal", "title": "Optimization completed",
            "body": "A valid timetable was generated, and the solver proved that no better timetable exists "
                    "under the optimization priorities.",
            "note": "",
        }
    elif s.startswith("feasible"):
        info = {
            "tone": "warn", "label": "Feasible", "title": "Optimization completed",
            "body": "A valid timetable was generated. However, the solver reached its time limit before "
                    "proving that the solution is optimal.",
            "note": "A better timetable may exist.",
        }
    elif s.startswith("infeasible"):
        info = {
            "tone": "error", "label": "Infeasible", "title": "No valid timetable found",
            "body": "No timetable can satisfy every constraint for this scenario.",
            "note": "Try fewer students or shortlists, more companies or panels, or more interview days.",
        }
    elif "time limit" in s:
        info = {
            "tone": "error", "label": "Time limit reached", "title": "Time limit reached",
            "body": "The solver stopped after its allotted time without finding a valid timetable.",
            "note": "Try a longer time limit or a smaller scenario.",
        }
    else:
        info = {"tone": "neutral", "label": str(status), "title": f"Solver status: {status}", "body": "", "note": ""}
    if fallback_used:
        extra = ("The solver's answer was worse than its starting timetable in priority order, "
                 "so the starting timetable is shown instead.")
        info["note"] = f"{info['note']} {extra}".strip()
    return info


def solver_error_status(message: str) -> str | None:
    """Map the optimizer's 'no complete schedule' error text to a status name, or None if it is something else."""
    m = message.lower()
    if "infeasible" in m:
        return "Infeasible"
    if "not solved" in m or "time limit" in m:
        return "Time limit reached"
    return None


def status_html(tone: str, label: str, title: str, body: str = "", note: str = "", plain: bool = False) -> str:
    """A compact status line: a coloured pill, a bold title and up to two short sentences."""
    text = " ".join([f"<strong>{esc(title)}</strong>"] + [esc(t) for t in (body, note) if t])
    return (
        f"<div class='status status-{tone}{' status-plain' if plain else ''}'><span class='pill pill-{tone}'>{esc(label)}</span>"
        f"<div class='status-text'>{text}</div></div>"
    )


def student_itinerary(schedule: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """One student's interviews in time order, with the wait before each one."""
    mine = schedule[schedule["student_id"] == student_id].sort_values("start_slot")
    segs = waiting_segments(schedule)
    segs = segs[segs["student_id"] == student_id]

    rows = []
    previous = None
    for r in mine.itertuples(index=False):
        wait, excess = "-", "-"
        if previous is not None and previous.day == r.day:
            between = segs[(segs["start_slot"] >= previous.end_slot) & (segs["end_slot"] <= r.start_slot)]
            wait = _fmt(between["minutes"].sum())
            excess = _fmt(between.loc[between["kind"] == "excess", "minutes"].sum())
        rows.append(
            {
                "Day": r.day,
                "Start": r.start_time,
                "End": r.end_time,
                "Company": r.company_name,
                "Panel": r.panel,
                "Wait since previous (min)": wait,
                "Of which excess (min)": excess,
            }
        )
        previous = r
    return pd.DataFrame(rows)


def itinerary_display(schedule: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """The itinerary as the Student Schedule page shows it: one 'Time' column, the company, the waits."""
    it = student_itinerary(schedule, student_id)
    out = pd.DataFrame(
        {
            "Time": it["Start"] + "\u2013" + it["End"],
            "Company": it["Company"],
            "Wait before (min)": it["Wait since previous (min)"],
            "Excess wait (min)": it["Of which excess (min)"],
        }
    )
    if it["Day"].nunique() > 1:
        out.insert(0, "Day", it["Day"])
    return out


def timetable_table(schedule: pd.DataFrame) -> pd.DataFrame:
    """The schedule as a readable table, in time order."""
    out = schedule.sort_values(["start_slot", "company_id", "student_id"])
    out = out[["day", "start_time", "end_time", "student_name", "company_name", "panel"]]
    return out.rename(
        columns={
            "day": "Day",
            "start_time": "Start",
            "end_time": "End",
            "student_name": "Student",
            "company_name": "Company",
            "panel": "Panel",
        }
    ).reset_index(drop=True)


def company_details_table(data: pd.DataFrame) -> pd.DataFrame:
    c = get_companies(data)
    return pd.DataFrame(
        {
            "ID": c["company_id"],
            "Company": c["company_name"],
            "Interview duration (min)": c["duration_slots"] * config.SLOT_MINUTES,
            "Panels": c["panels"],
            "Shortlisted students": c["num_candidates"],
        }
    )


def student_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    s = get_students(data)
    return pd.DataFrame(
        {
            "ID": s["student_id"],
            "Student": s["student_name"],
            "Interviews": s["num_interviews"],
            "Total interview duration (min)": s["total_interview_slots"] * config.SLOT_MINUTES,
        }
    )


def shortlist_table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Student ID": data["student_id"],
            "Company ID": data["company_id"],
            "Company": data["company_name"],
            "Interview duration (min)": data["duration_slots"] * config.SLOT_MINUTES,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def company_colors(*tables: pd.DataFrame) -> dict[str, str]:
    """A stable colour per company, from every table that mentions companies."""
    ids = sorted(set().union(*(set(t["company_id"]) for t in tables)))
    return {c: COMPANY_COLORS[i % len(COMPANY_COLORS)] for i, c in enumerate(ids)}


def _text_on(color: str) -> str:
    r, g, b = to_rgb(color)
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def _clock(slot_in_day: int) -> str:
    return config.slot_to_clock(0, slot_in_day)


def _style_time_axis(ax) -> None:
    per_hour = max(1, round(60 / config.SLOT_MINUTES))
    ticks = list(range(0, config.SLOTS_PER_DAY + 1, per_hour))
    ax.set_xlim(0, config.SLOTS_PER_DAY)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_clock(k) for k in ticks], fontsize=8.5, color=INK)
    ax.set_xlabel("Time of day", fontsize=8.5, color=INK)
    ax.xaxis.grid(True, color="#E6E9EE", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BUFFER_COLOR)


def student_timetable_figure(schedule: pd.DataFrame, colors: dict[str, str], day: int, title: str) -> Figure:
    """
    One row per student. Coloured blocks are interviews; the thin bars between
    them are the gaps: grey = the required buffer, red = excess waiting.
    """
    offset = (day - 1) * config.SLOTS_PER_DAY
    sched = schedule[schedule["day"] == day]
    students = sorted(sched["student_id"].unique())
    row = {s: i for i, s in enumerate(students)}
    segs = waiting_segments(schedule)
    segs = segs[segs["day"] == day]

    fig = Figure(figsize=(11, 1.8 + 0.34 * len(students)), layout="constrained")
    ax = fig.subplots()

    for kind, color in (("buffer", BUFFER_COLOR), ("excess", EXCESS_COLOR)):
        part = segs[segs["kind"] == kind]
        ax.barh([row[s] for s in part["student_id"]], part["slots"], left=part["start_slot"] - offset,
                height=0.3, color=color, linewidth=0)

    ax.barh([row[s] for s in sched["student_id"]], sched["duration_slots"], left=sched["start_slot"] - offset,
            height=0.68, color=[colors[c] for c in sched["company_id"]], edgecolor="white", linewidth=0.8)
    for r in sched.itertuples(index=False):
        ax.text(r.start_slot - offset + r.duration_slots / 2, row[r.student_id], r.company_id,
                ha="center", va="center", fontsize=8, color=_text_on(colors[r.company_id]))

    ax.set_yticks(range(len(students)))
    ax.set_yticklabels(students, fontsize=8.5, color=INK)
    ax.set_ylim(len(students) - 0.5, -0.5)
    _style_time_axis(ax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=NAVY)

    names = sched.drop_duplicates("company_id").set_index("company_id")["company_name"]
    handles = [Patch(color=colors[c], label=f"{c} {names[c]}") for c in sorted(names.index)]
    handles += [
        Patch(color=BUFFER_COLOR, label=f"{config.BUFFER_MINUTES}-minute buffer"),
        Patch(color=EXCESS_COLOR, label="Excess waiting"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), fontsize=8.5, frameon=False)
    return fig


def company_timetable_figure(schedule: pd.DataFrame, colors: dict[str, str], day: int, title: str) -> Figure:
    """One row per company panel, each block labelled with the student being interviewed."""
    offset = (day - 1) * config.SLOTS_PER_DAY
    sched = schedule[schedule["day"] == day]
    panels = schedule.groupby("company_id")["panel"].max()
    names = schedule.drop_duplicates("company_id").set_index("company_id")["company_name"]

    lanes = sorted({(r.company_id, r.panel) for r in sched.itertuples(index=False)})
    row = {lane: i for i, lane in enumerate(lanes)}
    labels = [names[c] if panels[c] == 1 else f"{names[c]} (panel {p})" for c, p in lanes]

    fig = Figure(figsize=(11, 1.5 + 0.6 * len(lanes)), layout="constrained")
    ax = fig.subplots()
    ax.barh([row[(r.company_id, r.panel)] for r in sched.itertuples(index=False)], sched["duration_slots"],
            left=sched["start_slot"] - offset, height=0.7,
            color=[colors[c] for c in sched["company_id"]], edgecolor="white", linewidth=0.8)
    for r in sched.itertuples(index=False):
        ax.text(r.start_slot - offset + r.duration_slots / 2, row[(r.company_id, r.panel)], r.student_id,
                ha="center", va="center", fontsize=8, color=_text_on(colors[r.company_id]))

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(len(lanes) - 0.5, -0.5)
    _style_time_axis(ax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=NAVY)
    return fig


def excess_by_student_figure(baseline: pd.DataFrame, optimized: pd.DataFrame) -> Figure | None:
    """Excess waiting per student, baseline next to optimized. None if nobody can wait."""
    b = waiting_times(baseline)[["student_id", "num_interviews", "excess_wait_minutes"]]
    o = waiting_times(optimized)[["student_id", "excess_wait_minutes"]]
    both = b.merge(o, on="student_id", suffixes=("_base", "_opt"))
    both = both[both["num_interviews"] > 1]  # a single interview can never involve waiting
    if both.empty:
        return None
    both = both.sort_values(["excess_wait_minutes_base", "student_id"], ascending=[False, True])

    x = np.arange(len(both))
    fig = Figure(figsize=(11, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.bar(x - 0.2, both["excess_wait_minutes_base"], 0.4, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + 0.2, both["excess_wait_minutes_opt"], 0.4, color=OPTIMIZED_COLOR, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(both["student_id"], rotation=90, fontsize=8, color=INK)
    ax.set_ylabel("Excess waiting (min)", fontsize=9, color=INK)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#E6E9EE", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BUFFER_COLOR)
    ax.legend(frameon=False, fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Page pieces
# ---------------------------------------------------------------------------
def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{esc(text)}</div>", unsafe_allow_html=True)


def show_checks(baseline: pd.DataFrame, optimized: pd.DataFrame, data: pd.DataFrame) -> None:
    """One quiet line when both timetables are valid; a visible error for each one that is not."""
    failed = []
    for label, schedule in (("Baseline", baseline), ("Optimized", optimized)):
        errors = validate_schedule(schedule, data)
        if errors:
            failed.append((label, errors))
    if not failed:
        st.markdown(
            status_html(
                "ok", "Checks passed", "Both timetables are valid.",
                f"No clashes, every shortlisted interview scheduled once, {config.BUFFER_MINUTES}-minute buffer respected.",
                plain=True,
            ),
            unsafe_allow_html=True,
        )
        return
    for label, errors in failed:
        shown = "\n".join(f"- {e}" for e in errors[:5])
        more = f"\n- ...and {len(errors) - 5} more" if len(errors) > 5 else ""
        st.error(
            f"{label}: {len(errors)} problem(s) found. The csv may belong to different data; "
            f"re-run the baseline and optimizer, or use Generate & Optimize.\n{shown}{more}"
        )


def show_run_error(err: Exception, has_results: bool) -> None:
    """A failed run, in the sidebar, in plain words, with a note about what is still on screen."""
    status = solver_error_status(str(err)) if isinstance(err, RuntimeError) else None
    if status:
        info = solver_status_info(status)
        st.error(f"**{info['title']}**\n\n{info['body']} {info['note']}")
    else:
        st.error(str(err))
    st.caption(
        "Your previous results are unchanged and still shown." if has_results
        else "Nothing was saved."
    )


def sidebar(data: pd.DataFrame | None, has_results: bool) -> None:
    with st.sidebar:
        st.markdown("<div class='side-title'>Scenario Configuration</div>", unsafe_allow_html=True)
        st.caption("Configure your placement scenario.")
        n_students = st.number_input("Students", min_value=5, max_value=100, value=config.NUM_STUDENTS, step=1)
        n_companies = st.number_input("Companies", min_value=2, max_value=12, value=config.NUM_COMPANIES, step=1)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=config.RANDOM_SEED, step=1)
        limit = st.number_input("Solver time limit (sec)", min_value=10, max_value=600,
                                value=int(config.SOLVER_TIME_LIMIT_SEC), step=10)
        clicked = st.button("Generate & Optimize", type="primary")
        st.caption("The new scenario will replace the current generated results.")

        if clicked:
            try:
                with st.spinner(f"Solving, up to {int(limit)} seconds..."):
                    result = run_pipeline(int(n_students), int(n_companies), int(seed), int(limit))
            except ImportError:
                st.error("PuLP is not installed. Run: pip install -r requirements.txt")
                st.caption("Your previous results are unchanged." if has_results else "Nothing was saved.")
            except (ValueError, RuntimeError) as err:
                show_run_error(err, has_results)
            else:
                summary = summarize_run(result)
                save_run_summary(summary)
                st.session_state["last_run"] = summary
                st.session_state["flash"] = (
                    f"New scenario generated: {int(n_students)} students, {int(n_companies)} companies, "
                    f"seed {int(seed)}. The previous results were replaced."
                )
                st.rerun()

        if data is not None:
            st.caption(
                f"Loaded now: {data['student_id'].nunique()} students, "
                f"{data['company_id'].nunique()} companies, {len(data)} interviews."
            )


def show_missing_files() -> None:
    st.info("No schedules to show yet. Create them from the project root:")
    st.code(
        "python -m scheduler.data_generator\n"
        "python -m scheduler.baseline\n"
        "python -m scheduler.optimizer",
        language="bash",
    )
    st.write("Or use Generate & Optimize in the sidebar.")


def optimization_details(run: dict | None) -> None:
    """The mathematical side, for the viva: priorities, model size, stage-by-stage solver results."""
    with st.expander("Optimization Details"):
        st.markdown("**Optimization priorities** (each is only improved if the ones above it stay as good)")
        st.markdown(
            "1. No interview clashes\n"
            f"2. The {config.BUFFER_MINUTES}-minute buffer between a student's interviews\n"
            "3. Shortest longest individual excess wait\n"
            "4. Shortest total excess wait\n"
            "5. Shortest placement day, if possible"
        )
        if not run:
            st.caption(
                "No solver run is recorded for these timetables, so there are no model or stage results "
                "to show. Run Generate & Optimize from the sidebar to fill this section."
            )
            return

        model = run.get("model")
        if model and model[0]:
            st.markdown("**MILP model size**")
            for col, label, value in zip(st.columns(3), ["Variables", "Binary variables", "Constraints"], model):
                col.markdown(metric_card_html(label, f"{int(value):,}"), unsafe_allow_html=True)

        stage_data = pd.DataFrame(run.get("stages") or [])
        if not stage_data.empty:
            st.markdown("**Stage-by-stage solver results**")
            st.dataframe(
                stage_data.rename(columns={
                    "name": "Optimization stage",
                    "status": "Solver status",
                    "seconds": "Time used (s)",
                    "limit": "Time allowed (s)",
                    "value": "Result (minutes)",
                }),
                hide_index=True,
            )
        steps = pd.DataFrame(run.get("trajectory") or [])
        if not steps.empty:
            st.markdown("**Step-by-step results** (how the numbers moved)")
            st.dataframe(
                steps.rename(columns={
                    "step": "Step",
                    "longest": "Longest wait (min)",
                    "total": "Total wait (min)",
                    "day": "Day duration (min)",
                    "note": "Note",
                }),
                hide_index=True,
            )
            st.caption(
                "Day duration is the end of the last interview, counted from the start of the day. "
                "Priorities are compared in this order: longest wait, then total wait, then day duration."
            )
        if run.get("start_note"):
            st.caption(run["start_note"])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def overview_tab(data, baseline, optimized) -> None:
    mb, mo = compute_metrics(baseline), compute_metrics(optimized)
    run = current_run(optimized)

    st.markdown(
        f"<p class='lead'>{esc(headline(mb['total_excess_wait_minutes'], mo['total_excess_wait_minutes']))}</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Interviews run {config.DAY_START} to {config.DAY_END}. Students need {config.BUFFER_MINUTES} minutes "
        "between interviews, so only waiting beyond that buffer counts as excess waiting."
    )

    rows = overview_rows(mb, mo)
    for col, r in zip(st.columns(len(rows)), rows):
        col.markdown(
            metric_card_html(r["label"], r["optimized"], delta_text(r["diff"], r["unit"]), r["state"]),
            unsafe_allow_html=True,
        )
    st.markdown(comparison_html(rows), unsafe_allow_html=True)

    section_title("Excess waiting by student")
    st.caption(
        f"Waiting beyond the {config.BUFFER_MINUTES}-minute buffer. "
        "Students with a single interview cannot wait and are left out."
    )
    fig = excess_by_student_figure(baseline, optimized)
    if fig is None:
        st.info("Nobody has more than one interview, so there is no waiting to compare.")
    else:
        st.pyplot(fig)

    section_title("Solver status")
    info = solver_status_info(run["status"] if run else None, bool(run and run.get("fallback_used")))
    if run and run.get("seconds") is not None:
        info["note"] = f"{info['note']} Solved in {_fmt(run['seconds'])} seconds.".strip()
    st.markdown(
        status_html(info["tone"], info["label"], info["title"], info["body"], info["note"]),
        unsafe_allow_html=True,
    )
    show_checks(baseline, optimized, data)

    with st.expander("All metrics"):
        st.dataframe(comparison_table(baseline, optimized), hide_index=True)
    optimization_details(run)


def schedule_tab(data, baseline, optimized) -> None:
    colors = company_colors(data, baseline, optimized)
    section_title("Interview Schedule Visualization")
    left, middle, right = st.columns(3)
    group = left.radio("Rows are", ["Students", "Companies"], horizontal=True, key="group")
    which = middle.radio("Schedule", ["Optimized", "Baseline"], horizontal=True, key="schedule_view")
    days = sorted(set(baseline["day"]) | set(optimized["day"]))
    day = right.selectbox("Day", days, key="day") if len(days) > 1 else days[0]

    schedule = optimized if which == "Optimized" else baseline
    total = compute_metrics(schedule)["total_excess_wait_minutes"]
    title = f"{which}: {_fmt(total)} min of excess waiting"
    draw = student_timetable_figure if group == "Students" else company_timetable_figure
    st.pyplot(draw(schedule, colors, int(day), title))

    with st.expander(f"{which} timetable (table and download)"):
        table = timetable_table(schedule)
        st.dataframe(table, hide_index=True)
        st.download_button(
            f"Download {which.lower()} timetable (csv)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{which.lower()}_timetable.csv",
            mime="text/csv",
        )


def student_tab(baseline, optimized) -> None:
    students = sorted(set(baseline["student_id"]) & set(optimized["student_id"]))
    if not students:
        st.info("No student appears in both schedules.")
        return
    student = st.selectbox("Select student", students, key="student")
    name = optimized.loc[optimized["student_id"] == student, "student_name"].iloc[0]
    wb = waiting_times(baseline).set_index("student_id").loc[student]
    wo = waiting_times(optimized).set_index("student_id").loc[student]

    st.markdown(
        f"<p class='student-head'>Student ID: <strong>{esc(str(student))}</strong> &middot; {esc(str(name))}</p>",
        unsafe_allow_html=True,
    )
    base_wait, opt_wait = int(wb["excess_wait_minutes"]), int(wo["excess_wait_minutes"])
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card_html("Total interviews", str(int(wo["num_interviews"]))), unsafe_allow_html=True)
    c2.markdown(metric_card_html("Baseline excess wait", f"{base_wait} min"), unsafe_allow_html=True)
    c3.markdown(
        metric_card_html("Optimized excess wait", f"{opt_wait} min",
                         delta_text(opt_wait - base_wait), change_state(opt_wait - base_wait)),
        unsafe_allow_html=True,
    )

    section_title("Optimized interview itinerary")
    st.dataframe(itinerary_display(optimized, student), hide_index=True)
    with st.expander("Baseline itinerary, for comparison"):
        st.dataframe(itinerary_display(baseline, student), hide_index=True)


def data_tab(data: pd.DataFrame) -> None:
    st.caption("Each shortlist represents one interview that must be assigned a time slot.")
    with st.expander("Company Details", expanded=True):
        st.dataframe(company_details_table(data), hide_index=True)
    with st.expander("Student Summary"):
        st.dataframe(student_summary_table(data), hide_index=True)
    with st.expander("Interview Shortlists"):
        st.dataframe(shortlist_table(data), hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Smart Placement Scheduler", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-eyebrow'>PLACEMENT MANAGEMENT SYSTEM</p>"
        "<div class='app-title'>Smart Placement Scheduler</div>"
        "<p class='app-sub'>Intelligent interview scheduling with optimized waiting times.</p>",
        unsafe_allow_html=True,
    )

    try:
        data = load_problem()
    except ValueError as err:
        sidebar(None, False)
        st.error(f"data/sample_data.csv is not usable: {err}")
        return

    baseline = load_csv(config.BASELINE_SCHEDULE_PATH)
    optimized = load_csv(config.OPTIMIZED_SCHEDULE_PATH)
    has_results = data is not None and baseline is not None and optimized is not None
    sidebar(data, has_results)

    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash)
    if not has_results:
        show_missing_files()
        return

    tab_overview, tab_schedule, tab_student, tab_data = st.tabs(
        ["Overview", "Interview Schedule", "Student Schedule", "Input Data"]
    )
    with tab_overview:
        overview_tab(data, baseline, optimized)
    with tab_schedule:
        schedule_tab(data, baseline, optimized)
    with tab_student:
        student_tab(baseline, optimized)
    with tab_data:
        data_tab(data)


if __name__ == "__main__":
    main()
