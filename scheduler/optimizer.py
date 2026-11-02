"""
MILP optimizer for the Smart Placement Slot Scheduler (PuLP + CBC).

Priorities (strict order, highest first)
----------------------------------------
    1. Prevent all interview clashes          -> hard constraint
    2. Respect the minimum buffer (15 min)    -> hard constraint
    3. Minimize the MAXIMUM individual excess waiting
    4. Minimize the TOTAL excess waiting
    5. If possible, minimize the placement-day duration

Priorities 1 and 2 are constraints: no schedule that breaks them is ever
returned, whatever it would gain elsewhere. Priorities 3 to 5 are solved
one after the other ("lexicographic" optimization). Each stage optimizes ONE
criterion and then LOCKS its best value, so a later stage can only choose among
schedules that are equally good on every earlier criterion. A lower priority can
therefore never be bought with a worse higher priority, which a weighted sum
cannot guarantee.

    stage 1: minimize  max_s excess_s                    (fairness to the worst-off student)
    stage 2: minimize  sum_s excess_s   s.t. max <= best from stage 1
    stage 3: minimize  last_end (day length)             s.t. max <= stage 1, total <= stage 2

Model
-----
Interview i = one shortlisted (student, company) pair with duration d_i slots.

Decision variable
    x[i, t] in {0,1}   interview i starts at slot t   (t in config.get_valid_start_slots(d_i))

Hard constraints
    (1) every interview starts exactly once:      sum_t x[i,t] = 1
    (2) student clash + buffer: for every student and slot, at most one
        interview is "occupying" them, where an interview occupies its own
        slots PLUS the BUFFER_SLOTS after it (15 min by default). So two of a
        student's interviews can never overlap or be closer than the buffer.
    (3) company clash: for every company and slot, at most `panels`
        interviews are running

Excess waiting time (gap-based, linear, no big-M)
    busy[s,t]   = 1 if student s is in an interview during slot t   (a sum of x)
    rest[s,t]   = 1 if slot t is inside the buffer right after one of s's
                  interviews                                          (a sum of x)
    before[s,t] >= 1 if s has been busy at any slot <= t that day   (continuous 0..1)
    after[s,t]  >= 1 if s is busy at any slot >= t that day         (continuous 0..1)
    excess[s,t] >= before + after - 1 - busy - rest   (an idle slot BETWEEN two
                                                       interviews that is not
                                                       covered by the buffer)

    excess_s = sum_t excess[s,t]   (lunch slots are skipped, each day is separate)

For two consecutive interviews with a gap of g slots this adds up to
max(0, g - BUFFER_SLOTS): a 15-minute gap costs nothing, a 45-minute gap costs
30 minutes. It is the same quantity metrics.waiting_times() reports as
excess_wait_slots, so the optimizer and the dashboard always agree.

Placement-day duration (stage 3 only)
    last_end >= end slot of every interview
    duration  = last_end        (time from the start of the day to the end of the last interview)
    Minimizing it makes the placement day finish as early as possible. With the
    default setup (one day, no lunch break) a schedule can always be slid to the
    start of the day, so this is the same as "first start to last end"
    (metrics "makespan_minutes").
    The day can never be shorter than the busiest company's workload (its
    interviews divided by its panels) or the busiest student's own interviews
    plus buffers. That bound is added as a valid cut: it costs nothing, and when
    a schedule reaches it the solver can stop at once, knowing it is optimal.

Time limit
    time_limit is the budget for the whole run. It is shared between the three
    stages (STAGE_TIME_SHARE); time a stage does not need rolls over to the next.
    If a stage hits its limit, its best schedule so far is locked in and the
    status says "Feasible (not proven optimal)".

Starting point and safety net
    When a valid warm start (the baseline) is given, heuristic.improve_schedule()
    first tries to make it better with a quick local search, and CBC starts from
    the better of the two. At the end the result is compared with that starting
    timetable in priority order: if the solver's answer is somehow WORSE (e.g. a
    time limit hit before CBC beat its own start), the starting timetable is
    returned instead, result.fallback_used is True and the status is never
    "Optimal".

Run from the project root (after generating data and running the baseline):

    python -m scheduler.optimizer
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field, replace

import pandas as pd
import pulp

from scheduler import config
from scheduler.data_generator import load_sample_data
from scheduler.heuristic import improve_schedule, priority_key
from scheduler.metrics import (
    assert_valid,
    build_schedule,
    compute_metrics,
    schedule_to_starts,
    validate_schedule,
)

# How the time limit is split between stage 1 (longest wait), stage 2 (total
# wait) and stage 3 (day duration). Leftover time rolls forward.
STAGE_TIME_SHARE = (0.4, 0.4, 0.2)

# The starting-timetable heuristic may use this share of the time limit (at most
# START_HEURISTIC_MAX_SECONDS); it stops earlier when no move helps any more.
START_HEURISTIC_SHARE = 0.1
START_HEURISTIC_MAX_SECONDS = 5.0

# On Windows PuLP only hands CBC the warm start when keepFiles=True ("warmStart requires
# keepFiles=True"). Without it the starting timetable is silently ignored. The scratch files
# this leaves in the working directory (<problem name>-pulp.*) are deleted after every solve.
KEEP_FILES_FOR_WARM_START = os.name == "nt"


def _remove_solver_files(problem_name: str) -> None:
    """Best-effort cleanup of the files PuLP keeps when keepFiles=True."""
    for path in glob.glob(f"{problem_name}-pulp.*"):
        try:
            os.remove(path)
        except OSError:
            pass


OPTIMAL = "Optimal"
FEASIBLE = "Feasible (not proven optimal)"
FALLBACK = "Starting timetable kept (solver result was worse)"
REJECTED = "Solver result rejected (an earlier priority got worse); previous schedule kept"


@dataclass
class StageResult:
    name: str          # what this stage minimized
    value: int         # best value reached, measured on the schedule (in slots)
    status: str        # OPTIMAL, FEASIBLE, or "Skipped (already optimal)"
    seconds: float
    time_limit: int = 0  # seconds CBC was allowed for this stage (0 = stage skipped)


@dataclass
class Snapshot:
    """The three priority numbers of the best schedule known at one step of the run (all in slots)."""
    key: str           # stable id: baseline, improved_start, stage1, stage2, stage3, final
    label: str         # readable name for tables
    longest: int       # priority 3: longest individual excess wait
    total: int         # priority 4: total excess wait
    day: int           # priority 5: end of the last interview, counted from the start of the day
    note: str = ""     # e.g. the stage's solver status, "skipped", "rejected"


@dataclass
class OptimizationResult:
    schedule: pd.DataFrame
    status: str                       # OPTIMAL only if every stage was proven optimal
    max_excess_wait_slots: int        # priority 3, measured on the returned schedule
    total_excess_wait_slots: int      # priority 4, measured on the returned schedule
    day_duration_slots: int           # priority 5, start of the day to the last interview's end
    solve_seconds: float
    stages: list[StageResult] = field(default_factory=list)
    fallback_used: bool = False       # True if the starting timetable beat the solver's answer
    start_improved: bool = False      # True if that starting timetable was the improved baseline, not the plain baseline
    start_note: str = ""              # what happened to the warm start before solving
    num_variables: int = 0            # size of the core model (before the stage-3 day-length cut)
    num_binaries: int = 0
    num_constraints: int = 0
    trajectory: list[Snapshot] = field(default_factory=list)  # how the numbers moved, step by step


def _day_duration_lower_bound(data: pd.DataFrame) -> int:
    """
    A duration (in slots) that no valid schedule can beat.

    * A company runs at most `panels` interviews at once, so its interviews take
      at least ceil(total interview slots / panels) slots from first to last.
    * A student's interviews cannot overlap, and on a single day they also need
      the buffer between them.
    """
    per_company = data.groupby("company_id").agg(
        slots=("duration_slots", "sum"), panels=("panels", "first")
    )
    company_bound = int((-(-per_company["slots"] // per_company["panels"])).max())

    per_student = data.groupby("student_id")["duration_slots"].agg(["sum", "count"])
    buffer = config.BUFFER_SLOTS if config.NUM_DAYS == 1 else 0
    student_bound = int((per_student["sum"] + (per_student["count"] - 1) * buffer).max())
    return max(company_bound, student_bound)


def _measure(schedule: pd.DataFrame) -> tuple[int, int, int]:
    """(longest excess wait, total excess wait, day duration = last end slot), all in slots, read from a schedule."""
    return priority_key(schedule)


def _keep_start_if_better(
    best: pd.DataFrame, stages: list[StageResult], start_schedule: pd.DataFrame | None
) -> tuple[pd.DataFrame, list[StageResult], bool]:
    """
    Safety net: never return a schedule that is worse, in priority order, than
    the valid starting timetable. Returns (schedule, stages, fallback_used). When
    the start wins, each stage's value is re-read from the returned schedule so
    the numbers always describe the timetable that is actually returned, and no
    stage keeps an "Optimal"/"Feasible" label. A stage that was REJECTED keeps that
    label, because it says what happened to that stage's own result.
    """
    if start_schedule is None:
        return best, stages, False
    start_key = priority_key(start_schedule)
    if start_key >= priority_key(best):
        return best, stages, False
    return (
        start_schedule,
        [replace(st, value=v, status=st.status if st.status == REJECTED else FALLBACK)
         for st, v in zip(stages, start_key)],
        True,
    )


def optimize_schedule(
    data: pd.DataFrame,
    time_limit: int = config.SOLVER_TIME_LIMIT_SEC,
    gap_rel: float = config.SOLVER_GAP_REL,
    msg: bool = config.SOLVER_MSG,
    warm_start: pd.DataFrame | None = None,
) -> OptimizationResult:
    """
    Solve the scheduling problem in priority order and return a validated schedule.

    time_limit: seconds for the WHOLE run (all three stages together).
    warm_start: an existing valid schedule (e.g. the baseline). CBC starts from
    it, so a good answer is available even if a time limit is hit early.
    """
    T = config.TOTAL_SLOTS
    SPD = config.SLOTS_PER_DAY
    blocked = config.get_blocked_slots()

    pairs = [(r.student_id, r.company_id, int(r.duration_slots)) for r in data.itertuples(index=False)]
    panels_of = data.groupby("company_id")["panels"].first().to_dict()

    prob = pulp.LpProblem("placement_scheduler", pulp.LpMinimize)

    # ---- decision variables x[(s,c), t] ----------------------------------
    x: dict[tuple[str, str, int], pulp.LpVariable] = {}
    starts_of: dict[tuple[str, str], list[int]] = {}
    for s, c, d in pairs:
        valid = config.get_valid_start_slots(d)
        starts_of[(s, c)] = valid
        for t in valid:
            x[s, c, t] = pulp.LpVariable(f"x_{s}_{c}_{t}", cat="Binary")

    # (1) each interview starts exactly once
    for s, c, d in pairs:
        prob += pulp.lpSum(x[s, c, t] for t in starts_of[(s, c)]) == 1, f"once_{s}_{c}"

    # ---- priorities 1 and 2: student clashes + required buffer -------------
    by_student: dict[str, list[tuple[str, int]]] = {}
    for s, c, d in pairs:
        by_student.setdefault(s, []).append((c, d))

    student_occupying: dict[tuple[str, int], list[pulp.LpVariable]] = {}  # interview + buffer
    student_busy: dict[tuple[str, int], list[pulp.LpVariable]] = {}       # interview only
    student_rest: dict[tuple[str, int], list[pulp.LpVariable]] = {}       # buffer only
    for s, items in by_student.items():
        for c, d in items:
            for start in starts_of[(s, c)]:
                for t in config.student_occupied_slots(start, d):
                    student_occupying.setdefault((s, t), []).append(x[s, c, start])
                for t in range(start, start + d):
                    student_busy.setdefault((s, t), []).append(x[s, c, start])
                for t in config.rest_slots(start, d):
                    student_rest.setdefault((s, t), []).append(x[s, c, start])

    for (s, t), vars_ in student_occupying.items():
        if len(vars_) > 1:
            prob += pulp.lpSum(vars_) <= 1, f"student_clash_{s}_{t}"

    # ---- priority 1: company clashes ---------------------------------------
    company_running: dict[tuple[str, int], list[pulp.LpVariable]] = {}
    for s, c, d in pairs:
        for start in starts_of[(s, c)]:
            for t in range(start, start + d):
                company_running.setdefault((c, t), []).append(x[s, c, start])

    for (c, t), vars_ in company_running.items():
        if len(vars_) > panels_of[c]:
            prob += pulp.lpSum(vars_) <= panels_of[c], f"company_clash_{c}_{t}"

    # ---- excess waiting time (used by priorities 3 and 4) -------------------
    # Students with a single interview can never wait, so they get no variables.
    excess_expr: dict[str, pulp.LpAffineExpression] = {}
    for s, items in by_student.items():
        if len(items) < 2:
            continue
        before = {t: pulp.LpVariable(f"before_{s}_{t}", 0, 1) for t in range(T)}
        after = {t: pulp.LpVariable(f"after_{s}_{t}", 0, 1) for t in range(T)}
        excess = {t: pulp.LpVariable(f"excess_{s}_{t}", 0, 1) for t in range(T) if t not in blocked}

        for t in range(T):
            busy_t = pulp.lpSum(student_busy.get((s, t), []))
            rest_t = pulp.lpSum(student_rest.get((s, t), []))
            prob += before[t] >= busy_t, f"before_busy_{s}_{t}"
            prob += after[t] >= busy_t, f"after_busy_{s}_{t}"
            if t % SPD != 0:                      # carry "already started today" forward
                prob += before[t] >= before[t - 1], f"before_carry_{s}_{t}"
            if (t + 1) % SPD != 0:                # carry "still to come today" backward
                prob += after[t] >= after[t + 1], f"after_carry_{s}_{t}"
            if t not in blocked:
                # idle between two interviews AND not inside the buffer -> excess
                prob += excess[t] >= before[t] + after[t] - 1 - busy_t - rest_t, f"excess_{s}_{t}"

        excess_expr[s] = pulp.lpSum(excess.values())

    max_excess = pulp.LpVariable("max_excess_wait", 0)
    for s, expr in excess_expr.items():
        prob += max_excess >= expr, f"max_excess_{s}"
    total_excess = pulp.lpSum(excess_expr.values())

    model_size = (prob.numVariables(), len(x), prob.numConstraints())

    # ---- starting timetable --------------------------------------------------
    t_start = time.time()
    deadline = t_start + time_limit

    trajectory: list[Snapshot] = []
    start_schedule: pd.DataFrame | None = None  # best VALID timetable known before solving
    start_note = ""
    start_improved = False
    if warm_start is not None:
        initial = warm_start
        if validate_schedule(warm_start, data):
            start_note = "The given warm start is not valid, so it was used only as a hint."
        else:
            start_schedule = warm_start
            budget = min(START_HEURISTIC_MAX_SECONDS, START_HEURISTIC_SHARE * time_limit)
            improved = improve_schedule(data, warm_start, budget)
            if priority_key(improved) < priority_key(warm_start):
                start_schedule = initial = improved
                start_improved = True
                start_note = "The starting timetable was improved before solving."
            else:
                start_note = "The baseline was already the best starting timetable found."
        given = schedule_to_starts(initial)
        for (s, c, t), var in x.items():
            var.setInitialValue(1 if given.get((s, c)) == t else 0)
        # diagnostics: where the run starts (priority_key, not _measure, so these rows are plain readings)
        trajectory.append(Snapshot("baseline", "Starting baseline", *priority_key(warm_start),
                                   "" if start_schedule is not None else "not a valid schedule"))
        if start_schedule is not None:
            trajectory.append(Snapshot("improved_start", "Improved starting schedule", *priority_key(start_schedule),
                                       "improved" if start_improved else "no improvement found"))
    use_warm = warm_start is not None

    def stage_time_limit(k: int) -> int:
        """Seconds for stage k: its share of what is left, so unused time rolls forward."""
        left = max(deadline - time.time(), 1.0)
        return max(1, int(left * STAGE_TIME_SHARE[k] / sum(STAGE_TIME_SHARE[k:])))

    stage_limits: dict[int, int] = {}  # the time limit each stage was actually given
    x_names = {var.name for var in x.values()}

    def restore_start(schedule: pd.DataFrame) -> None:
        """Make `schedule` the warm start of the next solve (used when a stage's result is rejected)."""
        given_now = schedule_to_starts(schedule)
        for (s_, c_, t_), var in x.items():
            var.setInitialValue(1 if given_now.get((s_, c_)) == t_ else 0)
        for var in prob.variables():
            if var.name not in x_names:
                var.varValue = None  # helper values of the rejected solution no longer match


    def solve_stage(k: int) -> tuple[pd.DataFrame | None, str]:
        """Run CBC on the current objective. Returns (schedule or None, status)."""
        # Stage 1 starts from the caller's warm start (if any); later stages always
        # start from the schedule the previous stage just found.
        stage_limits[k] = stage_time_limit(k)
        solver = pulp.PULP_CBC_CMD(
            msg=msg,
            timeLimit=stage_limits[k],
            gapRel=gap_rel,
            warmStart=use_warm or k > 0,
            keepFiles=KEEP_FILES_FOR_WARM_START,
        )
        prob.solve(solver)
        if KEEP_FILES_FOR_WARM_START:
            _remove_solver_files(prob.name)
        # PuLP reports "Optimal" even when CBC stopped on the time limit, so look
        # at sol_status (1 = proven optimal, 2 = feasible but not proven).
        sol_status = getattr(prob, "sol_status", None)
        chosen = {
            (s, c): t
            for (s, c, t), var in x.items()
            if var.value() is not None and var.value() > 0.5
        }
        if len(chosen) != len(pairs):
            return None, "no solution"
        status = OPTIMAL if sol_status == pulp.constants.LpSolutionOptimal else FEASIBLE
        return build_schedule(data, chosen), status

    stages: list[StageResult] = []
    best: pd.DataFrame | None = None
    all_optimal = True

    # ---- stage 1 (priority 3): minimize the longest individual excess wait ---
    t0 = time.time()
    prob.setObjective(max_excess)
    schedule, status = solve_stage(0)
    if schedule is None:
        raise RuntimeError(
            f"Solver found no complete schedule (status: {pulp.LpStatus[prob.status]}). "
            "Try a longer time limit, more days, or fewer shortlists."
        )
    best = schedule
    max_val, total_val, day_val = _measure(best)
    stages.append(StageResult("Longest excess wait", max_val, status, round(time.time() - t0, 2), stage_limits.get(0, 0)))
    all_optimal &= status == OPTIMAL
    trajectory.append(Snapshot("stage1", "After stage 1", max_val, total_val, day_val, status))
    prob += max_excess <= max_val, "lock_max_excess"       # later stages may not make it worse

    # ---- stage 2 (priority 4): minimize the total excess wait ----------------
    t0 = time.time()
    if max_val == 0:
        # Nobody waits beyond the buffer, so the total is 0 as well: nothing to optimize.
        stages.append(StageResult("Total excess wait", 0, "Skipped (already optimal)", 0.0))
    else:
        prob.setObjective(total_excess)
        schedule, status = solve_stage(1)
        if schedule is not None:
            candidate = _measure(schedule)
            # Accept only if priority 3 held and the total did not get worse than the stage-1 schedule.
            # The model locks the longest wait, but never trust the solver blindly: check it here.
            if candidate[0] <= max_val and candidate[1] <= total_val:
                best = schedule
                max_val, total_val, day_val = candidate
            else:
                status = REJECTED
                restore_start(best)  # stage 3 should start from the schedule we kept, not the rejected one
        else:
            status = FEASIBLE  # keep the stage-1 schedule
        stages.append(StageResult("Total excess wait", total_val, status, round(time.time() - t0, 2), stage_limits.get(1, 0)))
        all_optimal &= status == OPTIMAL
    trajectory.append(Snapshot("stage2", "After stage 2", max_val, total_val, day_val, stages[-1].status))
    if excess_expr:  # (nothing to lock when no student has two interviews)
        prob += total_excess <= total_val, "lock_total_excess"

    # ---- stage 3 (priority 5): minimize the placement-day duration -----------
    t0 = time.time()
    lower_bound = _day_duration_lower_bound(data)
    if day_val <= lower_bound:
        # The schedule already reaches the shortest possible day: nothing to improve.
        stages.append(StageResult("Day duration", day_val, "Skipped (already optimal)", 0.0))
    else:
        last_end = pulp.LpVariable("last_end", 0, T)
        for s_id, c_id, d in pairs:
            end_expr = pulp.lpSum((t + d) * x[s_id, c_id, t] for t in starts_of[(s_id, c_id)])
            prob += last_end >= end_expr, f"last_end_{s_id}_{c_id}"
        prob += last_end >= lower_bound, "day_duration_lower_bound"
        prob.setObjective(last_end)
        schedule, status = solve_stage(2)
        if schedule is not None:
            candidate = _measure(schedule)
            # Only accept it if priorities 3 and 4 held (they must, unless a bug or a tolerance slipped).
            if candidate[0] <= max_val and candidate[1] <= total_val and candidate[2] <= day_val:
                best = schedule
                max_val, total_val, day_val = candidate
            else:
                status = REJECTED
        else:
            status = FEASIBLE
        stages.append(StageResult("Day duration", day_val, status, round(time.time() - t0, 2), stage_limits.get(2, 0)))
        all_optimal &= status == OPTIMAL

    trajectory.append(Snapshot("stage3", "After stage 3", max_val, total_val, day_val, stages[-1].status))

    best, stages, fallback_used = _keep_start_if_better(best, stages, start_schedule)
    if fallback_used:
        max_val, total_val, day_val = _measure(best)
        all_optimal = False
    trajectory.append(Snapshot(
        "final", "Final returned schedule", max_val, total_val, day_val,
        "starting timetable kept" if fallback_used else "",
    ))

    assert_valid(best, data)  # never return an invalid timetable

    return OptimizationResult(
        schedule=best,
        status=OPTIMAL if all_optimal else FEASIBLE,
        max_excess_wait_slots=max_val,
        total_excess_wait_slots=total_val,
        day_duration_slots=day_val,
        solve_seconds=round(time.time() - t_start, 2),
        stages=stages,
        fallback_used=fallback_used,
        start_note=start_note,
        start_improved=start_improved,
        trajectory=trajectory,
        num_variables=model_size[0],
        num_binaries=model_size[1],
        num_constraints=model_size[2],
    )


def main() -> None:
    from scheduler.baseline import baseline_schedule

    data = load_sample_data()
    result = optimize_schedule(data, warm_start=baseline_schedule(data))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.schedule.to_csv(config.OPTIMIZED_SCHEDULE_PATH, index=False)

    print(f"Optimized schedule saved to {config.OPTIMIZED_SCHEDULE_PATH}")
    print(f"  status: {result.status} ({result.solve_seconds}s)")
    print(f"  model: {result.num_variables} variables ({result.num_binaries} binary), {result.num_constraints} constraints")
    if result.start_note:
        print(f"  start: {result.start_note}")
    if result.fallback_used:
        which = "improved baseline" if result.start_improved else "baseline"
        print(f"  note: the solver's answer was worse than the starting timetable ({which}), so that timetable is returned")
    print("  step-by-step (longest wait / total wait / day end, minutes):")
    for snap in result.trajectory:
        m = config.SLOT_MINUTES
        note = f"  [{snap.note}]" if snap.note else ""
        print(f"    {snap.label:<28} {snap.longest * m:>5} {snap.total * m:>6} {snap.day * m:>6}{note}")
    for i, stage in enumerate(result.stages, start=3):
        minutes = stage.value * config.SLOT_MINUTES
        print(f"  priority {i} - {stage.name}: {minutes} min [{stage.status}, {stage.seconds}s of {stage.time_limit}s allowed]")
    for key, value in compute_metrics(result.schedule).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
