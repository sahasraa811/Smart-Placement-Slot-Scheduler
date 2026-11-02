# Smart Placement Slot Scheduler

Builds a timetable for campus placement interviews that has **no clashes**, keeps the required **buffer** between a student's interviews, and then keeps students' **waiting time** and the **length of the placement day** as low as possible, in a fixed order of priority. It compares a simple first-come-first-served schedule with a schedule optimized by a mixed-integer linear program (MILP), and shows the difference in a Streamlit dashboard.

## What it does

1. **Sample data:** generates students, companies, interview durations and the shortlists (which company shortlisted which student).
2. **Baseline schedule:** places interviews first-come-first-served, with no thought for waiting. This is the "before".
3. **Optimized schedule:** solves a MILP with Python, PuLP and the CBC solver to assign start times.
4. **No clashes:** a student is never in two interviews at once, and a company never runs more interviews at once than it has panels.
5. **Less waiting, shorter day:** within those rules, the longest individual wait is made as small as possible first, then the total waiting, then the length of the day (see [Optimization priorities](#optimization-priorities)).
6. **Comparison and dashboard:** checks whether waiting really went down, and shows the timetables, waiting times and comparison.

## How waiting time is measured

Waiting time is the **gap between a student's interviews**, not the number of time slots a schedule uses. A schedule that ends late is not automatically a waiting one.

Nobody wants zero waiting, because students need time to move between rooms and reset. So every student gets a required **buffer** (15 minutes by default) between two interviews on the same day:

- The buffer is a **hard rule**. Back-to-back interviews are not allowed, in the baseline and the optimized schedule alike.
- Waiting up to the buffer is free. Only waiting **beyond the buffer** counts as **excess waiting**, and excess waiting is what the optimizer minimizes.

Example: a student has one interview from 09:00 to 09:30 and the next from 10:15 to 10:45. The gap is 45 minutes. 15 minutes of it is the buffer, so the excess waiting is 30 minutes. For several interviews, this is worked out for each pair of consecutive interviews and added up (`max(0, gap - buffer)` each).

Time before a student's first interview and after their last one is never waiting, and an optional lunch break is not counted as waiting. Each day is measured separately.

## Optimization priorities

The optimizer works through these goals in strict order. A goal is only improved among the schedules that are already best on every goal above it.

| Priority | Objective | Kind |
|---|---|---|
| 1 | Prevent all interview clashes (students and company panels) | Hard rule |
| 2 | Respect the 15-minute minimum buffer between a student's interviews | Hard rule |
| 3 | Minimize the **maximum** individual excess waiting | Optimized first |
| 4 | Minimize the **total** excess waiting | Optimized second |
| 5 | If possible, minimize the placement-day duration (the last interview ends as early as possible) | Optimized last |

Priorities 1 and 2 are never traded away: every schedule the optimizer returns is checked against them. Priorities 3 to 5 are solved one after the other. For example, a schedule where nobody waits more than 15 minutes beyond the buffer is always preferred to one with a lower total but one student who waits 30 minutes, and a shorter day is only chosen if it does not add any waiting.

## Example results

From the bundled sample data (20 students, 5 companies, 37 shortlists, seed 4) with a 60-second solver limit:

| | Baseline | Optimized |
|---|---|---|
| Longest excess wait (priority 3) | 210 min | 0 min |
| Total excess waiting (priority 4) | 645 min | 0 min |
| Placement-day length (priority 5) | 420 min (ends 16:00) | 420 min (ends 16:00) |
| Average per student | 32.25 min | 0 min |
| Students waiting beyond the buffer | 9 | 0 |

For this data, zero excess waiting is possible, so priorities 3 and 4 reach 0. The day cannot be shorter than 7 hours either: Company C2 has seven 60-minute interviews and only one panel. The baseline happens to end at the same time, but with 645 minutes of excess waiting. On harder data the three numbers will not all reach their best value, and on a very large problem the solver may stop at its time limit before proving a stage optimal (the status says so). The numbers above were proven optimal. The Overview page of the dashboard always shows the numbers for the files in `results/`.

## Project structure

```
Smart-Placement-Scheduler/
├── data/
│   └── sample_data.csv            shortlists: one row per student and company
├── results/
│   ├── baseline_schedule.csv      first-come-first-served timetable
│   ├── optimized_schedule.csv     MILP timetable
│   ├── run_summary.json           status and details of the last run from the dashboard (created on first use)
│   └── experiments.csv            timings from scheduler/experiments.py (created on first use)
├── scheduler/
│   ├── __init__.py
│   ├── config.py                  every setting: day length, slot size, buffer, solver limits
│   ├── data_generator.py          creates, validates, saves and loads the sample data
│   ├── baseline.py                greedy first-come-first-served schedule
│   ├── heuristic.py               local search that improves the baseline into a stronger starting schedule
│   ├── optimizer.py               the MILP (PuLP + CBC)
│   ├── metrics.py                 waiting time, comparison and the schedule validator
│   └── experiments.py             scalability experiments: stage-by-stage solver timings on larger data
├── tests/
│   ├── test_clashes.py            clash, buffer and panel rules
│   ├── test_waiting_time.py       waiting-time definition and optimizer vs metrics
│   ├── test_priorities.py         priority order, checked against an exhaustive search
│   ├── test_start_and_fallback.py heuristic result is valid, and the optimizer is never worse than its start
│   └── test_dashboard.py          dashboard helper functions and charts
├── .streamlit/config.toml         dashboard theme
├── app.py                         Streamlit dashboard
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

You need Python 3.10 or newer. Run everything from the project root.

```bash
pip install -r requirements.txt
```

`pulp[cbc]` installs the CBC solver together with PuLP, so there is nothing else to set up.

Create the data and both schedules (the repository already contains a copy of each):

```bash
python -m scheduler.data_generator    # data/sample_data.csv
python -m scheduler.baseline          # results/baseline_schedule.csv
python -m scheduler.optimizer         # results/optimized_schedule.csv
```

Each command prints its summary. The data generator takes options:

```bash
python -m scheduler.data_generator --students 30 --companies 6 --seed 1
```

The same seed always gives the same data. Regenerate the data, then re-run the baseline and the optimizer, because the schedules belong to the data they were built from.

To see how solve time grows with problem size (writes `results/experiments.csv`, leaves your data and schedules alone):

```bash
python -m scheduler.experiments
```

Open the dashboard:

```bash
streamlit run app.py
```

## The dashboard

A navy-and-white dashboard with four pages:

- **Overview:** a one-line verdict, four headline numbers (total excess waiting, longest excess waiting, placement duration, students with excess waiting), a baseline-against-optimized table (green for an improvement, grey if unchanged), excess waiting per student, and the solver status. A collapsed **Optimization Details** section holds the technical side for the viva: the optimization priorities, the MILP model size, the stage-by-stage solver results and the step-by-step numbers.
- **Interview Schedule:** Gantt charts by student or by company, one schedule (optimized or baseline) at a time. Between a student's interviews, grey bars are the free buffer and red bars are excess waiting. The timetable table and its CSV download sit below.
- **Student Schedule:** pick a student to see their number of interviews, baseline and optimized excess waiting, and their optimized itinerary (the baseline itinerary is one click away).
- **Input Data:** company details, student summary and interview shortlists, each in its own collapsible section. Each shortlist is one interview that must be given a time slot.

The sidebar (Scenario Configuration) generates new data and solves it in one step with **Generate & Optimize**; it replaces the files in `data/` and `results/`. The page waits while the solver runs, for up to the time limit you set. If the scenario is invalid or the solver finds no timetable, the sidebar says why and the previous results stay on screen.

**Solver status.** The Overview never calls a result "Optimal" unless the solver proved it. If the solver ran out of time, the status reads "Feasible": the timetable is valid, but a better one may exist. A run from the sidebar saves `results/run_summary.json`, so the status and Optimization Details are still there after a reload. For schedules created from the command line there is no run record, and the page says so.

## How it works

**Data.** Time is cut into slots (15 minutes by default) and durations are whole slots. Each company has an interview length and a number of panels (rooms that interview in parallel). The generator keeps each company's total interview time under 80% of its capacity, so the sample problem is comfortably solvable.

**Baseline.** Interviews are taken in a fixed order (by company, then student ID). Each one goes at the earliest start that is free for both the student and the company, with the student's buffer respected.

**Optimizer.** The model decides when each interview starts.

- Variable: `x[i, t] = 1` if interview `i` starts at slot `t`, only for starts that fit inside the day and avoid lunch.
- Each interview starts exactly once.
- Student clash and buffer: an interview "occupies" its own slots plus the buffer after it, and a student can be occupied by at most one interview per slot.
- Company clash: at most `panels` interviews run per slot.
- Excess waiting: for each student and slot, a variable counts an idle slot that lies between two of their interviews and is not inside a buffer. It stays linear, with no big-M constants.
- Objective, in three stages (priorities 3 to 5). Each stage minimizes one thing and then locks its best value as a constraint before the next stage starts:
  1. minimize the longest individual excess wait;
  2. with that locked, minimize the total excess wait (skipped if stage 1 already reached 0);
  3. with both locked, minimize the end of the last interview. The day can never be shorter than the busiest company's workload or the busiest student's own interviews plus buffers, so that bound is added as a cut, and the stage stops at once if it is reached.

The first stage is warm-started from the baseline, and each later stage starts from the schedule the stage before it found. The time limit covers all three stages together (40% / 40% / 20%, and unused time rolls forward). The status is `Optimal` only if every stage was proven optimal, otherwise `Feasible (not proven optimal)`, and `optimize_schedule` also returns what each stage achieved. Either way the schedule is validated before it is returned.

**Metrics.** `metrics.py` is the single place where waiting is defined. The optimizer, the metrics and the dashboard charts all use the same definition, and a test checks the optimizer against an exhaustive search on small problems.

## Configuration

Everything is in `scheduler/config.py`. After changing data settings, regenerate the data and re-run both schedulers.

| Setting | Default | Meaning |
|---|---|---|
| `BUFFER_MINUTES` | 15 | Required gap between a student's interviews. Must be a multiple of `SLOT_MINUTES`. Use 0 to allow back-to-back interviews. |
| `SLOT_MINUTES` | 15 | Length of one time slot |
| `DAY_START`, `DAY_END` | 09:00, 18:00 | Interview window each day |
| `NUM_DAYS` | 1 | Number of interview days |
| `LUNCH_BREAK` | `None` | Optional window with no interviews, e.g. `("13:00", "14:00")` |
| `NUM_STUDENTS`, `NUM_COMPANIES`, `RANDOM_SEED` | 20, 5, 4 | Sample data size and reproducibility |
| `DURATION_CHOICES_SLOTS`, `PANEL_WEIGHTS` | 2 to 4 slots, 70% one panel and 30% two | How sample companies are generated |
| `SOLVER_TIME_LIMIT_SEC` | 60 | How long CBC may run in total, shared by the three optimization stages |
| `SOLVER_GAP_REL` | 0.0 | Accept a solution within this fraction of the best bound, e.g. 0.01 for 1% |

## Using your own data

Provide a CSV at `data/sample_data.csv` (or pass another path to `load_sample_data`) with one row per shortlist and these columns:

```
student_id, student_name, company_id, company_name, duration_slots, panels
```

`duration_slots` and `panels` must be at least 1 and the same on every row of a company. No student may be shortlisted by the same company twice, and there must be no empty values. The data is validated on load, including capacity checks that catch problems that cannot be scheduled at all.

## Tests

```bash
python -m pytest
```

The tests cover the clash, panel, buffer and lunch rules, hand-computed waiting-time cases, the link between the optimizer and the metrics, the priority order (compared with an exhaustive search of every valid schedule on small problems, including one where a lower total would cost a longer worst wait), the optimizer never doing worse than the baseline, and the dashboard's tables and charts. Tests that need the solver are skipped if PuLP is not installed.

## Limitations and ideas for later

- **Solve time grows quickly.** The sample data solves in seconds, but larger instances may not be proven optimal within 60 seconds. The result is still valid, and the stage list shows which priority was not proven. For larger instances raise `SOLVER_TIME_LIMIT_SEC` or set `SOLVER_GAP_REL`.
- **The day is shortened last.** The length of the day never takes priority over waiting, so the day can end later than it could if waiting did not matter. To cap it, add a finish-by constraint.
- **Fixed durations.** Every interview length is known up front, and companies have a fixed number of panels for the whole day.
- **Not modelled:** student preferences, company availability windows, travel time between venues, and last-minute changes to a published timetable.
