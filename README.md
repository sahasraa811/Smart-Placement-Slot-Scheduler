# Smart Placement Slot Scheduler

## About the project

We built the Smart Placement Slot Scheduler to make campus placement interview scheduling more organized. During placements, a student may be shortlisted by multiple companies. If interview timings are not planned properly, students can end up with clashes or spend a long time waiting between interviews. At the same time, each company has a limited number of interview panels, so it cannot interview all shortlisted students at once.

Our project takes the student shortlists, interview durations and panel availability as input and creates a timetable that follows these requirements. We first create a simple first-come-first-served (FCFS) timetable. Then we use an optimization model to try to reduce students' excess waiting time and, after that, the overall duration of the placement day. We display both schedules and their metrics in a Streamlit dashboard so that the difference can be seen clearly.

## What the project does

1. **Generates input data:** creates sample students, companies, interview durations, panel counts and student-company shortlists.
2. **Creates a baseline timetable:** schedules interviews in a fixed order, placing each interview in the earliest slot that is available for both the student and the company. This gives us a reference schedule before optimization.
3. **Improves the starting timetable:** uses a heuristic/local-search step to try to improve the baseline before passing it to the optimizer.
4. **Optimizes the timetable:** formulates the scheduling problem as a Mixed-Integer Linear Programming (MILP) model using PuLP and the CBC solver.
5. **Checks the constraints:** makes sure that students do not have overlapping interviews, companies do not exceed their available panels, and the required buffer between a student's interviews is respected.
6. **Compares the results:** calculates waiting-time and schedule-duration metrics for the baseline and optimized timetables.
7. **Shows the results:** provides charts, tables, student-wise schedules and input-data summaries through a Streamlit web application.

## Why do we use a buffer?

We do not treat every minute between two interviews as avoidable waiting. Students may need some time to move between interview rooms or prepare for the next interview. For this reason, the project has a required **15-minute buffer** between two interviews of the same student on the same day by default.

The buffer is a hard scheduling rule. The optimizer cannot remove it just to reduce the waiting-time metric. Only the time **beyond** the required buffer is counted as excess waiting.

For example, if one interview ends at 09:30 and the next begins at 10:15, the gap is 45 minutes. After allowing the 15-minute buffer, the student's excess waiting time is 30 minutes. For students with multiple interviews, we calculate this for each pair of consecutive interviews and add the excess gaps.

Time before a student's first interview and after their last interview is not counted as waiting. The calculation is done separately for each day, and a configured lunch break is excluded from excess waiting.

## Optimization priorities

We do not combine all objectives into one weighted score. Instead, the optimizer handles them in a fixed order. It first solves one objective, locks the best value it found, and then moves to the next objective.

| Order | Requirement / objective | Type |
|---|---|---|
| 1 | No overlapping interviews for a student | Hard constraint |
| 2 | A company cannot conduct more simultaneous interviews than its number of panels | Hard constraint |
| 3 | Maintain the required buffer between a student's interviews | Hard constraint |
| 4 | Minimize the maximum excess waiting time of any student | First optimization objective |
| 5 | Minimize the total excess waiting time of all students | Second optimization objective |
| 6 | Minimize the placement-day duration (the time at which the last interview finishes) | Third optimization objective |

The order matters. We first try to improve the worst individual waiting time. We then reduce total excess waiting without making that first objective worse. Finally, we try to finish the placement day earlier without worsening the waiting-time objectives.

## Technologies used

- **Python** – implementation of the scheduling logic and data processing.
- **Pandas** – reading, processing and writing the input data, schedules and result tables.
- **PuLP** – building the Mixed-Integer Linear Programming model.
- **CBC** – solving the optimization model.
- **Streamlit** – building the interactive dashboard.
- **Matplotlib** – drawing the timetable and comparison charts.
- **Pytest** – testing the scheduling rules, metrics, optimizer and dashboard helper functions.

## Sample result

The repository includes sample data with **20 students, 5 companies and 37 shortlists**. The saved schedules in `results/` give the following comparison:

| Metric | Baseline schedule | Optimized schedule |
|---|---:|---:|
| Longest excess waiting time | 210 minutes | 0 minutes |
| Total excess waiting time | 645 minutes | 0 minutes |
| Average excess waiting per student | 32.25 minutes | 0 minutes |
| Students with excess waiting | 9 | 0 |
| Placement-day duration | 420 minutes | 420 minutes |
| Last interview finishes | 16:00 | 16:00 |

For this particular sample, the optimizer found a timetable with no excess waiting while keeping the placement-day duration the same as the baseline. This is a result for the bundled sample data, not a guarantee that every possible input will have zero waiting or the same finishing time. Larger or more constrained inputs may produce a different result, and the solver may need more time.

## Project folder structure

```text
smart-placement-slot-scheduler/
├── data/
│   └── sample_data.csv              # Sample student-company shortlist data
├── results/
│   ├── baseline_schedule.csv        # FCFS timetable
│   └── optimized_schedule.csv       # Optimized timetable
├── scheduler/
│   ├── __init__.py
│   ├── config.py                    # Main settings: time slots, buffer, days and solver limits
│   ├── data_generator.py            # Generates, validates, saves and loads input data
│   ├── baseline.py                  # Creates the FCFS baseline timetable
│   ├── heuristic.py                 # Tries to improve the starting timetable
│   ├── optimizer.py                 # MILP model and CBC optimization process
│   ├── metrics.py                   # Waiting-time calculations and schedule validation
│   └── experiments.py               # Runs scalability experiments
├── tests/
│   ├── test_clashes.py              # Student clashes, company panels and buffer rules
│   ├── test_waiting_time.py          # Waiting-time calculations
│   ├── test_priorities.py            # Checks the optimization priority order
│   ├── test_start_and_fallback.py   # Checks the starting schedule and fallback behaviour
│   └── test_dashboard.py             # Tests dashboard helper functions and charts
├── .streamlit/
│   └── config.toml                  # Streamlit theme settings
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest configuration
├── .gitignore
└── README.md
```

Files such as `results/run_summary.json` and `results/experiments.csv` are created when the corresponding dashboard run or experiment is performed. They are not included in the current project ZIP by default.

## Setup and installation

### 1. Install Python

Install **Python 3.10 or newer**. Make sure Python is available from your terminal. You can check the installation with:

```bash
python --version
```

### 2. Open the project folder

Open the extracted `smart-placement-slot-scheduler` folder in VS Code (or another editor). Run the following commands from this project folder, where `app.py` and `requirements.txt` are present.

### 3. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, you can activate it from Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file includes `pulp[cbc]`, which installs PuLP and its CBC solver dependency. If installation completes successfully, no separate solver setup should normally be required.

## Running the project

### Run the dashboard

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will show a local address in the terminal and usually open the dashboard in your browser. If it does not open automatically, copy the local address shown in the terminal into your browser.

### Dashboard pages

The dashboard has four main pages:

- **Overview:** shows the main comparison between the baseline and optimized schedules, including total excess waiting, longest excess waiting, placement-day duration and the number of students with excess waiting. It also includes the student-wise waiting chart, validation checks and solver status. The Optimization Details section provides additional information about the MILP model and the stages of optimization when a run summary is available.
- **Interview Schedule:** displays the timetable as a Gantt-style chart. You can view it by student or by company and switch between the baseline and optimized schedules. The timetable can also be viewed as a table and downloaded as a CSV file.
- **Student Schedule:** lets you select a student and view their interviews, baseline waiting time, optimized waiting time and interview itinerary.
- **Input Data:** displays company details, a student summary and the student-company interview shortlists used by the scheduler.

### Generate and optimize a new scenario

The sidebar contains **Scenario Configuration**. You can change the number of students, number of companies, random seed and solver time limit. Select **Generate & Optimize** to generate new sample data, create a baseline schedule and run the optimizer. The generated data and schedules replace the corresponding CSV files in `data/` and `results/`.

If the new scenario cannot be scheduled or an error occurs, the dashboard displays an error message. The previous results are kept on screen rather than being replaced with an invalid timetable.

## Running the individual modules

You can also run the data generation, baseline scheduling and optimization separately from the terminal. Run these commands from the project root.

### Generate sample data

```bash
python -m scheduler.data_generator
```

To choose a different number of students, companies or random seed:

```bash
python -m scheduler.data_generator --students 30 --companies 6 --seed 1
```

The generated data is saved to `data/sample_data.csv` by default. The same seed and settings produce the same sample data.

### Generate the baseline schedule

```bash
python -m scheduler.baseline
```

This reads `data/sample_data.csv` and saves the FCFS timetable to `results/baseline_schedule.csv`.

### Run the optimizer

```bash
python -m scheduler.optimizer
```

This reads the sample data, uses the baseline as a starting timetable, runs the optimization process and saves the result to `results/optimized_schedule.csv`.

**Important:** If you generate new input data, run the baseline and optimizer again. Otherwise, the saved schedules may belong to the previous input data.

### Run scalability experiments

```bash
python -m scheduler.experiments
```

This runs the scheduler on several larger sample scenarios and writes the timing and result details to `results/experiments.csv`. It does not replace the saved input data or the baseline and optimized schedule CSV files.

You can change the time limit and seed for the experiments:

```bash
python -m scheduler.experiments --limit 120 --seed 4
```

## How the scheduling model works

### 1. Input data

The input file `data/sample_data.csv` stores one row for every student-company shortlist. It contains these columns:

```text
student_id, student_name, company_id, company_name, duration_slots, panels
```

- `student_id` and `student_name` identify the student.
- `company_id` and `company_name` identify the company.
- `duration_slots` is the interview duration in slots. With the default 15-minute slot size, 2 slots mean 30 minutes, 3 mean 45 minutes and 4 mean 60 minutes.
- `panels` is the number of interviews that the company can conduct at the same time. For example, 2 panels mean the company can run up to 2 interviews in parallel.

The duration and panel count belong to a company, so they are repeated across that company's rows. The data loader validates the input, including checking that these values are consistent.

### 2. Time slots

The day is divided into equal time slots. By default, the interview window is 09:00 to 18:00 and each slot is 15 minutes long. Interview durations are represented as whole numbers of slots. The project can also be configured for multiple days and an optional lunch break.

### 3. Baseline scheduling

The baseline scheduler processes the shortlist rows in a fixed order (company and then student ID). For each interview, it searches for the earliest valid slot where:

- the company has an available panel for the full interview duration; and
- the student is free for the interview and the required buffer afterwards.

This creates a valid reference timetable, but the baseline does not directly optimize the waiting-time objectives.

### 4. Heuristic starting schedule

Before the MILP stages, the optimizer can use a local-search heuristic to try to improve the baseline timetable. This gives the solver a stronger starting point when an improvement is found. The baseline is still retained as a valid starting option, and the optimizer validates the schedule before returning it.

### 5. MILP optimization

The optimizer uses binary decision variables to represent whether a particular interview starts at a particular valid time slot. The model includes constraints to ensure that:

- every interview is assigned exactly one start time;
- a student cannot have overlapping interviews or violate the required buffer;
- a company cannot conduct more simultaneous interviews than its panel count; and
- interviews fit within the configured working day and do not overlap a blocked lunch period.

The optimization objectives are solved in stages: first the maximum individual excess waiting time, then total excess waiting time, and finally the end time of the last interview. The best value from each completed stage is added as a constraint before the next stage, preserving the priority order.

The total solver time limit is shared across the stages. A run is labelled **Optimal** only when the solver proves the stages optimal. If it finds a valid schedule but cannot prove optimality before the time limit, the result is labelled **Feasible**. A feasible schedule is still checked against the scheduling constraints; it simply may not be the best possible schedule.

### 6. Metrics and validation

The `scheduler/metrics.py` file contains the common waiting-time calculations and schedule validation functions. The same definition of excess waiting is used when comparing the baseline and optimized schedules. The project also checks that the schedules satisfy the student, company-panel and buffer rules.

## Configuration

The main settings are in `scheduler/config.py`. Change them there when you want to experiment with a different scheduling setup.

| Setting | Default | Meaning |
|---|---:|---|
| `SLOT_MINUTES` | 15 | Duration of one time slot, in minutes |
| `DAY_START` | `09:00` | Start of the interview day |
| `DAY_END` | `18:00` | End of the interview day; interviews must finish by this time |
| `NUM_DAYS` | 1 | Number of interview days |
| `LUNCH_BREAK` | `None` | Optional period during which interviews are not allowed |
| `BUFFER_MINUTES` | 15 | Required gap between interviews of the same student on the same day |
| `NUM_STUDENTS` | 20 | Default number of sample students |
| `NUM_COMPANIES` | 5 | Default number of sample companies |
| `RANDOM_SEED` | 4 | Seed used to generate repeatable sample data |
| `SOLVER_TIME_LIMIT_SEC` | 60 | Total time limit shared by the optimization stages |
| `SOLVER_GAP_REL` | 0.0 | Relative optimality gap accepted by CBC; 0.0 asks the solver to prove optimality |

After changing the time settings or the sample-data settings, regenerate the data and run the baseline and optimizer again so that all saved files match the same configuration.

## Using your own input data

You can replace `data/sample_data.csv` with your own CSV file, keeping the same column names:

```text
student_id, student_name, company_id, company_name, duration_slots, panels
```

Use one row for each student-company shortlist. `duration_slots` and `panels` must be positive and must remain the same for every row belonging to the same company. A student should not appear more than once for the same company. The input is validated when it is loaded. Make sure the data fits the configured day length, number of days, slot size and panel capacity.

## Running the tests

The project uses Pytest. From the project root, run:

```bash
python -m pytest -v
```

The tests cover student and company clashes, buffer rules, waiting-time calculations, optimization priorities, starting-schedule behaviour and dashboard helper functions. Some optimizer-related tests require PuLP and CBC to be installed.

## Current limitations

- **Solver time:** the optimization problem can become harder as the number of students and interviews increases. For larger scenarios, the solver may find a valid schedule without proving that it is optimal within the configured time limit.
- **Fixed interview durations:** the current input assumes that each company's interview duration is known and fixed.
- **Fixed panel availability:** each company has a fixed number of panels; changing panel availability over the day is not currently represented.
- **No student preferences:** the current model does not include preferred interview times or other student-specific preferences.
- **No travel-time model:** the buffer is a common fixed gap. Different travel times between venues are not represented separately.
- **No last-minute updates:** the project creates a timetable from the supplied input; it does not currently handle live changes such as cancellations or sudden interview rescheduling.

## Possible future improvements

Some extensions that could make the scheduler more realistic include adding company-specific availability windows, student time preferences, venue-wise travel times, variable panel availability and a way to update the timetable when placement schedules change.
