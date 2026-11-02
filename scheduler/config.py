"""
Central configuration for the Smart Placement Slot Scheduler.

Every tunable number lives here so experiments are repeatable: change a value,
regenerate the data, re-run the baseline and the optimizer.

Time model
----------
Time is cut into equal slots of SLOT_MINUTES minutes. Slots are numbered
0, 1, 2, ... across all days (a "global slot index"). Interview durations are
whole numbers of slots. An interview that starts at slot s and lasts d slots
occupies slots s, s+1, ..., s+d-1 and ends at the start of slot s+d.

Buffer
------
Zero waiting is unrealistic: a student needs time to move between rooms and
reset. BUFFER_MINUTES is therefore a REQUIRED gap between two interviews of the
same student on the same day (a hard rule for every schedule, baseline and
optimized alike). Only waiting BEYOND this buffer counts as "excess waiting",
and excess waiting is what the optimizer minimizes.

Both baseline.py and optimizer.py should use get_valid_start_slots() and
student_occupied_slots() so that they agree on when an interview may start and
when a student is considered busy.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
BASELINE_SCHEDULE_PATH = RESULTS_DIR / "baseline_schedule.csv"
OPTIMIZED_SCHEDULE_PATH = RESULTS_DIR / "optimized_schedule.csv"

# ---------------------------------------------------------------------------
# Time settings
# ---------------------------------------------------------------------------
SLOT_MINUTES = 15          # length of one slot
DAY_START = "09:00"        # first interview may start here (24h "HH:MM")
DAY_END = "18:00"          # last interview must finish by here
NUM_DAYS = 1               # number of interview days

# Optional lunch window with no interviews, e.g. ("13:00", "14:00").
# None means no lunch break. Whether lunch counts as "waiting" is decided in
# metrics.py, not here.
LUNCH_BREAK = None

# ---------------------------------------------------------------------------
# Scheduling rules
# ---------------------------------------------------------------------------
# Required buffer between two interviews of the same student on the same day
# (travel, rest, freshening up). Must be a multiple of SLOT_MINUTES.
#   * It is a HARD rule: back-to-back interviews are not allowed.
#   * Waiting up to the buffer is "free"; only the part of a gap that is LONGER
#     than the buffer is "excess waiting" and gets minimized.
# Use 0 to allow back-to-back interviews (then every gap counts as excess).
BUFFER_MINUTES = 15

# ---------------------------------------------------------------------------
# Sample data generation
# ---------------------------------------------------------------------------
RANDOM_SEED = 4                        # same seed -> same data every run
NUM_STUDENTS = 20
NUM_COMPANIES = 5
MIN_SHORTLISTS_PER_STUDENT = 1         # each student is shortlisted by at least this many companies
MAX_SHORTLISTS_PER_STUDENT = 3         # ... and at most this many
DURATION_CHOICES_SLOTS = [2, 3, 4]     # interview length in slots (30 / 45 / 60 min at 15-min slots)
PANEL_WEIGHTS = {1: 0.7, 2: 0.3}       # {panels: probability}; a panel = one parallel interview room

# A company's total interview time may use at most this share of its capacity
# (available slots x panels). Keeps the sample data comfortably solvable.
MAX_COMPANY_UTILIZATION = 0.8

# ---------------------------------------------------------------------------
# Solver settings (used by optimizer.py)
# ---------------------------------------------------------------------------
SOLVER_TIME_LIMIT_SEC = 60
SOLVER_GAP_REL = 0.0                   # 0.0 = prove optimality; e.g. 0.01 accepts a 1% gap
SOLVER_MSG = False                     # True prints CBC's log


# ---------------------------------------------------------------------------
# Derived values and helpers (no need to edit below this line)
# ---------------------------------------------------------------------------
def _to_minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


_DAY_MINUTES = _to_minutes(DAY_END) - _to_minutes(DAY_START)
if _DAY_MINUTES <= 0 or _DAY_MINUTES % SLOT_MINUTES != 0:
    raise ValueError(
        f"DAY_START to DAY_END ({_DAY_MINUTES} min) must be positive and a "
        f"multiple of SLOT_MINUTES ({SLOT_MINUTES})."
    )

SLOTS_PER_DAY = _DAY_MINUTES // SLOT_MINUTES
TOTAL_SLOTS = SLOTS_PER_DAY * NUM_DAYS

if BUFFER_MINUTES < 0 or BUFFER_MINUTES % SLOT_MINUTES != 0:
    raise ValueError(
        f"BUFFER_MINUTES ({BUFFER_MINUTES}) must be zero or a positive multiple "
        f"of SLOT_MINUTES ({SLOT_MINUTES})."
    )
BUFFER_SLOTS = BUFFER_MINUTES // SLOT_MINUTES


def get_blocked_slots() -> set[int]:
    """Global slot indices where no interview may take place (lunch break)."""
    if LUNCH_BREAK is None:
        return set()

    first = (_to_minutes(LUNCH_BREAK[0]) - _to_minutes(DAY_START)) // SLOT_MINUTES
    last_exclusive = math.ceil((_to_minutes(LUNCH_BREAK[1]) - _to_minutes(DAY_START)) / SLOT_MINUTES)
    if not 0 <= first < last_exclusive <= SLOTS_PER_DAY:
        raise ValueError("LUNCH_BREAK must lie inside the working day and end after it starts.")

    return {
        day * SLOTS_PER_DAY + s
        for day in range(NUM_DAYS)
        for s in range(first, last_exclusive)
    }


def available_slot_count() -> int:
    """Number of slots in which interviews can happen (all slots minus lunch)."""
    return TOTAL_SLOTS - len(get_blocked_slots())


def get_valid_start_slots(duration_slots: int) -> list[int]:
    """
    All global slots at which an interview of `duration_slots` may start.

    An interview must finish within the same day and must not overlap a blocked
    (lunch) slot.
    """
    blocked = get_blocked_slots()
    valid = []
    for start in range(TOTAL_SLOTS):
        if start % SLOTS_PER_DAY + duration_slots > SLOTS_PER_DAY:
            continue  # would run past the end of the day
        if any(s in blocked for s in range(start, start + duration_slots)):
            continue  # would overlap lunch
        valid.append(start)
    return valid


def rest_slots(start: int, duration_slots: int) -> range:
    """
    The buffer that follows an interview: the BUFFER_SLOTS slots right after it
    ends, clipped to the end of that day (the buffer never spills overnight).
    """
    end = start + duration_slots
    day_end = (start // SLOTS_PER_DAY + 1) * SLOTS_PER_DAY
    return range(end, min(end + BUFFER_SLOTS, day_end))


def student_occupied_slots(start: int, duration_slots: int) -> range:
    """
    Slots in which a student is 'taken' by an interview: the interview itself
    plus its buffer. Two of a student's interviews may not occupy the same slot,
    which is exactly the rule "no clash and at least BUFFER_SLOTS between them".
    """
    day_end = (start // SLOTS_PER_DAY + 1) * SLOTS_PER_DAY
    return range(start, min(start + duration_slots + BUFFER_SLOTS, day_end))


def slot_to_day(slot: int) -> int:
    """1-based day number of a global slot."""
    return slot // SLOTS_PER_DAY + 1


def slot_to_clock(slot: int, plus_slots: int = 0) -> str:
    """
    Clock time "HH:MM" at the start of `slot`, optionally moved forward by
    `plus_slots` slots. Use plus_slots=duration to get an interview's end time.
    """
    minutes = _to_minutes(DAY_START) + (slot % SLOTS_PER_DAY + plus_slots) * SLOT_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def interview_time_label(start_slot: int, duration_slots: int) -> str:
    """Readable label such as 'Day 1 09:00-09:45' for the dashboard."""
    return (
        f"Day {slot_to_day(start_slot)} "
        f"{slot_to_clock(start_slot)}-{slot_to_clock(start_slot, duration_slots)}"
    )
