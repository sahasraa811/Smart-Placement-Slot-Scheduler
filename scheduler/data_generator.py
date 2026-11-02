"""
Sample data generator for the Smart Placement Slot Scheduler.

The whole problem instance is stored in ONE csv (data/sample_data.csv) in long
format: one row per (student, company) shortlist.

    student_id, student_name, company_id, company_name, duration_slots, panels

* duration_slots - length of that company's interview, in slots (see config.py)
* panels         - how many interviews the company can run at the same time

Duration and panels belong to the company, so they repeat on every row of that
company. validate_data() checks they stay consistent.

Run from the project root:

    python -m scheduler.data_generator
    python -m scheduler.data_generator --students 30 --companies 6 --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd

from scheduler import config

COLUMNS = [
    "student_id",
    "student_name",
    "company_id",
    "company_name",
    "duration_slots",
    "panels",
]

# Made-up names, so no real company is implied.
COMPANY_NAMES = [
    "Nimbus Systems",
    "Orion Analytics",
    "Vertex Labs",
    "Quanta Works",
    "Helix Software",
    "Zenith Finance",
    "Apex Robotics",
    "Lumen Consulting",
    "Solace Networks",
    "Cobalt Dynamics",
    "Meridian Tech",
    "Aurora Data",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _make_companies(num_companies: int, rng: random.Random) -> list[dict]:
    width = max(1, len(str(num_companies)))
    panel_options = list(config.PANEL_WEIGHTS.keys())
    panel_probs = list(config.PANEL_WEIGHTS.values())

    companies = []
    for i in range(num_companies):
        base_name = COMPANY_NAMES[i % len(COMPANY_NAMES)]
        suffix = "" if i < len(COMPANY_NAMES) else f" {i // len(COMPANY_NAMES) + 1}"
        companies.append(
            {
                "company_id": f"C{i + 1:0{width}d}",
                "company_name": base_name + suffix,
                "duration_slots": rng.choice(config.DURATION_CHOICES_SLOTS),
                "panels": rng.choices(panel_options, weights=panel_probs, k=1)[0],
            }
        )
    return companies


def generate_sample_data(
    num_students: int = config.NUM_STUDENTS,
    num_companies: int = config.NUM_COMPANIES,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a random but reproducible problem instance (same seed -> same data).

    Each student is shortlisted by MIN..MAX_SHORTLISTS_PER_STUDENT companies.
    A company stops accepting shortlists once its total interview time reaches
    MAX_COMPANY_UTILIZATION of its capacity, so the instance stays solvable.
    """
    rng = random.Random(seed)
    companies = _make_companies(num_companies, rng)

    horizon = config.available_slot_count()
    capacity = {
        c["company_id"]: math.floor(config.MAX_COMPANY_UTILIZATION * horizon * c["panels"])
        for c in companies
    }
    used = {c["company_id"]: 0 for c in companies}

    width = max(2, len(str(num_students)))
    students = [(f"S{i:0{width}d}", f"Student {i:0{width}d}") for i in range(1, num_students + 1)]

    # Shuffle so that early student IDs do not always get first pick.
    order = students[:]
    rng.shuffle(order)

    rows = []
    for student_id, student_name in order:
        wanted = rng.randint(config.MIN_SHORTLISTS_PER_STUDENT, config.MAX_SHORTLISTS_PER_STUDENT)
        open_companies = [
            c for c in companies
            if used[c["company_id"]] + c["duration_slots"] <= capacity[c["company_id"]]
        ]
        if not open_companies:
            raise ValueError(
                "No company has capacity left for another shortlist. Reduce "
                "NUM_STUDENTS or MAX_SHORTLISTS_PER_STUDENT, or add companies or days."
            )

        for c in rng.sample(open_companies, min(wanted, len(open_companies))):
            used[c["company_id"]] += c["duration_slots"]
            rows.append(
                {
                    "student_id": student_id,
                    "student_name": student_name,
                    "company_id": c["company_id"],
                    "company_name": c["company_name"],
                    "duration_slots": c["duration_slots"],
                    "panels": c["panels"],
                }
            )

    df = (
        pd.DataFrame(rows, columns=COLUMNS)
        .sort_values(["student_id", "company_id"])
        .reset_index(drop=True)
    )
    validate_data(df)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> None:
    """
    Raise ValueError if the data is malformed or obviously unschedulable.

    The capacity checks are NECESSARY conditions only: passing them does not
    guarantee a clash-free timetable exists, but failing them guarantees none does.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("The data has no rows.")
    if df[COLUMNS].isna().any().any():
        raise ValueError("The data contains empty values.")
    if df.duplicated(["student_id", "company_id"]).any():
        raise ValueError("A student is shortlisted by the same company more than once.")

    for col in ("duration_slots", "panels"):
        if (df[col] < 1).any():
            raise ValueError(f"{col} must be at least 1.")
        if (df.groupby("company_id")[col].nunique() > 1).any():
            raise ValueError(f"{col} differs between rows of the same company.")

    # Every interview length needs at least one legal start time.
    for duration in sorted(df["duration_slots"].unique()):
        if not config.get_valid_start_slots(int(duration)):
            raise ValueError(
                f"A {duration}-slot interview does not fit in the working day / lunch window."
            )

    horizon = config.available_slot_count()

    # Company capacity: total interview time must fit in horizon x panels.
    load = df.groupby("company_id").agg(total=("duration_slots", "sum"), panels=("panels", "first"))
    overloaded = load[load["total"] > horizon * load["panels"]]
    if not overloaded.empty:
        raise ValueError(
            f"Companies need more interview time than is available: {list(overloaded.index)}"
        )

    # Student capacity: interviews plus the required buffers must fit in the horizon.
    per_student = df.groupby("student_id")["duration_slots"].agg(["sum", "count"])
    needed = per_student["sum"] + config.BUFFER_SLOTS * (per_student["count"] - 1)
    too_busy = needed[needed > horizon]
    if not too_busy.empty:
        raise ValueError(
            f"Students with more interview time than available: {list(too_busy.index)}"
        )


# ---------------------------------------------------------------------------
# Save / load / summaries
# ---------------------------------------------------------------------------
def save_sample_data(df: pd.DataFrame, path: Path | str = config.SAMPLE_DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_sample_data(path: Path | str = config.SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Read the csv and validate it. Used by baseline.py, optimizer.py and app.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with: python -m scheduler.data_generator"
        )
    df = pd.read_csv(path)
    validate_data(df)
    return df


def get_students(df: pd.DataFrame) -> pd.DataFrame:
    """One row per student: how many interviews and how many slots they need in total."""
    return (
        df.groupby(["student_id", "student_name"], as_index=False)
        .agg(num_interviews=("company_id", "count"), total_interview_slots=("duration_slots", "sum"))
        .sort_values("student_id")
        .reset_index(drop=True)
    )


def get_companies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per company: interview length, panels and number of shortlisted students."""
    return (
        df.groupby(["company_id", "company_name"], as_index=False)
        .agg(
            duration_slots=("duration_slots", "first"),
            panels=("panels", "first"),
            num_candidates=("student_id", "count"),
        )
        .sort_values("company_id")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample placement-interview data.")
    parser.add_argument("--students", type=int, default=config.NUM_STUDENTS)
    parser.add_argument("--companies", type=int, default=config.NUM_COMPANIES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=config.SAMPLE_DATA_PATH)
    args = parser.parse_args()

    df = generate_sample_data(args.students, args.companies, args.seed)
    path = save_sample_data(df, args.output)

    companies = get_companies(df)
    companies["slots_needed"] = companies["duration_slots"] * companies["num_candidates"]
    companies["slots_available"] = config.available_slot_count() * companies["panels"]

    print(f"Saved {len(df)} shortlists ({df['student_id'].nunique()} students, "
          f"{df['company_id'].nunique()} companies) to {path}")
    print(f"Time grid: {config.NUM_DAYS} day(s), {config.SLOTS_PER_DAY} slots/day "
          f"of {config.SLOT_MINUTES} min, seed={args.seed}\n")
    print(companies.to_string(index=False))


if __name__ == "__main__":
    main()
