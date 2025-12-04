"""
TSci: Planner Agent

Plans experiments, model sweeps, and orchestrates the training pipeline.
Writes to tsci.jobs for scheduling.
"""

import duckdb
import os

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def create_job(
    con: duckdb.DuckDBPyConnection, job_name: str, job_type: str, config: dict = None
) -> int:
    """
    Create a new job in tsci.jobs.

    Returns:
        job_id of the created job
    """
    result = con.execute(
        "SELECT COALESCE(MAX(job_id), 0) + 1 FROM tsci.jobs"
    ).fetchone()
    job_id = result[0]

    con.execute(
        """
        INSERT INTO tsci.jobs (job_id, job_name, job_type, config_json)
        VALUES (?, ?, ?, ?)
    """,
        [job_id, job_name, job_type, str(config) if config else None],
    )

    return job_id


def plan_training_sweep(horizons: list = None) -> None:
    """
    Plan a training sweep across all horizons.
    """
    if horizons is None:
        horizons = ["1w", "1m", "3m", "6m", "12m"]

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}")

    for horizon in horizons:
        job_id = create_job(
            con,
            job_name=f"Training Sweep - {horizon}",
            job_type="training",
            config={"horizon": horizon, "symbol": "ZL"},
        )
        print(f"Created job {job_id} for horizon {horizon}")


if __name__ == "__main__":
    plan_training_sweep()
