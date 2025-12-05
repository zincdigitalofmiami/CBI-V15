from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import yaml
import logging
import subprocess
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LocalScheduler")

CONFIG_PATH = "config/schedulers/ingestion_schedules.yaml"


def run_job(script_path: str):
    """
    Executes a Python script as a subprocess.
    """
    logger.info(f"Starting job: {script_path}")
    try:
        # Use uv run to execute in the correct environment
        result = subprocess.run(
            ["uv", "run", "python", script_path], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"Job {script_path} completed successfully.")
            logger.info(result.stdout)
        else:
            logger.error(f"Job {script_path} failed with code {result.returncode}")
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"Failed to execute job {script_path}: {e}")


def load_schedule():
    """
    Loads the schedule from YAML and configures the scheduler.
    """
    scheduler = BlockingScheduler()

    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # This is a simplified loader. In a real scenario, we would parse the Cloud Scheduler config
    # and map it to APScheduler cron triggers. For now, we define a default mapping or
    # look for specific 'local_schedule' keys if we added them.

    # Example: Hardcoded for demonstration until we parse the full YAML
    # In the future, we will parse 'schedule: "0 16 * * *"' from the YAML.

    logger.info("Loading schedules...")

    # Job 1: ScrapeCreators News Buckets (The Firehose)
    # Runs every hour to fetch from the 25+ configured targets
    scheduler.add_job(
        run_job,
        CronTrigger(minute=0),  # Run at top of every hour
        args=["src/ingestion/scrapecreators/collect_news_buckets.py"],
        name="scrapecreators-news",
        id="scrapecreators_news_hourly",
        replace_existing=True
    )
    logger.info("✅ Added job: ScrapeCreators News Buckets (Hourly)")

    # Job 2: ScrapeCreators Trump/Social
    # Runs every hour
    scheduler.add_job(
        run_job,
        CronTrigger(minute=15),  # Run at :15 of every hour
        args=["src/ingestion/scrapecreators/buckets/collect_trump_truth_social.py"],
        name="scrapecreators-trump",
        id="scrapecreators_trump_hourly",
        replace_existing=True
    )
    logger.info("✅ Added job: ScrapeCreators Trump/Social (Hourly)")

    # Placeholder for Databento (commented out until script is ready)
    # scheduler.add_job(run_job, CronTrigger(hour=16, minute=0), args=["src/ingestion/databento/collect_daily.py"])

    logger.info(f"Scheduler configured with {len(scheduler.get_jobs())} jobs. Starting...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    # For testing, just print that it works
    logger.info("Local Scheduler initialized. (No jobs configured yet)")
    # load_schedule()
