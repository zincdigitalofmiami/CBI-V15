# Scripts Directory

Operational logic for the CBI-V15 pipeline.

## Structure
- `setup/`: Environment checks, extension installation.
- `ingestion/`: Fetchers for Databento, FRED, etc.
- `schedulers/`: Cron triggers and orchestrators.
- `maintenance/`: Database vacuum, backup, cleanup.

**Rule:** No loose scripts in root. Categorize everything.

