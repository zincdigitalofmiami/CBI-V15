# Orchestration

Master orchestrator scripts that coordinate multiple ingestion jobs.

## Scripts

- `collect_all_buckets.py` - Runs all bucket-level news collectors in sequence

## Usage

```bash
python collect_all_buckets.py
```

## Notes

Individual source collectors are in their respective folders (DataBento, FRED, etc.).
This folder contains scripts that coordinate multiple collectors.

