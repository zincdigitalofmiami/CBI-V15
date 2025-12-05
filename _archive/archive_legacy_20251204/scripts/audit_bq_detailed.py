#!/usr/bin/env python3
"""
Detailed BigQuery audit - writes comprehensive results to file
"""
import subprocess
import json
import sys
from pathlib import Path

PROJECT_ID = "cbi-v15"
OUTPUT_FILE = Path(__file__).parent.parent / "BQ_DETAILED_AUDIT.txt"

def run_bq_command(cmd):
    """Run bq command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("DETAILED BIGQUERY AUDIT - CBI-V15")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # 1. List all datasets with details
    output_lines.append("📊 DATASETS:")
    output_lines.append("-" * 80)
    stdout, stderr, rc = run_bq_command(f"command bq ls --project_id={PROJECT_ID} --format=prettyjson")
    if rc == 0 and stdout:
        try:
            datasets = json.loads(stdout)
            for ds in datasets:
                ds_id = ds.get('datasetId', 'unknown')
                location = ds.get('location', 'unknown')
                created = ds.get('creationTime', 'unknown')
                output_lines.append(f"  {ds_id}")
                output_lines.append(f"    Location: {location}")
                output_lines.append(f"    Created: {created}")
                output_lines.append("")
        except json.JSONDecodeError:
            output_lines.append(stdout)
    else:
        output_lines.append(f"  Error: {stderr}")
    output_lines.append("")
    
    # 2. For each dataset, list tables with details
    output_lines.append("=" * 80)
    output_lines.append("TABLES BY DATASET:")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Get dataset list
    stdout, _, rc = run_bq_command(f"command bq ls --project_id={PROJECT_ID} --format=csv")
    datasets = []
    if rc == 0 and stdout:
        lines = stdout.strip().split('\n')[1:]  # Skip header
        datasets = [line.split(',')[0].strip() for line in lines if line.strip() and not line.startswith('datasetId')]
    
    for dataset in datasets:
        if not dataset:
            continue
            
        output_lines.append(f"📁 Dataset: {dataset}")
        output_lines.append("-" * 80)
        
        # List tables
        stdout, stderr, rc = run_bq_command(
            f"command bq ls --project_id={PROJECT_ID} --dataset_id={dataset} --max_results=100 --format=prettyjson"
        )
        
        if rc == 0 and stdout:
            try:
                tables = json.loads(stdout)
                if isinstance(tables, list) and len(tables) > 0:
                    for table in tables:
                        table_id = table.get('tableReference', {}).get('tableId', 'unknown')
                        table_type = table.get('type', 'TABLE')
                        num_rows = table.get('numRows', 'unknown')
                        created = table.get('creationTime', 'unknown')
                        modified = table.get('lastModifiedTime', 'unknown')
                        
                        output_lines.append(f"  {table_type}: {table_id}")
                        if num_rows != 'unknown':
                            output_lines.append(f"    Rows: {num_rows:,}")
                        output_lines.append(f"    Created: {created}")
                        output_lines.append(f"    Modified: {modified}")
                        
                        # Try to get date range if there's a date column
                        try:
                            stdout_query, _, _ = run_bq_command(
                                f"command bq query --use_legacy_sql=false --project_id={PROJECT_ID} "
                                f"--format=csv --quiet "
                                f"'SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date "
                                f"FROM `{PROJECT_ID}.{dataset}.{table_id}` LIMIT 1'"
                            )
                            if stdout_query and 'min_date' in stdout_query:
                                parts = stdout_query.strip().split('\n')
                                if len(parts) > 1:
                                    data = parts[1].split(',')
                                    if len(data) >= 3:
                                        output_lines.append(f"    Date Range: {data[1]} to {data[2]}")
                        except:
                            pass
                        
                        output_lines.append("")
                else:
                    output_lines.append("  (no tables found)")
            except json.JSONDecodeError:
                # Fallback to simple list
                stdout_simple, _, _ = run_bq_command(
                    f"command bq ls --project_id={PROJECT_ID} --dataset_id={dataset} --max_results=100"
                )
                if stdout_simple:
                    lines = stdout_simple.strip().split('\n')[2:]  # Skip header
                    for line in lines:
                        if line.strip():
                            table_id = line.split()[0]
                            output_lines.append(f"  TABLE: {table_id}")
                else:
                    output_lines.append("  (no tables found)")
        else:
            output_lines.append(f"  Error accessing dataset: {stderr}")
        
        output_lines.append("")
    
    # 3. Summary
    output_lines.append("=" * 80)
    output_lines.append("SUMMARY:")
    output_lines.append("=" * 80)
    output_lines.append(f"  Datasets: {len(datasets)}")
    
    total_tables = 0
    for dataset in datasets:
        stdout, _, _ = run_bq_command(
            f"command bq ls --project_id={PROJECT_ID} --dataset_id={dataset} --format=csv"
        )
        if stdout:
            lines = stdout.strip().split('\n')
            table_count = len([l for l in lines[1:] if l.strip() and not l.startswith('datasetId')])
            total_tables += table_count
    
    output_lines.append(f"  Total Tables: {total_tables}")
    output_lines.append("")
    
    # Write results
    output_text = "\n".join(output_lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output_text)
    
    print(output_text)
    print(f"\n✅ Detailed audit written to: {OUTPUT_FILE}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())






