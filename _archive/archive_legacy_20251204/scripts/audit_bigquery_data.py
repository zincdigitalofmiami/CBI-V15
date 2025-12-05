#!/usr/bin/env python3
"""
Audit BigQuery datasets and tables in cbi-v15 project
Writes results to BQ_AUDIT_RESULTS.txt for review
"""
import subprocess
import json
import sys
from pathlib import Path

PROJECT_ID = "cbi-v15"
OUTPUT_FILE = Path(__file__).parent.parent / "BQ_AUDIT_RESULTS.txt"

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
    output_lines.append("BIGQUERY DATA AUDIT - CBI-V15")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # 1. List all datasets
    output_lines.append("📊 DATASETS IN cbi-v15:")
    output_lines.append("-" * 80)
    stdout, stderr, rc = run_bq_command(f"bq ls --project_id={PROJECT_ID} --format=prettyjson")
    if rc == 0 and stdout:
        try:
            datasets = json.loads(stdout)
            for ds in datasets:
                ds_id = ds.get('datasetId', 'unknown')
                location = ds.get('location', 'unknown')
                output_lines.append(f"  Dataset: {ds_id}")
                output_lines.append(f"    Location: {location}")
                output_lines.append("")
        except json.JSONDecodeError:
            output_lines.append(stdout)
    else:
        output_lines.append(f"  Error: {stderr}")
    output_lines.append("")
    
    # 2. For each dataset, list tables
    output_lines.append("=" * 80)
    output_lines.append("TABLES BY DATASET:")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Common dataset names to check
    common_datasets = ['external', 'raw', 'staging', 'models', 'neural', 'forecasting_data_warehouse']
    
    for dataset in common_datasets:
        output_lines.append(f"📁 Dataset: {dataset}")
        output_lines.append("-" * 80)
        stdout, stderr, rc = run_bq_command(f"bq ls --project_id={PROJECT_ID} --dataset_id={dataset} --max_results=100")
        if rc == 0:
            if stdout.strip():
                # Parse table list
                lines = stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            table_id = parts[0]
                            output_lines.append(f"  Table: {dataset}.{table_id}")
                            
                            # Get table info
                            stdout_info, stderr_info, rc_info = run_bq_command(
                                f"bq show --project_id={PROJECT_ID} --format=prettyjson {PROJECT_ID}:{dataset}.{table_id}"
                            )
                            if rc_info == 0:
                                try:
                                    info = json.loads(stdout_info)
                                    num_rows = info.get('numRows', 'unknown')
                                    created = info.get('creationTime', 'unknown')
                                    modified = info.get('lastModifiedTime', 'unknown')
                                    output_lines.append(f"    Rows: {num_rows}")
                                    output_lines.append(f"    Created: {created}")
                                    output_lines.append(f"    Modified: {modified}")
                                    
                                    # Try to get date range if there's a date column
                                    stdout_query, _, _ = run_bq_command(
                                        f"bq query --use_legacy_sql=false --project_id={PROJECT_ID} "
                                        f"'SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date "
                                        f"FROM `{PROJECT_ID}.{dataset}.{table_id}` LIMIT 1'"
                                    )
                                    if stdout_query and 'min_date' in stdout_query:
                                        output_lines.append(f"    Date Range: {stdout_query.strip()}")
                                except:
                                    pass
                            output_lines.append("")
            else:
                output_lines.append(f"  (no tables found or dataset doesn't exist)")
        else:
            output_lines.append(f"  Error accessing dataset: {stderr}")
        output_lines.append("")
    
    # 3. Check for key vendor tables
    output_lines.append("=" * 80)
    output_lines.append("KEY VENDOR TABLES CHECK:")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    vendor_checks = [
        ("Databento", ["databento", "dbnt"]),
        ("FRED", ["fred", "economic"]),
        ("ScrapeCreators", ["scrapecreators", "trump", "news"]),
        ("USDA", ["usda", "export"]),
        ("CFTC", ["cftc", "cot"]),
    ]
    
    for vendor_name, search_terms in vendor_checks:
        output_lines.append(f"🔍 {vendor_name}:")
        found = False
        for term in search_terms:
            stdout, _, rc = run_bq_command(
                f"bq query --use_legacy_sql=false --project_id={PROJECT_ID} "
                f"'SELECT table_schema, table_name FROM `{PROJECT_ID.upper()}.INFORMATION_SCHEMA.TABLES` "
                f"WHERE LOWER(table_name) LIKE \\'%{term}%\\' ORDER BY table_schema, table_name LIMIT 10'"
            )
            if rc == 0 and stdout and 'table_name' in stdout:
                output_lines.append(f"  Found tables matching '{term}':")
                output_lines.append(stdout)
                found = True
        if not found:
            output_lines.append(f"  No tables found for {vendor_name}")
        output_lines.append("")
    
    # Write results
    output_text = "\n".join(output_lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output_text)
    
    print(output_text)
    print(f"\n✅ Full audit written to: {OUTPUT_FILE}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())






