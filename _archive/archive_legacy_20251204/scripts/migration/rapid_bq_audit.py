#!/usr/bin/env python3
"""
Rapid BigQuery Audit - Phase 1.1
Quick catalog of all tables across datasets for INSANE PACE backfill.
Minimal validation - speed first.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# BigQuery datasets to audit
DATASETS = [
    'raw',
    'staging', 
    'features',
    'training',
    'forecasts',
    'api',
    'reference',
    'ops'
]

PROJECT_ID = 'cbi-v15'
OUTPUT_DIR = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration')
MANIFEST_FILE = OUTPUT_DIR / 'bq_quick_manifest.json'


def get_table_info(dataset: str, table: str) -> Dict:
    """Get minimal table info: row count and date range."""
    query = f"""
    SELECT 
        COUNT(*) as row_count,
        MIN(date) as min_date,
        MAX(date) as max_date
    FROM `{PROJECT_ID}.{dataset}.{table}`
    """
    
    try:
        result = subprocess.run(
            ['bq', 'query', '--use_legacy_sql=false', '--format=json', '--max_rows=1', query],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                row = data[0]
                row_count = row.get('row_count', 0)
                # Convert to int if string
                if isinstance(row_count, str):
                    try:
                        row_count = int(row_count)
                    except (ValueError, TypeError):
                        row_count = 0
                return {
                    'dataset': dataset,
                    'table': table,
                    'row_count': int(row_count) if row_count else 0,
                    'min_date': str(row.get('min_date', '')),
                    'max_date': str(row.get('max_date', ''))
                }
    except Exception as e:
        print(f"Error querying {dataset}.{table}: {e}")
    
    return {
        'dataset': dataset,
        'table': table,
        'row_count': 0,
        'min_date': '',
        'max_date': ''
    }


def list_tables_in_dataset(dataset: str) -> List[str]:
    """List all tables in a dataset."""
    query = f"""
    SELECT table_name
    FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE'
    """
    
    try:
        result = subprocess.run(
            ['bq', 'query', '--use_legacy_sql=false', '--format=json', query],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return [row['table_name'] for row in data]
    except Exception as e:
        print(f"Error listing tables in {dataset}: {e}")
    
    return []


def audit_dataset(dataset: str) -> List[Dict]:
    """Audit a single dataset - parallel table queries."""
    print(f"Auditing dataset: {dataset}")
    tables = list_tables_in_dataset(dataset)
    
    if not tables:
        print(f"  No tables found in {dataset}")
        return []
    
    print(f"  Found {len(tables)} tables")
    
    # Parallel queries for speed
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(get_table_info, dataset, table): table 
            for table in tables
        }
        
        for future in as_completed(futures):
            table = futures[future]
            try:
                info = future.result()
                results.append(info)
                print(f"    {table}: {info['row_count']} rows")
            except Exception as e:
                print(f"    {table}: ERROR - {e}")
                results.append({
                    'dataset': dataset,
                    'table': table,
                    'row_count': 0,
                    'min_date': '',
                    'max_date': '',
                    'error': str(e)
                })
    
    return results


def main():
    """Run rapid audit across all datasets."""
    print("=" * 60)
    print("RAPID BIGQUERY AUDIT - Phase 1.1")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parallel dataset audits
    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(audit_dataset, ds): ds for ds in DATASETS}
        
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"ERROR auditing {dataset}: {e}")
    
    # Create manifest
    manifest = {
        'project_id': PROJECT_ID,
        'audit_timestamp': datetime.now().isoformat(),
        'total_tables': len(all_results),
        'total_rows': sum(r.get('row_count', 0) for r in all_results),
        'tables': all_results
    }
    
    # Save manifest
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total tables: {manifest['total_tables']}")
    print(f"Total rows: {manifest['total_rows']:,}")
    print(f"Manifest saved: {MANIFEST_FILE}")
    print()
    
    # Summary by dataset
    print("Summary by dataset:")
    for dataset in DATASETS:
        dataset_tables = [r for r in all_results if r['dataset'] == dataset]
        dataset_rows = sum(r.get('row_count', 0) for r in dataset_tables)
        print(f"  {dataset}: {len(dataset_tables)} tables, {dataset_rows:,} rows")


if __name__ == '__main__':
    main()

