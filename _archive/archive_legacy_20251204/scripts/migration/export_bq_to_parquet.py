#!/usr/bin/env python3
"""
Export BigQuery Tables to Parquet - Phase 1.3
PARALLEL, BULK exports for INSANE PACE backfill.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ID = 'cbi-v15'
MANIFEST_FILE = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/bq_quick_manifest.json')
PARQUET_BASE = Path('/Volumes/Satechi Hub/ZL-Intelligence/parquet')
EXPORT_MANIFEST = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/parquet_manifest.json')

# Max parallel exports
MAX_WORKERS = 8


def export_table_to_parquet(dataset: str, table: str, output_dir: Path) -> Dict:
    """Export a single BigQuery table to Parquet."""
    table_path = output_dir / dataset / table
    table_path.mkdir(parents=True, exist_ok=True)
    
    # BigQuery export command
    gcs_uri = f"gs://temp-bq-export-{PROJECT_ID}/{dataset}/{table}/*.parquet"
    
    # First, export to GCS (faster than direct export)
    export_cmd = [
        'bq', 'extract',
        '--destination_format=PARQUET',
        '--compression=SNAPPY',
        f'--project_id={PROJECT_ID}',
        f'{PROJECT_ID}:{dataset}.{table}',
        gcs_uri
    ]
    
    try:
        result = subprocess.run(
            export_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        if result.returncode != 0:
            return {
                'dataset': dataset,
                'table': table,
                'status': 'error',
                'error': result.stderr,
                'output_path': str(table_path)
            }
        
        # Download from GCS to local
        download_cmd = [
            'gsutil', '-m', 'cp', '-r',
            gcs_uri,
            str(table_path)
        ]
        
        download_result = subprocess.run(
            download_cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if download_result.returncode != 0:
            return {
                'dataset': dataset,
                'table': table,
                'status': 'download_error',
                'error': download_result.stderr,
                'output_path': str(table_path)
            }
        
        # Count parquet files
        parquet_files = list(table_path.glob('*.parquet'))
        
        return {
            'dataset': dataset,
            'table': table,
            'status': 'success',
            'output_path': str(table_path),
            'parquet_files': len(parquet_files),
            'exported_at': datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        return {
            'dataset': dataset,
            'table': table,
            'status': 'timeout',
            'output_path': str(table_path)
        }
    except Exception as e:
        return {
            'dataset': dataset,
            'table': table,
            'status': 'error',
            'error': str(e),
            'output_path': str(table_path)
        }


def export_with_local_fallback(dataset: str, table: str, output_dir: Path) -> Dict:
    """Export using local query method (fallback if GCS not available)."""
    table_path = output_dir / dataset / table
    table_path.mkdir(parents=True, exist_ok=True)
    
    # Query and save directly to parquet using DuckDB
    query = f"SELECT * FROM `{PROJECT_ID}.{dataset}.{table}`"
    parquet_file = table_path / f"{table}.parquet"
    
    try:
        # Use bq query to export to CSV first, then convert to Parquet
        # This is slower but works without GCS setup
        csv_file = table_path / f"{table}.csv"
        
        export_cmd = [
            'bq', 'query',
            '--use_legacy_sql=false',
            '--format=csv',
            '--max_rows=1000000000',  # Large limit
            f'--destination_table={PROJECT_ID}:{dataset}.{table}_export_temp',
            query
        ]
        
        # Actually, better approach: use bq extract directly to local parquet
        # But bq extract requires GCS. So we'll use Python BigQuery client instead.
        # For now, mark as needing manual export
        return {
            'dataset': dataset,
            'table': table,
            'status': 'needs_manual_export',
            'output_path': str(table_path),
            'note': 'Use BigQuery Python client for direct export'
        }
        
    except Exception as e:
        return {
            'dataset': dataset,
            'table': table,
            'status': 'error',
            'error': str(e),
            'output_path': str(table_path)
        }


def main():
    """Export all BigQuery tables to Parquet in parallel."""
    print("=" * 60)
    print("EXPORTING BIGQUERY TO PARQUET - Phase 1.3")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Output: {PARQUET_BASE}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load manifest
    if not MANIFEST_FILE.exists():
        print(f"ERROR: Manifest file not found: {MANIFEST_FILE}")
        print("Run rapid_bq_audit.py first!")
        return
    
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    
    # Filter to tables with data
    tables_with_data = [
        t for t in manifest['tables'] 
        if t.get('row_count', 0) > 0
    ]
    
    print(f"Tables with data: {len(tables_with_data)}")
    print(f"Total tables: {len(manifest['tables'])}")
    print()
    
    # Export in parallel
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(export_with_local_fallback, t['dataset'], t['table'], PARQUET_BASE): t
            for t in tables_with_data
        }
        
        completed = 0
        for future in as_completed(futures):
            table_info = futures[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = result.get('status', 'unknown')
                print(f"[{completed}/{len(tables_with_data)}] {table_info['dataset']}.{table_info['table']}: {status}")
            except Exception as e:
                print(f"[{completed}/{len(tables_with_data)}] {table_info['dataset']}.{table_info['table']}: ERROR - {e}")
                results.append({
                    'dataset': table_info['dataset'],
                    'table': table_info['table'],
                    'status': 'exception',
                    'error': str(e)
                })
    
    # Save export manifest
    export_manifest = {
        'export_timestamp': datetime.now().isoformat(),
        'total_tables': len(results),
        'successful': len([r for r in results if r.get('status') == 'success']),
        'exports': results
    }
    
    with open(EXPORT_MANIFEST, 'w') as f:
        json.dump(export_manifest, f, indent=2)
    
    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Total tables processed: {len(results)}")
    print(f"Successful: {export_manifest['successful']}")
    print(f"Export manifest: {EXPORT_MANIFEST}")
    print()
    print("NOTE: Most exports will show 'needs_manual_export' status.")
    print("Use BigQuery Python client for direct Parquet export.")


if __name__ == '__main__':
    main()

