#!/usr/bin/env python3
"""
Export BigQuery Tables to Parquet - Phase 1.3 (V2)
Uses BigQuery Python client for direct Parquet export.
PARALLEL, BULK exports for INSANE PACE backfill.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from google.cloud import bigquery
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install google-cloud-bigquery pandas pyarrow")
    exit(1)

PROJECT_ID = 'cbi-v15'
MANIFEST_FILE = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/bq_quick_manifest.json')
PARQUET_BASE = Path('/Volumes/Satechi Hub/ZL-Intelligence/parquet')
EXPORT_MANIFEST = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/parquet_manifest.json')

# Max parallel exports
MAX_WORKERS = 4  # Reduced to avoid BigQuery quota limits


def export_table_to_parquet(client: bigquery.Client, dataset: str, table: str, output_dir: Path) -> Dict:
    """Export a single BigQuery table to Parquet by querying and saving."""
    table_path = output_dir / dataset / table
    table_path.mkdir(parents=True, exist_ok=True)
    
    table_id = f"{PROJECT_ID}.{dataset}.{table}"
    parquet_file = table_path / f"{table}.parquet"
    
    try:
        # Query table and convert to DataFrame
        query = f"SELECT * FROM `{table_id}`"
        df = client.query(query).to_dataframe()
        
        # Save to Parquet with compression
        df.to_parquet(
            parquet_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Get file size and row count
        file_size = parquet_file.stat().st_size if parquet_file.exists() else 0
        row_count = len(df)
        
        return {
            'dataset': dataset,
            'table': table,
            'status': 'success',
            'output_path': str(parquet_file),
            'file_size_bytes': file_size,
            'row_count': row_count,
            'exported_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'dataset': dataset,
            'table': table,
            'status': 'error',
            'error': str(e),
            'output_path': str(parquet_file)
        }


def main():
    """Export all BigQuery tables to Parquet in parallel."""
    print("=" * 60)
    print("EXPORTING BIGQUERY TO PARQUET - Phase 1.3 (V2)")
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
    
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    
    # Export in parallel
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(export_table_to_parquet, client, t['dataset'], t['table'], PARQUET_BASE): t
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
                size_mb = result.get('file_size_bytes', 0) / (1024 * 1024)
                print(f"[{completed}/{len(tables_with_data)}] {table_info['dataset']}.{table_info['table']}: {status} ({size_mb:.1f} MB)")
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
        'total_size_mb': sum(r.get('file_size_bytes', 0) for r in results) / (1024 * 1024),
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
    print(f"Total size: {export_manifest['total_size_mb']:.1f} MB")
    print(f"Export manifest: {EXPORT_MANIFEST}")


if __name__ == '__main__':
    main()

