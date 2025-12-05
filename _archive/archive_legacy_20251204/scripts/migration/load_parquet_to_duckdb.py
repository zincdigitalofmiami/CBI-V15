#!/usr/bin/env python3
"""
Load Parquet into DuckDB - Phase 1.4
Uses COPY statements for fast bulk loading.
"""

import duckdb
import json
from pathlib import Path
from typing import Dict, List

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')
PARQUET_BASE = Path('/Volumes/Satechi Hub/ZL-Intelligence/parquet')
EXPORT_MANIFEST = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/parquet_manifest.json')
LOAD_MANIFEST = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/duckdb_load_manifest.json')


def get_table_schema_from_parquet(conn: duckdb.DuckDBPyConnection, parquet_file: Path) -> str:
    """Infer table schema from Parquet file."""
    try:
        # Read schema from Parquet
        result = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_file}')").fetchall()
        return result
    except Exception as e:
        print(f"    ⚠ Could not infer schema: {e}")
        return None


def load_table_to_duckdb(conn: duckdb.DuckDBPyConnection, dataset: str, table: str, parquet_file: Path) -> Dict:
    """Load a single Parquet file into DuckDB using CREATE TABLE AS SELECT."""
    from datetime import datetime
    
    schema_table = f"{dataset}.{table}"
    
    try:
        # Check if table exists, drop if it does (for clean reload)
        conn.execute(f"DROP TABLE IF EXISTS {schema_table}")
        
        # Use CREATE TABLE AS SELECT from Parquet (fast and preserves schema)
        create_sql = f"""
        CREATE TABLE {schema_table} AS 
        SELECT * FROM read_parquet('{parquet_file}')
        """
        
        conn.execute(create_sql)
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {schema_table}").fetchone()[0]
        
        # Create indexes on common columns
        try:
            # Check if date column exists and create index
            columns = conn.execute(f"DESCRIBE {schema_table}").fetchall()
            col_names = [col[0].lower() for col in columns]
            
            if 'date' in col_names:
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{dataset}_{table}_date ON {schema_table}(date)")
            
            if 'symbol' in col_names:
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{dataset}_{table}_symbol ON {schema_table}(symbol)")
        except Exception as idx_error:
            pass  # Index creation is optional
        
        return {
            'dataset': dataset,
            'table': table,
            'status': 'success',
            'row_count': row_count,
            'parquet_file': str(parquet_file),
            'loaded_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'dataset': dataset,
            'table': table,
            'status': 'error',
            'error': str(e),
            'parquet_file': str(parquet_file)
        }


def main():
    """Load all Parquet files into DuckDB."""
    from datetime import datetime
    
    print("=" * 60)
    print("LOADING PARQUET INTO DUCKDB - Phase 1.4")
    print("=" * 60)
    print(f"DuckDB: {DUCKDB_PATH}")
    print(f"Parquet Base: {PARQUET_BASE}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load export manifest
    if not EXPORT_MANIFEST.exists():
        print(f"ERROR: Export manifest not found: {EXPORT_MANIFEST}")
        print("Run export_bq_to_parquet_v2.py first!")
        return
    
    with open(EXPORT_MANIFEST) as f:
        export_data = json.load(f)
    
    # Filter to successful exports
    successful_exports = [
        e for e in export_data['exports']
        if e.get('status') == 'success'
    ]
    
    print(f"Tables to load: {len(successful_exports)}")
    print()
    
    # Connect to DuckDB
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        # Load tables in order: raw → staging → features → training → forecasts → reference
        dataset_order = ['raw', 'staging', 'features', 'training', 'forecasts', 'reference', 'ops']
        
        results = []
        for dataset in dataset_order:
            dataset_exports = [e for e in successful_exports if e['dataset'] == dataset]
            
            if not dataset_exports:
                continue
            
            print(f"Loading {dataset} schema ({len(dataset_exports)} tables)...")
            
            for export_info in dataset_exports:
                parquet_file = Path(export_info['output_path'])
                
                if not parquet_file.exists():
                    print(f"  ⚠ {export_info['table']}: Parquet file not found")
                    results.append({
                        'dataset': dataset,
                        'table': export_info['table'],
                        'status': 'file_not_found',
                        'parquet_file': str(parquet_file)
                    })
                    continue
                
                result = load_table_to_duckdb(conn, dataset, export_info['table'], parquet_file)
                results.append(result)
                
                status_icon = '✓' if result['status'] == 'success' else '✗'
                row_count = result.get('row_count', 0)
                print(f"  {status_icon} {export_info['table']}: {row_count:,} rows")
        
        # Save load manifest
        load_manifest = {
            'load_timestamp': datetime.now().isoformat(),
            'total_tables': len(results),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'total_rows': sum(r.get('row_count', 0) for r in results if r.get('status') == 'success'),
            'loads': results
        }
        
        with open(LOAD_MANIFEST, 'w') as f:
            json.dump(load_manifest, f, indent=2)
        
        print()
        print("=" * 60)
        print("LOAD COMPLETE")
        print("=" * 60)
        print(f"Total tables loaded: {load_manifest['successful']}")
        print(f"Total rows: {load_manifest['total_rows']:,}")
        print(f"Load manifest: {LOAD_MANIFEST}")
        
        # Verify data integrity
        print()
        print("Verifying data integrity...")
        for result in results:
            if result['status'] == 'success':
                dataset = result['dataset']
                table = result['table']
                duckdb_count = result.get('row_count', 0)
                
                # Get original count from export manifest
                export_info = next(
                    (e for e in successful_exports 
                     if e['dataset'] == dataset and e['table'] == table),
                    None
                )
                
                if export_info:
                    original_count = export_info.get('row_count', 0)
                    if duckdb_count == original_count:
                        print(f"  ✓ {dataset}.{table}: {duckdb_count:,} rows (match)")
                    else:
                        print(f"  ⚠ {dataset}.{table}: DuckDB={duckdb_count:,}, Original={original_count:,} (mismatch)")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

