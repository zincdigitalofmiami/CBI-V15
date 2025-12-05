#!/usr/bin/env python3
"""
Get Detailed BigQuery Schemas - Update Plan with Actual Schema
Queries BigQuery INFORMATION_SCHEMA to get complete table structures.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ID = 'cbi-v15'
DATASETS = ['raw', 'staging', 'features', 'training', 'forecasts', 'reference', 'ops']
OUTPUT_DIR = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration')
SCHEMA_FILE = OUTPUT_DIR / 'bq_schemas_detailed.json'


def get_table_schema(dataset: str, table: str) -> Dict:
    """Get detailed schema for a single table."""
    query = f"""
    SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
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
            columns = []
            for row in data:
                columns.append({
                    'name': row.get('column_name', ''),
                    'type': row.get('data_type', ''),
                    'nullable': row.get('is_nullable', 'YES') == 'YES',
                    'default': row.get('column_default', None)
                })
            
            return {
                'dataset': dataset,
                'table': table,
                'columns': columns,
                'column_count': len(columns)
            }
    except Exception as e:
        print(f"Error getting schema for {dataset}.{table}: {e}")
    
    return {
        'dataset': dataset,
        'table': table,
        'columns': [],
        'column_count': 0
    }


def list_tables_in_dataset(dataset: str) -> List[str]:
    """List all tables in a dataset."""
    query = f"""
    SELECT table_name
    FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE'
    ORDER BY table_name
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


def main():
    """Get schemas for all tables across all datasets."""
    print("=" * 60)
    print("GETTING BIGQUERY SCHEMAS - Detailed")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_schemas = []
    
    # Process each dataset
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        tables = list_tables_in_dataset(dataset)
        
        if not tables:
            print(f"  No tables found")
            continue
        
        print(f"  Found {len(tables)} tables")
        
        # Get schemas in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(get_table_schema, dataset, table): table 
                for table in tables
            }
            
            for future in as_completed(futures):
                table = futures[future]
                try:
                    schema = future.result()
                    all_schemas.append(schema)
                    print(f"    {table}: {schema['column_count']} columns")
                except Exception as e:
                    print(f"    {table}: ERROR - {e}")
    
    # Organize by dataset
    schemas_by_dataset = {}
    for schema in all_schemas:
        dataset = schema['dataset']
        if dataset not in schemas_by_dataset:
            schemas_by_dataset[dataset] = []
        schemas_by_dataset[dataset].append(schema)
    
    # Save detailed schemas
    output = {
        'project_id': PROJECT_ID,
        'timestamp': datetime.now().isoformat(),
        'total_tables': len(all_schemas),
        'datasets': schemas_by_dataset
    }
    
    with open(SCHEMA_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("=" * 60)
    print("SCHEMA EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total tables: {len(all_schemas)}")
    print(f"Schema file: {SCHEMA_FILE}")
    print()
    
    # Summary by dataset
    print("Summary by dataset:")
    for dataset in DATASETS:
        if dataset in schemas_by_dataset:
            tables = schemas_by_dataset[dataset]
            total_cols = sum(t['column_count'] for t in tables)
            print(f"  {dataset}: {len(tables)} tables, {total_cols} total columns")


if __name__ == '__main__':
    main()

