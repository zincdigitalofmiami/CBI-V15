#!/usr/bin/env python3
"""
Compare DuckDB Schema with BigQuery Schema
Checks what's actually in the external drive DuckDB vs what should be there.
"""

import duckdb
import json
from pathlib import Path
from typing import Dict, List

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')
BQ_SCHEMA_FILE = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/bq_schemas_detailed.json')
OUTPUT_FILE = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/schema_comparison.json')


def get_duckdb_schemas(conn: duckdb.DuckDBPyConnection) -> Dict:
    """Get all schemas and tables from DuckDB."""
    schemas = {}
    
    # Get all schemas
    schema_list = conn.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_temp', 'main')
        ORDER BY schema_name
    """).fetchall()
    
    for (schema_name,) in schema_list:
        schemas[schema_name] = {}
        
        # Get tables in this schema
        tables = conn.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{schema_name}' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """).fetchall()
        
        for (table_name,) in tables:
            # Get columns
            columns = conn.execute(f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            
            col_list = []
            for col_name, col_type, nullable in columns:
                col_list.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': nullable == 'YES'
                })
            
            # Get row count
            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}").fetchone()[0]
            except:
                row_count = 0
            
            schemas[schema_name][table_name] = {
                'columns': col_list,
                'column_count': len(col_list),
                'row_count': row_count
            }
    
    return schemas


def compare_schemas(bq_schemas: Dict, duckdb_schemas: Dict) -> Dict:
    """Compare BigQuery and DuckDB schemas."""
    comparison = {
        'missing_datasets': [],
        'missing_tables': {},
        'extra_tables': {},
        'schema_mismatches': {},
        'row_count_differences': {}
    }
    
    # Check datasets
    bq_datasets = set(bq_schemas['datasets'].keys())
    duckdb_datasets = set(duckdb_schemas.keys())
    
    comparison['missing_datasets'] = list(bq_datasets - duckdb_datasets)
    comparison['extra_datasets'] = list(duckdb_datasets - bq_datasets)
    
    # Check tables in each dataset
    for dataset in bq_datasets:
        if dataset not in duckdb_schemas:
            comparison['missing_tables'][dataset] = list(bq_schemas['datasets'][dataset])
            continue
        
        bq_tables = {t['table']: t for t in bq_schemas['datasets'][dataset]}
        duckdb_tables = duckdb_schemas[dataset]
        
        bq_table_names = set(bq_tables.keys())
        duckdb_table_names = set(duckdb_tables.keys())
        
        # Missing tables
        missing = bq_table_names - duckdb_table_names
        if missing:
            comparison['missing_tables'][dataset] = [
                {'table': t, 'columns': bq_tables[t]['columns']} 
                for t in missing
            ]
        
        # Extra tables
        extra = duckdb_table_names - bq_table_names
        if extra:
            comparison['extra_tables'][dataset] = [
                {'table': t, 'columns': duckdb_tables[t]['columns']} 
                for t in extra
            ]
        
        # Compare existing tables
        common_tables = bq_table_names & duckdb_table_names
        for table_name in common_tables:
            bq_table = bq_tables[table_name]
            duckdb_table = duckdb_tables[table_name]
            
            # Column comparison
            bq_cols = {c['name']: c for c in bq_table['columns']}
            duckdb_cols = {c['name']: c for c in duckdb_table['columns']}
            
            bq_col_names = set(bq_cols.keys())
            duckdb_col_names = set(duckdb_cols.keys())
            
            missing_cols = bq_col_names - duckdb_col_names
            extra_cols = duckdb_col_names - bq_col_names
            type_mismatches = []
            
            for col_name in bq_col_names & duckdb_col_names:
                bq_type = bq_cols[col_name]['type']
                duckdb_type = duckdb_cols[col_name]['type']
                
                # Normalize types for comparison
                bq_norm = normalize_type(bq_type)
                duckdb_norm = normalize_type(duckdb_type)
                
                if bq_norm != duckdb_norm:
                    type_mismatches.append({
                        'column': col_name,
                        'bq_type': bq_type,
                        'duckdb_type': duckdb_type
                    })
            
            if missing_cols or extra_cols or type_mismatches:
                comparison['schema_mismatches'][f"{dataset}.{table_name}"] = {
                    'missing_columns': list(missing_cols),
                    'extra_columns': list(extra_cols),
                    'type_mismatches': type_mismatches,
                    'bq_column_count': len(bq_cols),
                    'duckdb_column_count': len(duckdb_cols)
                }
            
            # Row count comparison
            bq_row_count = bq_table.get('row_count', 0)
            duckdb_row_count = duckdb_table.get('row_count', 0)
            
            if bq_row_count != duckdb_row_count:
                comparison['row_count_differences'][f"{dataset}.{table_name}"] = {
                    'bq_rows': bq_row_count,
                    'duckdb_rows': duckdb_row_count,
                    'difference': duckdb_row_count - bq_row_count
                }
    
    return comparison


def normalize_type(db_type: str) -> str:
    """Normalize database types for comparison."""
    db_type = db_type.upper()
    
    # BigQuery types
    if db_type in ['STRING', 'VARCHAR']:
        return 'STRING'
    if db_type in ['INT64', 'BIGINT', 'INTEGER']:
        return 'INTEGER'
    if db_type in ['FLOAT64', 'DOUBLE', 'FLOAT']:
        return 'FLOAT'
    if db_type == 'DATE':
        return 'DATE'
    if db_type == 'BOOL' or db_type == 'BOOLEAN':
        return 'BOOLEAN'
    if db_type == 'TIMESTAMP':
        return 'TIMESTAMP'
    
    return db_type


def main():
    """Compare BigQuery and DuckDB schemas."""
    print("=" * 60)
    print("COMPARING BIGQUERY AND DUCKDB SCHEMAS")
    print("=" * 60)
    print(f"DuckDB: {DUCKDB_PATH}")
    print(f"BigQuery Schema: {BQ_SCHEMA_FILE}")
    print()
    
    # Load BigQuery schemas
    if not BQ_SCHEMA_FILE.exists():
        print(f"ERROR: BigQuery schema file not found: {BQ_SCHEMA_FILE}")
        print("Run get_bq_schemas.py first!")
        return
    
    with open(BQ_SCHEMA_FILE) as f:
        bq_schemas = json.load(f)
    
    # Connect to DuckDB
    if not DUCKDB_PATH.exists():
        print(f"ERROR: DuckDB database not found: {DUCKDB_PATH}")
        return
    
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        print("Reading DuckDB schemas...")
        duckdb_schemas = get_duckdb_schemas(conn)
        
        print(f"Found {len(duckdb_schemas)} schemas in DuckDB")
        for schema, tables in duckdb_schemas.items():
            print(f"  {schema}: {len(tables)} tables")
        
        print()
        print("Comparing schemas...")
        comparison = compare_schemas(bq_schemas, duckdb_schemas)
        
        # Save comparison
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print summary
        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        if comparison['missing_datasets']:
            print(f"\nMissing Datasets ({len(comparison['missing_datasets'])}):")
            for ds in comparison['missing_datasets']:
                print(f"  - {ds}")
        
        if comparison['extra_datasets']:
            print(f"\nExtra Datasets ({len(comparison['extra_datasets'])}):")
            for ds in comparison['extra_datasets']:
                print(f"  - {ds}")
        
        if comparison['missing_tables']:
            print(f"\nMissing Tables:")
            for dataset, tables in comparison['missing_tables'].items():
                print(f"  {dataset}: {len(tables)} tables")
                for table in tables[:5]:  # Show first 5
                    if isinstance(table, dict):
                        col_count = len(table.get('columns', []))
                        print(f"    - {table['table']} ({col_count} cols)")
                    else:
                        print(f"    - {table}")
                if len(tables) > 5:
                    print(f"    ... and {len(tables) - 5} more")
        
        if comparison['extra_tables']:
            print(f"\nExtra Tables:")
            for dataset, tables in comparison['extra_tables'].items():
                print(f"  {dataset}: {len(tables)} tables")
                for table in tables[:5]:
                    print(f"    - {table['table']} ({table['column_count']} cols)")
        
        if comparison['schema_mismatches']:
            print(f"\nSchema Mismatches ({len(comparison['schema_mismatches'])} tables):")
            for table, mismatch in list(comparison['schema_mismatches'].items())[:10]:
                print(f"  {table}:")
                if mismatch['missing_columns']:
                    print(f"    Missing columns: {mismatch['missing_columns']}")
                if mismatch['extra_columns']:
                    print(f"    Extra columns: {mismatch['extra_columns']}")
                if mismatch['type_mismatches']:
                    print(f"    Type mismatches: {len(mismatch['type_mismatches'])}")
        
        if comparison['row_count_differences']:
            print(f"\nRow Count Differences ({len(comparison['row_count_differences'])} tables):")
            for table, diff in list(comparison['row_count_differences'].items())[:10]:
                print(f"  {table}: BQ={diff['bq_rows']:,}, DuckDB={diff['duckdb_rows']:,}, Diff={diff['difference']:,}")
        
        print()
        print(f"Full comparison saved to: {OUTPUT_FILE}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

