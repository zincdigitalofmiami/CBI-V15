# DuckDB Technical Learning Notes for ZL Pipeline

## 1. Data Ingestion Optimization
### CSV Loading (Performance Critical)
- **Type Inference**: Use `auto_type_candidates` to enforce strict types (e.g., `TIMESTAMP`, `DECIMAL`) and avoid `VARCHAR` overhead.
  ```sql
  SELECT * FROM read_csv('prices.csv', auto_type_candidates = ['TIMESTAMP', 'DECIMAL(18,4)', 'BIGINT']);
  ```
- **Bulk Loading**: Use `COPY` statement instead of `INSERT` for massive speedups.
  ```sql
  COPY zl_prices FROM 'data/zl_futures.csv' (HEADER, DELIMITER ',');
  ```
- **Schema Enforcement**: Define table schema first, then `COPY` to prevent type drift.

### Parquet (Storage Format)
- **Projection Pushdown**: DuckDB automatically pushes `SELECT` columns down to the Parquet reader, minimizing I/O.
- **Glob Patterns**: Read partitioned data efficiently.
  ```sql
  SELECT * FROM read_parquet('data/training_set/*.parquet');
  ```
- **Filename Column**: Use `filename` virtual column to track source files (useful for partition management).

### JSON (Policy Documents)
- **Structured Extraction**: Use `read_json` with explicit `columns` definition for policy text metadata.
  ```sql
  SELECT * FROM read_json('policy_docs.json', 
      columns = {doc_id: 'UBIGINT', content: 'VARCHAR', date: 'DATE'});
  ```
- **Nested Data**: Use `->>` operator to extract text from nested JSON without parsing the whole object.

## 2. Secrets Management (Security)
### Persistent Secrets
- Store credentials securely on disk (unencrypted in `~/.duckdb/stored_secrets` by default) to avoid passing keys in every query.
- **S3 / MotherDuck**:
  ```sql
  CREATE PERSISTENT SECRET motherduck_auth (
      TYPE MOTHERDUCK,
      TOKEN 'your-token-here'
  );
  
  CREATE PERSISTENT SECRET s3_access (
      TYPE S3,
      KEY_ID 'access_key',
      SECRET 'secret_key',
      REGION 'us-east-1'
  );
  ```
- **Provider**: Use `CONFIG` provider (default) or `credential_chain` for auto-discovery (AWS).

## 3. Application to ZL Architecture
- **Fact Table**: `ZL_FACT_PRICES_FEATURES` should be populated via `COPY` from Parquet/CSV.
- **Policy Features**: Ingest JSON policy docs -> Extract features -> Join with Fact Table.
- **Secrets**: Use `CREATE PERSISTENT SECRET` in the initialization script to set up MotherDuck and Databento (S3-compatible) access once.
