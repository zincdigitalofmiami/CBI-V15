-- TSci Tables: Jobs, Runs, QA Checks

-- 1. Jobs (Persistent Definitions)
CREATE TABLE IF NOT EXISTS tsci.jobs (
    job_id               BIGINT PRIMARY KEY,
    job_name             TEXT NOT NULL,
    job_type             TEXT NOT NULL, -- 'training','forecast','qa','ingestion'
    description          TEXT,
    is_enabled           BOOLEAN NOT NULL DEFAULT TRUE,
    schedule_cron        TEXT,
    symbol               TEXT,
    horizon              TEXT,
    model_id             TEXT,
    config_json          TEXT,
    last_run_id          BIGINT,
    last_run_at          TIMESTAMP,
    last_run_status      TEXT,
    created_by           TEXT NOT NULL DEFAULT 'tsci',
    created_at           TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated_by           TEXT,
    updated_at           TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tsci_jobs_type_enabled ON tsci.jobs (job_type, is_enabled);

-- 2. Runs (Execution Logs)
CREATE TABLE IF NOT EXISTS tsci.runs (
    run_id               BIGINT PRIMARY KEY,
    job_id               BIGINT,
    model_id             TEXT,
    horizon              TEXT,
    symbol               TEXT,
    status               TEXT NOT NULL, -- 'queued','running','success','failed','skipped'
    error_message        TEXT,
    started_at           TIMESTAMP NOT NULL DEFAULT current_timestamp,
    finished_at          TIMESTAMP,
    params_json          TEXT,
    metrics_json         TEXT,
    training_rows        BIGINT,
    validation_rows      BIGINT,
    test_rows            BIGINT,
    git_commit           TEXT,
    artifact_path        TEXT,
    created_by           TEXT NOT NULL DEFAULT 'tsci',
    created_at           TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_tsci_runs_job ON tsci.runs (job_id);
CREATE INDEX IF NOT EXISTS idx_tsci_runs_model_status ON tsci.runs (model_id, status);

-- 3. QA Checks (Structured Results)
CREATE TABLE IF NOT EXISTS tsci.qa_checks (
    check_id             BIGINT PRIMARY KEY,
    run_id               BIGINT,
    check_name           TEXT NOT NULL,
    check_type           TEXT NOT NULL, -- 'schema','nulls','drift','leakage','performance'
    target_table         TEXT NOT NULL,
    target_column        TEXT,
    status               TEXT NOT NULL, -- 'passed','failed','warning'
    severity             TEXT NOT NULL DEFAULT 'medium',
    expected_value       TEXT,
    observed_value       TEXT,
    details_json         TEXT,
    created_at           TIMESTAMP NOT NULL DEFAULT current_timestamp,
    created_by           TEXT NOT NULL DEFAULT 'tsci'
);

CREATE INDEX IF NOT EXISTS idx_tsci_qa_checks_table ON tsci.qa_checks (target_table);
CREATE INDEX IF NOT EXISTS idx_tsci_qa_checks_run ON tsci.qa_checks (run_id);
