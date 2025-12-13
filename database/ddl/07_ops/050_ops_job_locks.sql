-- Ops: Job Locks
-- Distributed locking for concurrent jobs

CREATE TABLE IF NOT EXISTS ops.job_locks (
    lock_name VARCHAR PRIMARY KEY,
    locked_by VARCHAR NOT NULL,  -- Worker ID
    locked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    -- Metadata
    job_type VARCHAR,
    notes TEXT
);

-- Used for backpressure / preventing duplicate runs
-- Lock expires after TTL (typically 10-15 min for inference, 2h for training)

