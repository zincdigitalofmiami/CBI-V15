-- Train/Validation/Test Split Definitions
-- Time-series aware splits with embargo periods

CREATE TABLE IF NOT EXISTS reference.train_val_test_splits (
    split_id VARCHAR PRIMARY KEY,
    split_name VARCHAR NOT NULL,
    description TEXT,
    
    -- Date boundaries
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    embargo_days INT DEFAULT 5,  -- Gap between train and val
    val_start DATE NOT NULL,
    val_end DATE NOT NULL,
    test_start DATE,
    test_end DATE,
    
    -- Metadata
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default production split
INSERT INTO reference.train_val_test_splits VALUES (
    'prod_v1',
    'Production Split V1',
    '5-year train, 1-year val, 5-day embargo',
    '2019-01-01', '2023-12-31',
    5,
    '2024-01-06', '2024-12-31',
    NULL, NULL,
    TRUE, CURRENT_TIMESTAMP
) ON CONFLICT DO NOTHING;

