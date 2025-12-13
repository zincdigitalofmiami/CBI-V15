-- Raw USDA Crop Progress
-- Weekly crop condition and progress reports

CREATE TABLE IF NOT EXISTS raw.usda_crop_progress (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    state VARCHAR,
    metric VARCHAR NOT NULL,  -- 'planted_pct', 'emerged_pct', 'condition_good_excellent_pct'
    value DECIMAL(5, 2),
    source VARCHAR DEFAULT 'usda_crop_progress',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity, state, metric)
);

-- Key states: Iowa, Illinois, Minnesota, Indiana, Nebraska
-- Metrics: planting progress, crop condition ratings

CREATE INDEX IF NOT EXISTS idx_crop_progress_state 
    ON raw.usda_crop_progress(state);

