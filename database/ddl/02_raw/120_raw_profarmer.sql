-- Raw ProFarmer Data
-- Premium agricultural intelligence

CREATE TABLE IF NOT EXISTS raw.profarmer_articles (
    article_id VARCHAR PRIMARY KEY,
    published_date TIMESTAMP NOT NULL,
    category VARCHAR,  -- 'crop_tour', 'market_outlook', 'policy', 'weather'
    title VARCHAR NOT NULL,
    content TEXT,
    crops_mentioned VARCHAR[],  -- ['soybeans', 'corn', 'wheat']
    regions_mentioned VARCHAR[],
    sentiment_score DECIMAL(5, 4),
    source VARCHAR DEFAULT 'profarmer',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw.profarmer_crop_tour (
    tour_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    state VARCHAR,
    crop VARCHAR,
    yield_estimate DECIMAL(6, 2),
    pod_count DECIMAL(6, 2),
    ear_count DECIMAL(6, 2),
    notes TEXT,
    source VARCHAR DEFAULT 'profarmer',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tour_id, date, state)
);

-- ProFarmer Crop Tour is key for pre-harvest yield estimates

CREATE INDEX IF NOT EXISTS idx_profarmer_category 
    ON raw.profarmer_articles(category);
CREATE INDEX IF NOT EXISTS idx_profarmer_date 
    ON raw.profarmer_articles(published_date);

