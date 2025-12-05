tha---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# BigQuery Integration with PyTorch Training

## Actual CBI-V15 Architecture (November 2025)

### Hybrid Python + BigQuery Pattern (Already in Production)

**CRITICAL**: CBI-V15 already uses a **hybrid system** - this is NOT theoretical, it's the actual production pattern.

```
    ↓
External Drive (/Volumes/Satechi Hub/) + BigQuery Raw Tables
    ↓
HYBRID Feature Engineering:
    ├── BigQuery SQL: Correlations (CORR() OVER), regimes, moving averages
    ├── Python: Sentiment (NLP), policy extraction, complex interactions
    ↓
BigQuery Training Tables (training.zl_training_prod_allhistory_*)
    ↓
PyTorch Training on M4 Mac (MPS backend)
    ↓
Upload Predictions to BigQuery → Dashboard Reads Views
```

## BigQuery Responsibilities

### 1. Light Calculations (SQL)

**What BigQuery Calculates**:
```sql
CORR(zl_price, fed_funds_rate) OVER w30 AS corr_zl_fed_30d
CORR(zl_price, vix) OVER w30 AS corr_zl_vix_30d
CORR(zl_price, dollar_index) OVER w30 AS corr_zl_dxy_30d

-- Regimes from FRED thresholds
CASE 
  WHEN fed_funds_rate < 1.0 THEN 'ultra_low'
  WHEN fed_funds_rate < 2.5 THEN 'low'
  WHEN fed_funds_rate < 4.0 THEN 'normal'
  ELSE 'high'
END AS rate_regime

-- Simple interactions
zl_return * fed_rate_change AS zl_fed_interaction
zl_return * dollar_momentum AS zl_dxy_interaction

AVG(zl_price) OVER w20 AS ma_20
AVG(zl_price) OVER w50 AS ma_50
```

**Existing SQL Files**:
- `config/bigquery/bigquery-sql/advanced_feature_engineering.sql` ✅ EXISTS
- `config/bigquery/bigquery-sql/signals/create_big8_signal_views.sql` ✅ EXISTS
- `config/bigquery/bigquery-sql/POPULATE_MOVING_AVERAGES.sql` ✅ EXISTS

### 2. Scheduling & Orchestration

**BigQuery Scheduled Queries** (or Cloud Scheduler):
```sql
-- Daily feature refresh
CREATE OR REPLACE TABLE features.zl_daily_features AS
SELECT 
  date,
  zl_price,
  -- Correlations from BigQuery SQL
  corr_zl_fed_30d,
  corr_zl_vix_30d,
  -- Regimes from BigQuery SQL
  rate_regime,
  av_rsi,
  av_macd,
  av_bbands_upper,
  -- Python features (calculated separately, then joined)
  sentiment_score,
  policy_impact
FROM 
LEFT JOIN features.zl_correlations corr ON zl.date = corr.date
LEFT JOIN features.zl_sentiment sent ON zl.date = sent.date
```

### 3. Storage & Data Warehouse

**BigQuery Tables Structure**:
```
raw_intelligence.* (7 tables)
├── fred_economic_data (30+ series)

features.* (feature views)
├── zl_correlations (BQ SQL calculated)
├── zl_regimes (BQ SQL calculated)
├── zl_sentiment (Python calculated, stored)
└── master_feature_universe (all features combined)

training.* (training tables)
├── zl_training_prod_allhistory_1w (305 cols)
├── zl_training_prod_allhistory_1m (449 cols)
├── zl_training_prod_allhistory_3m (305 cols)
├── zl_training_prod_allhistory_6m (305 cols)
└── zl_training_prod_allhistory_12m (306 cols)
```

## Python Responsibilities

### 1. Data Collection

**Pattern** (follow `collect_fred_comprehensive.py`):
```python
    """Follow existing pattern - don't redesign"""
    
    def collect(self):
        
        # 2. Save to external drive (consistent with other scripts)
        data.to_parquet(external_path / f'{date}.parquet')
        
        # 3. Upload to BigQuery (direct upload)
        client = bigquery.Client()
        job = client.load_table_from_dataframe(
            data,
            write_disposition='WRITE_APPEND'
        )
        
        logger.info(f"Uploaded {len(data)} rows to BigQuery")
```

### 2. Complex Feature Engineering

**Python Calculates**:
```python
# Sentiment (NLP)
def calculate_sentiment(news_text):
    """Complex NLP processing"""
    sentiment_scores = finbert_model(news_text)
    return sentiment_scores

# Policy extraction
def extract_policy_impacts(news_data):
    """Extract RFS, tariffs, subsidies from text"""
    policy_features = {
        'rfs_mention': detect_rfs(news_data),
        'tariff_impact': extract_tariff(news_data),
        'subsidy_effect': extract_subsidy(news_data)
    }
    return policy_features

# Complex interactions
def calculate_amplified_features(base_features):
    """850+ features from 18 symbols"""
    # Complex multi-factor calculations
    return amplified_features
```

**Existing Python Files**:
- `scripts/features/feature_calculations.py` ✅ EXISTS (900+ lines)
- `scripts/features/calculate_amplified_features.py` ✅ EXISTS
- `scripts/features/build_all_features.py` ✅ EXISTS


### Pre-Calculated Technicals (Don't Recalculate)

```
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, VWAP
RSI, MACD, STOCH, STOCHRSI, WILLR, ADX, ADXR
BBANDS, ATR, NATR, TRANGE
OBV, AD, ADOSC
MOM, ROC, AROON, CCI, MFI, TRIX
HT_TRENDLINE, HT_SINE, HT_DCPERIOD, etc.
```

**Strategy**: Store in BigQuery as-is, join to training tables

```python
# DON'T recalculate - just load and use
av_data = client.query("""
    SELECT date, rsi, macd, sma_20, ema_50, bbands_upper, bbands_lower
    WHERE symbol = 'SOYB'
""").to_dataframe()

# Merge with other features
final_features = pd.merge(existing_features, av_data, on='date')
```

## PyTorch Training Integration

### Data Loading from BigQuery

```python
class BigQueryDataset(Dataset):
    """
    Load training data from BigQuery training tables
    """
    
    def __init__(self, horizon='1m', mode='train'):
        self.horizon = horizon
        self.mode = mode
        
        # Load from BigQuery training table
        query = f"""
        SELECT *
        FROM `cbi-v15.training.zl_training_prod_allhistory_{horizon}`
        WHERE date >= '2000-01-01'  -- All 25+ years
        ORDER BY date
        """
        
        self.data = client.query(query).to_dataframe()
        
        # Split train/val (time-based, never shuffle)
        split_date = '2020-01-01'
        if mode == 'train':
            self.data = self.data[self.data['date'] < split_date]
        else:
            self.data = self.data[self.data['date'] >= split_date]
        
        # Extract features (30-60 curated)
        self.features = self._extract_features()
        self.targets = self._extract_targets()
        
    def _extract_features(self):
        """Extract curated 30-60 features"""
        feature_categories = {
            'prices': ['close', 'close_returns', 'log_returns'],
            'correlations': ['corr_zl_fed_30d', 'corr_zl_vix_30d'],  # From BQ SQL
            'regimes': ['rate_regime', 'volatility_regime'],  # From BQ SQL
            'weather': ['brazil_precip_7d_zscore', 'brazil_gdd'],  # From NOAA
            'sentiment': ['sentiment_score', 'sentiment_ma_7d'],  # From Python NLP
            'positioning': ['cftc_managed_money_long', 'cftc_commercial_short']  # From CFTC
        }
        
        features = []
        for category, cols in feature_categories.items():
            features.extend([col for col in cols if col in self.data.columns])
        
        return self.data[features].values
    
    def __getitem__(self, idx):
        # Get sequence of 252 trading days
        sequence = self.features[idx:idx+252]
        target = self.targets[idx+252]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)
```

### Training Workflow

```python
class CBI_V14_TrainingPipeline:
    """
    Complete training pipeline integrated with BigQuery
    """
    
    def __init__(self):
        self.client = bigquery.Client()
        
    def export_training_data(self, horizon='1m'):
        """
        Export from BigQuery to parquet for local training
        Uses existing script: scripts/export_training_data.py
        """
        query = f"""
        SELECT *
        FROM `cbi-v15.training.zl_training_prod_allhistory_{horizon}`
        ORDER BY date
        """
        
        df = self.client.query(query).to_dataframe()
        
        # Save to external drive
        export_path = Path('/Volumes/Satechi Hub/Projects/CBI-V15/TrainingData/exports/')
        df.to_parquet(export_path / f'zl_training_{horizon}.parquet')
        
        return export_path
    
    def train_model(self, horizon='1m'):
        """
        Train PyTorch model on M4 Mac
        """
        # Load parquet from export
        data_path = self.export_training_data(horizon)
        dataset = BigQueryDataset(horizon=horizon, mode='train')
        
        # Train with PyTorch MPS
        model = TemporalConvolutionalNetwork(input_dim=50)
        trainer = M4OptimizedTraining(model, dataset)
        trainer.train()
        
        # Save model
        model_path = Path(f'Models/local/{horizon}/tcn_model.pt')
        torch.save(model.state_dict(), model_path)
        
        return model_path
    
    def upload_predictions(self, predictions, horizon='1m'):
        """
        Upload predictions back to BigQuery
        Uses existing script: scripts/upload_predictions.py
        """
        predictions_df = pd.DataFrame({
            'date': predictions['dates'],
            'horizon': horizon,
            'predicted_price': predictions['prices'],
            'confidence': predictions['confidence']
        })
        
        # Upload to BigQuery
        job = self.client.load_table_from_dataframe(
            predictions_df,
            'cbi-v15.predictions.zl_predictions',
            write_disposition='WRITE_APPEND'
        )
        
        logger.info(f"Uploaded {len(predictions_df)} predictions to BigQuery")
```

## Daily Pipeline Integration

### Complete Workflow

```python
class DailyPipeline:
    """
    Daily production pipeline integrating BigQuery and PyTorch
    """
    
    def run_daily(self):
        """
        Complete daily workflow
        """
        # 1. Data Collection (Python scripts)
        # - FRED: collect_fred_comprehensive.py → External drive + BQ
        
        # 2. Feature Engineering (Hybrid)
        # - BigQuery SQL: Refresh correlations, regimes (scheduled query)
        # - Python: Calculate sentiment, policy features
        
        # 3. Training Table Refresh (BigQuery)
        # - Join all features into training tables
        # - Export parquet for local training
        
        # 4. Model Training (M4 Mac PyTorch)
        # - Load parquet from export
        # - Train with PyTorch MPS backend
        # - Save model locally
        
        # 5. Generate Predictions (M4 Mac PyTorch)
        # - Load latest features from BigQuery
        # - Run PyTorch inference
        # - Upload predictions to BigQuery
        
        # 6. Dashboard (Reads BigQuery Views)
        # - Dashboard reads from predictions.* tables
        # - No direct model access needed
```

## Feature Computation Strategy

### Where Each Feature is Calculated

| Feature Type | Calculation Location | Example |
|--------------|---------------------|---------|
| **ZL Correlations** | BigQuery SQL | CORR(zl_price, fed_rate) |
| **Regimes** | BigQuery SQL | CASE WHEN fed_rate < 1.0... |
| **Sentiment** | Python (NLP) | FinBERT sentiment scoring |
| **Policy Features** | Python (extraction) | RFS mentions, tariff impacts |
| **Weather Aggregations** | Python | Brazil precip z-scores |
| **CFTC Positioning** | Python (from CFTC data) | Managed money flows |

### Integration Pattern

```python

# 2. Load BigQuery-calculated correlations
bq_correlations = load_from_bq('features.zl_correlations')

# 3. Calculate Python features
python_features = calculate_sentiment(news_data)
python_features.update(calculate_policy_impacts(news_data))

# 4. Merge all features
final_features = pd.merge(
    av_features,
    bq_correlations,
    on='date'
).merge(
    python_features,
    on='date'
)

# 5. Store in training table
write_to_bq(final_features, 'training.zl_training_prod_allhistory_1m')
```

## Key Takeaways

1. **BigQuery owns**: Light calculations (correlations, regimes), scheduling, storage
2. **Python owns**: Complex features (sentiment, NLP, policy), data collection
4. **PyTorch**: Trains on M4 Mac, uploads predictions to BigQuery
5. **Dashboard**: Reads from BigQuery views (no direct model access)

**This is the ACTUAL production architecture - not theoretical.**

---

*Updated: November 17, 2025 - Based on complete architecture audit*

## Architecture Audit Confirmation (Nov 17, 2025)

**VERIFIED**: The hybrid architecture described in this document is the ACTUAL production system:
- External drive usage: `/Volumes/Satechi Hub/` (620MB raw data)
- BigQuery SQL features: `advanced_feature_engineering.sql`, `create_big8_signal_views.sql`
- Python features: `feature_calculations.py` (900+ lines)
- Training tables: `training.zl_training_prod_allhistory_*` (305-449 features)
- No Cloud Run jobs (all local/cron currently)

- Integration: Follows existing pattern (Python → External drive + BQ)
- Premium Plan75: 75 API calls/minute
- MCP server: Configured and tested

See `docs/plans/ARCHITECTURE_REVIEW_REPORT.md` for complete audit findings.
