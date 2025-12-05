---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# CURSOR MASTER INSTRUCTION SET
## Dataset Audit Rules After Any Major Market Move

**Version**: 1.0  
**Date**: November 14, 2025  
**Binding**: ALL AI assistants must follow this protocol  
**Override**: Cannot be bypassed without explicit human approval

---

## 0. TRIGGER CONDITIONS (When Audit Must Execute)

Cursor **MUST** automatically enter "Post-Move Audit Mode" when **any** of these conditions occur:

### Market Regime Triggers

| Trigger | Threshold | Data Source | Check Frequency |
|---------|-----------|-------------|-----------------|
| **VIX Spike** | > 25 | `raw_intelligence.vol_vix_daily` | Every 15min during market hours |
| **USD/BRL Move** | > 3% in 24h | `raw_intelligence.fx_usd_brl_daily` | Every hour |
| **ZL Price Move** | > 2% in session | `raw_intelligence.commodity_soybean_oil_daily` | Real-time |
| **FCPO Move** | > 3% in 24h | `raw_intelligence.commodity_palm_oil_daily` | Every 4 hours |
| **Drought Spike** | > +2 σ | `raw_intelligence.weather_*_daily` | Daily |
| **USDA Report** | Release day | USDA calendar | Scheduled |
| **China Quota** | Announcement | `raw_intelligence.trade_china_*` | Event-driven |
| **Tariff Update** | Policy change | `raw_intelligence.policy_*` | Event-driven |

### Derivation Sources

These triggers are derived from:
- **Big 7 Signals**: VIX, USD, Weather, Policy, Demand, Logistics, Sentiment
- **Regime Detector**: Crisis classification algorithm
- **Ultimate Single Signal Architecture**: Signal consolidation framework

**Implementation**: All trigger conditions must be checked by monitoring cron jobs (see `config/system/cron/market_monitor.sh`)

---

## 1. WHERE CURSOR MUST AUDIT (Canonical Locations Only)

Cursor is **FORBIDDEN** from creating new datasets. Only these locations are valid:

### Raw Layer (Bronze) - `raw_intelligence.*`

**Purpose**: Unprocessed ingestion data with full provenance

**Tables** (partial list):
```
raw_intelligence.commodity_soybean_oil_daily
raw_intelligence.commodity_palm_oil_daily
raw_intelligence.vol_vix_daily
raw_intelligence.fx_dxy_usd_index_daily
raw_intelligence.weather_brazil_daily
raw_intelligence.weather_argentina_daily
raw_intelligence.policy_trump_intelligence_daily
raw_intelligence.trade_china_soybean_imports_monthly
raw_intelligence.news_scored_daily
raw_intelligence.shipping_baltic_dry_index_daily
```

**Source**: Documented in `docs/reference/HISTORICAL_FUTURES_DATA.md`

**Characteristics**:
- Has `time TIMESTAMP` (not just DATE)
- Has `source STRING` (TE, Barchart, GDELT, NOAA, USDA)
- Has `ingest_timestamp TIMESTAMP`
- Has `provenance_uuid STRING`

---

### Curated Layer (Silver) - `curated.*`

**Purpose**: Aggregated, validated features

**Tables** (partial list):
```
curated.weather_aggregates_daily
curated.geopolitical_aggregates_daily
curated.substitution_aggregates_daily
curated.technical_aggregates_daily
curated.crush_spread_features_daily
```

**Source**: Documented in `docs/reference/COMPREHENSIVE_SIGNAL_UNIVERSE.md`

**Characteristics**:
- Has `signal_date DATE` (canonical date key)
- All upstream nulls resolved
- No duplicate dates
- Aggregations validated

---

### Feature Layer (Gold) - `training.*`

**Purpose**: Model-ready training matrices

**Tables**:
```
training.zl_training_prod_allhistory_1w
training.zl_training_prod_allhistory_1m
training.zl_training_prod_allhistory_3m
training.zl_training_prod_allhistory_6m
training.zl_training_prod_allhistory_12m

training.zl_training_full_allhistory_1w  (1,948+ features)
training.zl_training_full_allhistory_1m
...

training.regime_calendar  (13,102 rows)
training.regime_weights   (11 rows, weights 50-5000)
```

**Source**: Documented in `ARCHITECTURE_ALIGNMENT_COMPLETE.md`

**Characteristics**:
- Partitioned by `as_of_date DATE`
- Clustered by `(horizon, regime)`
- Has `row_weight INT64` (from regime_weights)
- Has all target columns: `target_return`, `target_price`

---

### Prediction Layer - `predictions.*`

**Purpose**: Model outputs uploaded from local training

**Tables**:
```
predictions.zl_1w_inference_{model}_{version}
predictions.zl_1m_inference_{model}_{version}
...

predictions.vw_zl_1w_latest  (views)
predictions.vw_zl_1m_latest
...
```

**Source**: Created by `scripts/upload_predictions.py`

**Characteristics**:
- Has `as_of_date DATE`
- Has `predicted_price FLOAT64`
- Has `predicted_return FLOAT64`
- Has `model_name STRING`, `model_version STRING`
- Has `created_at TIMESTAMP`

---

## 2. WHAT CURSOR MUST AUDIT (Mandatory Checks in Order)

After any trigger condition, execute these checks **in this exact order**:

---

### (A) Raw Layer Audit - "Integrity First"

**Priority**: CRITICAL (if raw data is corrupt, everything downstream fails)

#### Check 1: Timestamp Gaps

```sql
-- Run on EVERY raw table after trigger
SELECT 
  table_name,
  COUNT(*) as total_rows,
  MIN(time) as earliest,
  MAX(time) as latest,
  COUNT(DISTINCT DATE(time)) as unique_dates,
  DATE_DIFF(MAX(DATE(time)), MIN(DATE(time)), DAY) + 1 as expected_dates,
  COUNT(DISTINCT DATE(time)) - (DATE_DIFF(MAX(DATE(time)), MIN(DATE(time)), DAY) + 1) as gap_count
FROM `raw_intelligence.*`
WHERE DATE(time) >= CURRENT_DATE() - 90
GROUP BY table_name
HAVING gap_count != 0;
```

**Required**: Zero gaps, zero duplicates, zero timezone drift

**Action on Failure**:
- Log to `monitoring.data_quality_events`
- Alert: "CRITICAL: Timestamp gaps in {table}"
- Quarantine affected rows
- Halt downstream processing

---

#### Check 2: Value Sanity

**TE Palm Oil Corruption Check** (ALWAYS run this):

```sql
-- Known bug: TE sometimes reports 1021.0 for palm oil
SELECT DATE(time), close, source
FROM `raw_intelligence.commodity_palm_oil_daily`
WHERE close = 1021.0 
  AND source = 'trading_economics'
  AND DATE(time) >= CURRENT_DATE() - 7;
```

**Source**: Documented in `docs/handoffs/CRITICAL_CONTEXT_READ_FIRST.md`

**Action on Detection**:
- Quarantine rows with `close = 1021.0`
- Backfill from Barchart or alternative source
- Log: "TE palm oil corruption detected"

---

**FX Value Bounds**:

```sql
-- FX must be within historical ±5σ
WITH stats AS (
  SELECT 
    AVG(close) as mean,
    STDDEV(close) as std
  FROM `raw_intelligence.fx_*_daily`
  WHERE DATE(time) >= CURRENT_DATE() - 365
)
SELECT *
FROM `raw_intelligence.fx_*_daily`
WHERE ABS(close - stats.mean) > 5 * stats.std
  AND DATE(time) >= CURRENT_DATE() - 7;
```

**Action on Detection**: Quarantine, alert, manual review

---

**Commodity Non-Negative Check**:

```sql
-- Prices must never be negative
SELECT table_name, DATE(time), close, volume
FROM `raw_intelligence.commodity_*_daily`
WHERE close < 0 OR volume < 0
  AND DATE(time) >= CURRENT_DATE() - 7;
```

**Action on Detection**: Critical alert, halt ingestion

---

#### Check 3: Source Reliability Scoring

**Every row MUST have metadata** (as defined in `docs/reference/V14_METADATA_EXPANSION.md`):

```sql
SELECT 
  COUNT(*) as total_rows,
  SUM(CASE WHEN source_name IS NULL THEN 1 ELSE 0 END) as missing_source,
  SUM(CASE WHEN ingest_timestamp IS NULL THEN 1 ELSE 0 END) as missing_ingest,
  SUM(CASE WHEN provenance_uuid IS NULL THEN 1 ELSE 0 END) as missing_uuid
FROM `raw_intelligence.*`
WHERE DATE(time) >= CURRENT_DATE() - 7;
```

**Required**: 100% metadata coverage

**Action on Failure**: Reject ingestion, log to quarantine

---

#### Check 4: Quarantine Bad Rows

**Quarantine table** (must exist):

```sql
CREATE TABLE IF NOT EXISTS `raw_intelligence.quarantine_scrapes` (
  original_table STRING,
  quarantine_reason STRING,
  quarantine_timestamp TIMESTAMP,
  original_row JSON
) PARTITION BY DATE(quarantine_timestamp);
```

**Move violations**:

```sql
INSERT INTO `raw_intelligence.quarantine_scrapes`
SELECT 
  '{table_name}' as original_table,
  '{reason}' as quarantine_reason,
  CURRENT_TIMESTAMP() as quarantine_timestamp,
  TO_JSON_STRING(t) as original_row
FROM `raw_intelligence.{table}` t
WHERE {violation_condition};

DELETE FROM `raw_intelligence.{table}`
WHERE {violation_condition};
```

**Never overwrite production tables with bad data.**

---

### (B) Curated Layer Audit - "Aggregation Check"

**Priority**: HIGH (if aggregates are stale, models use old data)

#### Check 1: Weather Aggregates

```sql
-- Must be updated within 24 hours of raw data
SELECT 
  MAX(signal_date) as latest_aggregate,
  (SELECT MAX(DATE(time)) FROM `raw_intelligence.weather_*_daily`) as latest_raw,
  DATE_DIFF(
    (SELECT MAX(DATE(time)) FROM `raw_intelligence.weather_*_daily`),
    MAX(signal_date),
    DAY
  ) as lag_days
FROM `curated.weather_aggregates_daily`;
```

**Required**: `lag_days <= 1`

**Source**: `docs/reference/ULTIMATE_SIGNAL_ARCHITECTURE.md` (Weather Intelligence section)

**Action on Failure**: Rebuild aggregates, alert

---

#### Check 2: Geopolitical Aggregates

**Validate Trump policy scoring**:

```sql
SELECT 
  signal_date,
  trump_tariff_mentions,
  china_hostility_score,
  ice_enforcement_impact,
  lobbying_expenditure
FROM `curated.geopolitical_aggregates_daily`
WHERE signal_date >= CURRENT_DATE() - 7
  AND (
    trump_tariff_mentions IS NULL OR
    china_hostility_score IS NULL
  );
```

**Source**: `docs/reference/NEWS_SENTIMENT_ANALYSIS_GUIDE.md`

**Required**: No nulls in key fields

---

#### Check 3: Substitution Aggregates

**Validate soy:palm ratio calculation**:

```sql
SELECT 
  signal_date,
  soy_palm_ratio,
  logistics_disruption_score,
  freight_index
FROM `curated.substitution_aggregates_daily`
WHERE signal_date >= CURRENT_DATE() - 7
  AND (soy_palm_ratio IS NULL OR soy_palm_ratio <= 0);
```

**Source**: `docs/reference/DYNAMIC_SUBSTITUTION_ECONOMICS.md`

**Required**: Ratio must be positive and reasonable (0.5 - 5.0)

---

#### Check 4: Upstream Null Explosion Detection

**Most common failure mode**: null explosions after pivot/aggregation.

**Action**: For each curated table, run column-level null counts directly (no joins), e.g.:
```sql
SELECT COUNTIF(soy_palm_ratio IS NULL) AS null_soy_palm_ratio
FROM `curated.substitution_aggregates_daily`
WHERE signal_date >= CURRENT_DATE() - 7;
```
Trace back to the upstream source that feeds the pivot and fix there.

---

#### Check 5: Date Join Alignment

**All aggregates must share the same signal_date range**:

```sql
WITH date_ranges AS (
  SELECT 'weather' as source, MIN(signal_date) as min_d, MAX(signal_date) as max_d
  FROM `curated.weather_aggregates_daily`
  UNION ALL
  SELECT 'geopolitical', MIN(signal_date), MAX(signal_date)
  FROM `curated.geopolitical_aggregates_daily`
  UNION ALL
  SELECT 'substitution', MIN(signal_date), MAX(signal_date)
  FROM `curated.substitution_aggregates_daily`
)
SELECT *
FROM date_ranges
WHERE max_d != (SELECT MAX(max_d) FROM date_ranges);
```

**Required**: All aggregates end on same date (today or yesterday)

---

### (C) Feature Layer Audit - "Training Alignment Check"

**Priority**: CRITICAL (if training tables are wrong, models fail)

**This is where Cursor always screws up. Be strict.**

#### Check 1: Training Table Existence

```sql
-- All 10 production training tables must exist
SELECT table_name
FROM `training.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE 'zl_training_prod_allhistory_%'
ORDER BY table_name;
```

**Expected output**:
```
zl_training_prod_allhistory_12m
zl_training_prod_allhistory_1m
zl_training_prod_allhistory_1w
zl_training_prod_allhistory_3m
zl_training_prod_allhistory_6m
```

**Required**: Exactly 5 prod tables, 5 full tables

---

#### Check 2: Column Ordering Consistency

```sql
-- All training tables must have same column order (except horizon-specific features)
WITH column_lists AS (
  SELECT table_name, ARRAY_AGG(column_name ORDER BY ordinal_position) as cols
  FROM `training.INFORMATION_SCHEMA.COLUMNS`
  WHERE table_name LIKE 'zl_training_prod_allhistory_%'
  GROUP BY table_name
)
SELECT table_name, cols
FROM column_lists
WHERE ARRAY_LENGTH(cols) != (SELECT MAX(ARRAY_LENGTH(cols)) FROM column_lists);
```

**Required**: All tables have similar column counts (±10 columns due to horizon-specific features)

---

#### Check 3: No Null Targets

```sql
-- Critical: training tables must never have null targets
SELECT 
  table_name,
  COUNT(*) as null_target_count
FROM `training.zl_training_prod_allhistory_*`
WHERE target_return IS NULL OR target_price IS NULL
GROUP BY table_name
HAVING null_target_count > 0;
```

**Required**: Zero null targets

**Action on Failure**: Rebuild training tables, investigate label calculation

---

#### Check 4: Regime Weight Alignment

```sql
-- Every row must have a valid regime weight (no joins required)
SELECT as_of_date, regime, row_weight
FROM `training.zl_training_prod_allhistory_1m`
WHERE as_of_date >= CURRENT_DATE() - 90
  AND row_weight IS NULL;
```

```sql
-- Optional: ensure weights match canonical list
SELECT DISTINCT regime
FROM `training.zl_training_prod_allhistory_1m`
WHERE regime NOT IN (SELECT regime FROM `training.regime_weights`);
```

**Required**: 100% regime weight alignment and no missing weights

**Expected weights**: 50-5000 (documented in `scripts/migration/REGIME_WEIGHTS_RESEARCH.md`)

---

#### Check 5: No Legacy Names

```sql
-- FORBIDDEN: Any reference to old naming convention
SELECT table_name
FROM `models_v4.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE 'production_training_data_%'
  AND table_name NOT LIKE '%_shim_%';
```

**Required**: Zero matches (except shim views which are temporary)

**Action on Detection**: Alert "LEGACY NAMING DETECTED", halt migration

---

### (D) Prediction Layer Audit - "Signal Integrity Check"

**Priority**: HIGH (if predictions are stale, dashboard shows old forecasts)

#### Check 1: Prediction Files Exist Locally

**Local filesystem check**:

```bash
# Must find predictions.parquet in model directories
find Models/local/horizon_*/prod/*/*/predictions.parquet -mtime -7
```

**Required**: At least 1 prediction file per horizon (5 total minimum)

**Expected structure**:
```
Models/local/horizon_1m/prod/baselines/lightgbm_dart_v001/predictions.parquet
Models/local/horizon_1m/prod/baselines/xgboost_dart_v001/predictions.parquet
...
```

---

#### Check 2: Predictions Uploaded to BigQuery

```sql
-- Check upload recency
SELECT 
  table_name,
  MAX(created_at) as latest_upload,
  COUNT(*) as row_count
FROM `predictions.zl_*_inference_*`
GROUP BY table_name
HAVING latest_upload < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
```

**Required**: All prediction tables updated within 24 hours

**Action on Failure**: Run `scripts/upload_predictions.py`

---

#### Check 3: Views Resolve Correctly

```sql
-- All latest views must return data
SELECT 'vw_zl_1w_latest' as view_name, COUNT(*) as rows FROM `predictions.vw_zl_1w_latest`
UNION ALL
SELECT 'vw_zl_1m_latest', COUNT(*) FROM `predictions.vw_zl_1m_latest`
UNION ALL
SELECT 'vw_zl_3m_latest', COUNT(*) FROM `predictions.vw_zl_3m_latest`
UNION ALL
SELECT 'vw_zl_6m_latest', COUNT(*) FROM `predictions.vw_zl_6m_latest`
UNION ALL
SELECT 'vw_zl_12m_latest', COUNT(*) FROM `predictions.vw_zl_12m_latest`;
```

**Required**: All views return >0 rows

---

#### Check 4: Model Metadata Present

**Local filesystem check**:

```bash
# Every model directory must have these files
for model_dir in Models/local/horizon_*/prod/*/*_v*/; do
  if [[ ! -f "$model_dir/model.bin" ]] || \
     [[ ! -f "$model_dir/columns_used.txt" ]] || \
     [[ ! -f "$model_dir/run_id.txt" ]] || \
     [[ ! -f "$model_dir/feature_importance.csv" ]]; then
    echo "MISSING METADATA: $model_dir"
  fi
done
```

**Required**: All 4 metadata files present

---

#### Check 5: MAPE Recalculation

**Must run MAPE engine after any prediction upload**:

```sql
-- Source: docs/reference/FORECAST_PERFORMANCE_TRACKING.md
CALL `monitoring.sp_calculate_mape`(
  horizon => '1m',
  lookback_days => 90
);
```

**Required**: MAPE updated for all horizons after upload

**Tables updated**:
- `monitoring.mape_historical_tracking`
- `monitoring.vw_forecast_performance_tracking`

---

#### Check 6: Sharpe Recalculation

**Must run Sharpe engine after any prediction upload**:

```sql
-- Source: docs/reference/SOYBEAN_SHARPE_RATIO_GUIDE.md
CALL `monitoring.sp_calculate_sharpe`(
  asset => 'soybean_oil',
  horizon => '1m',
  lookback_days => 90
);
```

**Tables updated**:
- `monitoring.soybean_sharpe_historical_tracking`
- `monitoring.vw_soybean_sharpe_metrics`

---

#### Check 7: Regime Classification Integrity

**Ensure master regime classification matches latest data**:

```sql
-- Regime calendar must be up-to-date
SELECT 
  MAX(date) as latest_regime_date,
  CURRENT_DATE() as today,
  DATE_DIFF(CURRENT_DATE(), MAX(date), DAY) as lag_days
FROM `training.regime_calendar`;
```

**Required**: `lag_days <= 1`

**Action on Failure**: Extend regime calendar (run `scripts/migration/04_create_regime_tables.sql`)

---

## 3. HOW CURSOR MUST CONSOLIDATE (Canonical Flow)

**THE ONLY VALID FLOW**:

```
RAW → curated → training → Models/local → predictions → dashboard
```

### Forbidden Actions

❌ **Creating new datasets** (use existing 7 only)  
❌ **Scattering data** (consolidate to canonical locations)  
❌ **Bypassing raw layer** (everything starts from raw_intelligence)  
❌ **Direct dashboard writes** (must flow through predictions)

### Dataset Mapping Rules

| Data Type | MUST Go To | NOT Allowed |
|-----------|------------|-------------|
| Scraped/ingested | `raw_intelligence.*` | `staging`, `temp`, `scratch` |
| Aggregated features | `curated.*` | `features`, `signals`, `neural` |
| Training matrices | `training.*` | `models_v4`, `models`, `ml_data` |
| Model outputs | `predictions.*` | `forecasts`, `results`, `outputs` |
| Performance tracking | `monitoring.*` | `metrics`, `stats`, `analytics` |

### Canonical Dataset List (ONLY THESE)

1. `raw_intelligence` - Raw ingestion
2. `curated` - Aggregated features
3. `training` - Training matrices
4. `predictions` - Model outputs
5. `monitoring` - Performance tracking
6. `vegas_intelligence` - Sales intel (isolated)
7. `archive` - Legacy snapshots

**Source**: Defined across entire documentation ecosystem (Ultimate Signal Architecture, V14 Metadata Expansion, Critical Context, etc.)

---

## 4. CURSOR'S POST-MOVE AUDIT CHECKLIST (Execute in Order)

After **any** trigger condition from Section 0, execute this checklist:

### □ (1) Raw Layer Audit

- [ ] Validate timestamps (no gaps, no duplicates, sorted)
- [ ] Validate values (sanity checks, TE palm oil corruption)
- [ ] Validate metadata (source, ingest_timestamp, provenance_uuid)
- [ ] Quarantine bad rows (to `raw_intelligence.quarantine_scrapes`)

**Estimated Time**: 5-10 minutes  
**Failure Action**: Halt downstream processing, alert

---

### □ (2) Curated Layer Audit

- [ ] Rebuild weather aggregates (if raw data updated)
- [ ] Rebuild geopolitical aggregates
- [ ] Rebuild substitution aggregates
- [ ] Rebuild technical aggregates
- [ ] Check alignment (all aggregates end on same date)
- [ ] Ensure no null propagation

**Estimated Time**: 10-15 minutes  
**Failure Action**: Rerun aggregation jobs, fix join logic

---

### □ (3) Training Layer Audit

- [ ] Validate feature completeness (all expected columns present)
- [ ] Validate regime weights (50-5000 scale, 100% merge)
- [ ] Validate targets (no nulls in target columns)
- [ ] Ensure all training tables updated (prod + full, all horizons)
- [ ] Check column ordering consistency

**Estimated Time**: 5-10 minutes  
**Failure Action**: Rebuild training tables from ULTIMATE_DATA_CONSOLIDATION.sql

---

### □ (4) Prediction Layer Audit

- [ ] Validate local prediction files exist
- [ ] Upload predictions to BigQuery (`scripts/upload_predictions.py`)
- [ ] Update views (`vw_zl_{h}_latest`)
- [ ] Recompute MAPE (`monitoring.sp_calculate_mape`)
- [ ] Recompute Sharpe (`monitoring.sp_calculate_sharpe`)
- [ ] Validate model metadata (model.bin, columns_used.txt, run_id.txt, feature_importance.csv)

**Estimated Time**: 15-20 minutes  
**Failure Action**: Regenerate predictions locally, rerun upload

---

### □ (5) Dashboard Audit

- [ ] Ensure all views resolve (vw_zl_*_latest)
- [ ] Ensure signal metrics up-to-date
- [ ] Ensure regime classification matches latest data
- [ ] Verify MAPE/Sharpe display correctly
- [ ] Test API endpoints (`/api/forecast/{horizon}`)

**Estimated Time**: 5 minutes  
**Failure Action**: Restart Next.js server, check API logs

---

## 5. ENFORCEMENT POLICY (Machine-Readable Spec)

### Audit Execution Rules

**Trigger Detection**:
```yaml
triggers:
  vix_spike:
    threshold: 25
    check_frequency: "every 15 minutes"
    source_table: "raw_intelligence.vol_vix_daily"
  
  fx_move:
    threshold: 0.03  # 3%
    check_frequency: "every hour"
    source_table: "raw_intelligence.fx_usd_brl_daily"
  
  zl_move:
    threshold: 0.02  # 2%
    check_frequency: "real-time"
    source_table: "raw_intelligence.commodity_soybean_oil_daily"
```

**Audit Sequence**:
```yaml
audit_sequence:
  - name: "Raw Layer Integrity"
    priority: "CRITICAL"
    timeout: "10 minutes"
    failure_action: "halt_downstream"
    
  - name: "Curated Layer Alignment"
    priority: "HIGH"
    timeout: "15 minutes"
    failure_action: "rebuild_aggregates"
    
  - name: "Training Layer Validation"
    priority: "CRITICAL"
    timeout: "10 minutes"
    failure_action: "rebuild_training_tables"
    
  - name: "Prediction Layer Upload"
    priority: "HIGH"
    timeout: "20 minutes"
    failure_action: "regenerate_predictions"
```

**Dataset Boundaries**:
```yaml
allowed_datasets:
  - raw_intelligence
  - curated
  - training
  - predictions
  - monitoring
  - vegas_intelligence
  - archive

forbidden_actions:
  - create_new_dataset
  - write_to_legacy: ["models_v4", "signals", "neural", "forecasting_data_warehouse"]
  - bypass_raw_layer: true
  - skip_validation: true
```

**Validation Thresholds**:
```yaml
validation_rules:
  timestamp_gaps:
    max_allowed: 0
    severity: "CRITICAL"
  
  null_targets:
    max_allowed: 0
    severity: "CRITICAL"
  
  regime_weight_mismatch:
    max_allowed: 0
    severity: "CRITICAL"
  
  prediction_staleness:
    max_age_hours: 24
    severity: "HIGH"
  
  aggregate_lag:
    max_days: 1
    severity: "HIGH"
```

---

## 6. AUTOMATIC REMEDIATION (If Possible)

### Auto-Fixable Issues

| Issue | Detection | Auto-Fix Command |
|-------|-----------|------------------|
| Stale aggregates | Lag > 1 day | `CALL curated.sp_rebuild_aggregates()` |
| Missing predictions | Upload > 24h old | `python scripts/upload_predictions.py` |
| Regime calendar outdated | Lag > 1 day | `bq query < scripts/migration/04_create_regime_tables.sql` |
| MAPE not updated | After prediction upload | `CALL monitoring.sp_calculate_mape()` |

### Manual Intervention Required

| Issue | Detection | Action |
|-------|-----------|--------|
| TE palm oil corruption | `close = 1021.0` | Quarantine, backfill from alternative |
| Null explosion | Curated layer nulls | Trace to raw layer, fix join |
| Timestamp gaps | Missing dates | Investigate ingestion failure |
| Training table corruption | Inconsistent columns | Rebuild from ULTIMATE_DATA_CONSOLIDATION |

---

## 7. LOGGING & MONITORING

### Required Log Entries

Every audit must write to `monitoring.data_quality_events`:

```sql
INSERT INTO `monitoring.data_quality_events` (
  event_timestamp,
  event_type,
  severity,
  dataset_name,
  table_name,
  check_name,
  status,
  details
) VALUES (
  CURRENT_TIMESTAMP(),
  'POST_MOVE_AUDIT',
  'INFO',  -- or 'WARNING', 'CRITICAL'
  'raw_intelligence',
  'commodity_soybean_oil_daily',
  'timestamp_gaps',
  'PASS',
  '{"gaps_found": 0, "rows_checked": 1234}'
);
```

### Alert Conditions

**CRITICAL** (page on-call):
- Timestamp gaps in raw layer
- Null targets in training layer
- Regime weight mismatch
- Quarantine rate > 5%

**HIGH** (email alert):
- Aggregate lag > 1 day
- Prediction staleness > 24 hours
- MAPE not updated after upload

**INFO** (log only):
- Successful audit completion
- Auto-remediation success

---

## CONCLUSION

This instruction set is **binding** for all AI assistants (Cursor, GPT, Claude, etc.).

**The Principle**: After every major market move, the entire data pipeline must be audited and validated before predictions are trusted.

**The Discipline**: Follow the checklist in order. Never skip steps. Never create new datasets.

**The Edge**: Systems that don't audit post-move get wrecked by stale data and corrupt signals.

---

**Last Updated**: November 14, 2025  
**Version**: 1.0  
**Status**: MANDATORY COMPLIANCE  
**Next**: Implement automated trigger detection + audit execution
