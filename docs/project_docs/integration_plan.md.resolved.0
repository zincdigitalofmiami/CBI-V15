# Integration Plan: Legacy Schema + AnoFox + UI Redesign

## Phase 1: Preserve Legacy Work (4 Months of Development)

### What NOT to Lose
From CBI-V15, we have:
- **Regime Classification System** (buckets)
- **276 Engineered Features**
- **Big 8 Drivers** with correlations
- **BigQuery Schema** (8 datasets, 29+ tables)
- **Feature Engineering Logic** (Python-first pipeline)

### DuckDB Local Setup (External Drive)

**Location**: `/Volumes/Satechi Hub/ZL-Intelligence/duckdb/`

**Migration Plan**:
1. Export all BigQuery tables → Parquet
2. Load Parquet → Local DuckDB
3. Preserve all schemas, partitions, indexes

### Schema to Preserve
- `raw.*` - Source data
- `staging.*` - Normalized data
- `features.*` - 276 engineered features
- `training.*` - ML-ready matrices
- `forecasts.*` - Model outputs
- `reference.*` - Regimes, buckets, splits

## Phase 2: Learn AnoFox Modeling

### What is AnoFox?
Need to research:
- Model architectures they support
- How they handle regime detection
- Their feature engineering vs our 276 features
- Integration points

### Possible Integration Strategies

**Option A: Use AnoFox as Training Engine**
- Keep our feature engineering
- Feed AnoFox our `training.daily_ml_matrix`
- AnoFox outputs forecasts
- We add our regime/bucket context

**Option B: Hybrid Approach**
- AnoFox handles baseline forecasts
- Our regime engine adjusts forecasts per bucket
- Combine SHAP values from both

**Option C: Full Replacement**
- Migrate our features into AnoFox format
- Let AnoFox handle everything
- Risk: Lose our bucket intelligence

**Recommendation**: Start with Option A

## Phase 3: Two-Tier Admin Structure

### Page 1: Admin (General Operations)
**Route**: `/admin`
**Purpose**: Data uploads, system health, refresh buttons
**Users**: Operations team

### Page 2: Quant Admin (Hidden)
**Route**: `/quant-admin` (not in main navigation)
**Purpose**: Your quantitative work, AnoFox results viewer
**Users**: You only
**Features**:
- Model performance metrics
- SHAP explainability
- Regime bucket analysis
- Feature importance over time
- AnoFox experiment results

## Phase 4: UI Redesign

### Design Specifications (from images)

**Colors**:
```css
--background: rgb(0, 0, 0);
--foreground: rgb(255, 255, 255);
--card-bg: rgba(30, 30, 35, 0.5);
--border: rgba(255, 255, 255, 0.1);
```

**Typography**:
- **Headers**: Ultra-thin weight (100-200)
- **Font**: System fonts or Inter/Manrope Thin
- **Body**: Light weight (300)

**Gauges**:
- TradingView-style circular gauges
- Thin arc strokes (2-3px)
- Colored segments (not full circles)
- SHAP Impact values below
- Dark gray unfilled portion

**Components to Update**:
1. All headers → thin fonts
2. Background → pure black
3. Cards → minimal borders
4. Gauges → TradingView style
5. Navigation → thin text

## Next Steps

1. **Immediate**: Research AnoFox documentation
2. **Then**: Map CBI-V15 schema to DuckDB local
3. **Then**: Design AnoFox integration strategy
4. **Then**: UI redesign (fonts, colors, gauges)
5. **Finally**: Create Quant Admin page
