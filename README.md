# CBI-V15: Soybean Oil Forecasting Platform

**Institutional-grade ZL (soybean oil futures) price forecasting using Dataform ETL, Mac M4 training, and BigQuery storage.**

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Node.js (for Dataform)
- Google Cloud SDK (`gcloud`)
- macOS Keychain access (for API keys)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
cd dataform && npm install
```

### 3. Verify Setup
```bash
# Test BigQuery connection
python3 scripts/ingestion/test_connections.py

# Verify BigQuery structure
python3 scripts/setup/verify_bigquery_setup.py
```

### 4. Store API Keys
```bash
./scripts/setup/store_api_keys.sh
```

### 5. Connect Dataform to GitHub
1. Go to Google Cloud Console → Dataform
2. Click "Connect Repository"
3. Repository: `zincdigitalofmiami/CBI-V15`
4. Branch: `main`
5. **Root Directory: `dataform/`** ⚠️ Critical
6. Click "Connect"

### 6. First Data Ingestion
```bash
python3 src/ingestion/databento/collect_daily.py
```

### 7. Run Dataform Transformations
```bash
cd dataform
npx dataform compile  # Verify
npx dataform run --tags staging  # Build staging tables
npx dataform run --tags features  # Build feature tables
npx dataform test  # Run assertions
```

---

## 📁 Project Structure

```
CBI-V15/
├── dataform/              # BigQuery ETL (Dataform)
│   ├── definitions/      # SQL transformations
│   │   ├── 01_raw/      # Source declarations
│   │   ├── 02_staging/  # Cleaned data
│   │   ├── 03_features/ # Engineered features
│   │   ├── 04_training/ # Training tables
│   │   ├── 05_assertions/# Data quality gates
│   │   └── 06_api/      # Public views
│   └── includes/        # Shared SQL functions
│
├── src/                  # Python source code
│   ├── ingestion/       # Data collection scripts
│   ├── training/         # Model training (Mac M4)
│   ├── features/        # Feature engineering
│   └── utils/           # Utilities (Keychain, BigQuery)
│
├── scripts/              # Operational scripts
│   ├── setup/           # Setup scripts
│   ├── ingestion/       # Ingestion helpers
│   └── export/          # Data export scripts
│
└── config/              # Configuration files
    └── schedulers/      # Cloud Scheduler configs
```

---

## 🏗️ Architecture

### Data Flow
```
External APIs → Raw Layer → Staging Layer → Features Layer → Training Layer
     ↓              ↓             ↓              ↓               ↓
  Databento    BigQuery      Cleaned      Engineered      ML-Ready
  FRED         (raw)         (staging)    (features)      (training)
  ScrapeCreators
  USDA/CFTC/EIA
```

### Training Flow
```
Training Data (Parquet) → Mac M4 Training → Models → Predictions → BigQuery
```

---

## 📊 BigQuery Datasets

| Dataset | Purpose | Location |
|---------|---------|----------|
| `raw` | Source data declarations | `us-central1` |
| `staging` | Cleaned, normalized data | `us-central1` |
| `features` | Engineered features (276 total) | `us-central1` |
| `training` | Training-ready tables | `us-central1` |
| `forecasts` | Model predictions | `us-central1` |
| `api` | Public API views | `us-central1` |
| `reference` | Reference data (regimes, splits) | `us-central1` |
| `ops` | Operations monitoring | `us-central1` |

---

## 🔑 Key Features

### Big 8 Drivers
1. **Crush Margin** (0.961 correlation - #1!)
2. **China Imports** (-0.813 correlation)
3. **Dollar Index** (-0.658)
4. **Fed Policy** (-0.656)
5. **Tariffs** (0.647)
6. **Biofuels** (-0.601)
7. **Crude Oil** (0.584)
8. **VIX** (0.398)

### 276 Features Total
- Technical indicators (19)
- FX indicators (16)
- Cross-asset correlations (112)
- Cross-asset betas (28)
- Lagged features (96)
- Fundamental spreads (4)
- News sentiment (bucket-specific)
- Weather anomalies
- CFTC positioning
- And more...

---

## 🧪 Testing

### Connection Test
```bash
python3 scripts/ingestion/test_connections.py
```

### Dataform Compilation
```bash
cd dataform
npx dataform compile
```

### Data Quality Assertions
```bash
cd dataform
npx dataform test
```

---

## 📚 Documentation

- **[NEXT_ACTIONS.md](NEXT_ACTIONS.md)** - Complete checklist
- **[READY_FOR_PRODUCTION.md](READY_FOR_PRODUCTION.md)** - Production readiness
- **[SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)** - Setup verification
- **[DATAFORM_REPO_CONNECTION.md](DATAFORM_REPO_CONNECTION.md)** - Dataform setup

---

## 🔧 Configuration

### Environment Variables
- API keys stored in macOS Keychain (local scripts)
- API keys stored in GCP Secret Manager (Cloud Scheduler)

### Dataform Variables
See `dataform/dataform.json` for configuration:
- Project: `cbi-v15`
- Location: `us-central1`
- Datasets: `raw`, `staging`, `features`, `training`, etc.

---

## 🎯 Current Status

**Infrastructure**: ✅ 100% Complete  
**Dataform**: ✅ Ready (needs GitHub connection)  
**Code**: ✅ Ready  
**Documentation**: ✅ Complete  

**Next Steps:**
1. Connect Dataform to GitHub (UI)
2. Store API keys
3. Begin data ingestion
4. Run Dataform transformations
5. Train baseline models

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Last Updated**: November 28, 2025
