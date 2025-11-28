# CBI-V15: Soybean Oil Forecasting Platform

[![CI/CD](https://github.com/zincdigitalofmiami/cbi-v15/workflows/Dataform/badge.svg)](https://github.com/zincdigitalofmiami/cbi-v15/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Institutional-grade ZL (soybean oil futures) price forecasting using Dataform ETL, Mac M4 training, and BigQuery storage.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for Dataform)
- Google Cloud SDK
- macOS Keychain (for API keys)
- GCP project: `cbi-v15` (us-central1)

### Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/zincdigitalofmiami/cbi-v15.git
   cd cbi-v15
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add API keys to macOS Keychain (see docs/setup/API_KEYS.md)
   ```

4. **Initialize Dataform**
   ```bash
   cd dataform
   npm install -g @dataform/cli
   npm install
   dataform init
   dataform compile
   ```

5. **Verify Connections**
   ```bash
   python scripts/setup/verify_connections.py
   ```

6. **Run Data Quality Checks**
   ```bash
   python scripts/validation/data_quality_checks.py
   ```

---

## Architecture

- **ETL**: Dataform (BigQuery transformations, version controlled)
- **Training**: Mac M4 (PyTorch MPS + LightGBM, 100% local)
- **Storage**: BigQuery (us-central1 only)
- **Dashboard**: Next.js/Vercel

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## Key Features

- **Big 8 Drivers**: Complete coverage of primary ZL drivers (Crush Margin, China Imports, Dollar, Fed, Tariffs, Biofuels, Crude, VIX)
- **Weather Intelligence**: Multi-source weather aggregation (NOAA, INMET, Argentina SMN, Google Public Datasets)
- **Political Intelligence**: FEC, GDELT, Trump policy signals, silence detection
- **Sentiment Analysis**: FinBERT-powered news sentiment (biofuel, China, tariffs buckets)
- **Data Quality**: Automated assertions and monitoring
- **Technical Indicators**: Comprehensive library (pandas-ta, 100+ indicators)

---

## Documentation

- [Architecture](docs/architecture/) - System design and data flow
- [Data Sources](docs/data-sources/) - API references and schemas
- [Features](docs/features/) - Feature engineering documentation
- [Training](docs/training/) - Model training guides
- [AI Assistant Guide](docs/reference/AI_ASSISTANT_GUIDE.md) - Quick start for AI assistants
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

---

## Project Structure

```
CBI-V15/
├── dataform/          # BigQuery Dataform (primary ETL)
├── src/               # Python source code
├── scripts/           # Operational scripts
├── config/            # Configuration files
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── dashboard/         # Next.js dashboard
```

See [docs/architecture/MASTER_PLAN.md](docs/architecture/MASTER_PLAN.md) for complete structure.

---

## Workflow

1. **Data Collection**: Python scripts → BigQuery (via Dataform)
2. **ETL**: Dataform transforms (raw → staging → features → training)
3. **Export**: Export training data to Parquet
4. **Train**: Mac M4 training (LightGBM, PyTorch/TFT)
5. **Predict**: Generate forecasts locally
6. **Upload**: Upload predictions to BigQuery
7. **Dashboard**: Vercel reads BigQuery views

---

## Data Sources

- **Market**: Databento (futures, FX, options)
- **Economic**: FRED (55-60 series)
- **Weather**: NOAA, INMET, Argentina SMN, Google Public Datasets
- **Agricultural**: USDA (WASDE, crop progress, exports)
- **Positioning**: CFTC (COT)
- **Energy**: EIA (biofuels, RINs)
- **News/Policy**: ScrapeCreators (Trump + buckets), GDELT
- **Political**: FEC (contributions, PACs)
- **Vegas Intel**: Glide API

---

## Critical Rules

1. **NO FAKE DATA** - Only real, verified data
2. **us-central1 ONLY** - All GCP resources
3. **NO COSTLY RESOURCES** - Approval required >$5/month
4. **API KEYS** - Keychain (Mac) or Secret Manager (GCP)
5. **Dataform First** - All ETL in Dataform
6. **Mac Training Only** - No cloud training

See [docs/reference/BEST_PRACTICES.md](docs/reference/BEST_PRACTICES.md) for complete guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Support

- Issues: [GitHub Issues](https://github.com/zincdigitalofmiami/cbi-v15/issues)
- Documentation: [docs/](docs/)
- Security: [SECURITY.md](SECURITY.md)

---

**Last Updated**: November 28, 2025

