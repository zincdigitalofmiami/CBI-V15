# config / System Configuration

## Purpose
Configuration files for all system components. **YAML/JSON only, no code.**

## Directory Structure

```
config/
├── README.md                 # This file
├── data_sources.yaml         # Master data source definitions (223 lines)
├── codex/                    # AI/LLM configuration
├── env-templates/            # Environment variable templates
├── ingestion/                # Data ingestion configuration
│   ├── README.md
│   └── sources.yaml          # API endpoints, rate limits, schedules
├── requirements/             # Python dependencies
│   ├── requirements.txt      # Production dependencies
│   └── requirements-research.txt  # Research/notebook dependencies
├── schedulers/               # Job scheduling
│   └── ingestion_schedules.yaml
└── training/                 # Model training configuration
    ├── README.md
    └── model_config.yaml     # Horizons, hyperparameters, splits
```

## Key Files

### data_sources.yaml (Master)
All external APIs and endpoints:
- **Market Data**: Databento, Polygon
- **Economic**: FRED, Treasury, BLS, ECB, BCB, PBOC
- **Commodities**: USDA, EIA
- **News/Sentiment**: ScrapeCreators, Glide

### ingestion/sources.yaml
Runtime configuration for data collectors:
```yaml
databento:
  api_key_env: DATABENTO_API_KEY
  rate_limit: 1000
  symbols:
    primary: [ZL]           # Hourly
    secondary: [ZS, ZM, CL, HO, FCPO]  # Every 4 hours
```

### training/model_config.yaml
Model training parameters:
```yaml
horizons:
  - name: 1w   # 5 trading days
  - name: 1m   # 20 trading days
  - name: 3m   # 60 trading days
  - name: 6m   # 120 trading days

models:
  lightgbm:
    num_leaves: 31
    learning_rate: 0.05
```

## What Belongs Here
| ✅ Belongs | ❌ Does NOT Belong |
|-----------|-------------------|
| API endpoints | Python scripts → `src/` |
| Rate limits | SQL files → `database/` |
| Model hyperparameters | Secrets → `.env` |
| Schedule definitions | Actual schedulers → Trigger.dev |
| Feature flags | Business logic |

## Environment Variables
Config files reference env vars, never hardcode secrets:
```yaml
# ✅ CORRECT
api_key_env: DATABENTO_API_KEY

# ❌ WRONG - Never do this
api_key: "db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE"
```

## Naming Convention
- Files: `{purpose}.yaml` or `{purpose}.json`
- Folders: lowercase, hyphenated if multi-word

## Related Files
- `.env` - Actual secret values (gitignored)
- `config/env-templates/` - Template `.env` files for setup


