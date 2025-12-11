# CBI-V15 Best Practices

**Date:** December 9, 2025  
**Status:** Production Standards

---

## CRITICAL RULES (Must Always Follow)

### Data Quality
1. **NO FAKE DATA** - NEVER use placeholders, synthetic data, or fake values
2. **ALWAYS CHECK BEFORE CREATING** - Verify tables/datasets/files exist before creating
3. **ALWAYS AUDIT AFTER WORK** - Run data quality checks after any data modification

### Cost & Resource Management
4. **NO COSTLY RESOURCES WITHOUT APPROVAL** - Do NOT create paid resources without explicit approval (>$5/month)

### Research & Validation
5. **RESEARCH BEST PRACTICES** - ALWAYS research online for best practices before implementing
6. **RESEARCH QUANT FINANCE** - For modeling features, research quant finance best practices

### Security
7. **API KEYS** - macOS Keychain or `.env` file, NEVER hardcoded
8. **Configuration** - YAML/JSON files, environment variables, never hardcoded

### Architecture
9. **DuckDB/MotherDuck First** - All ETL in SQL macros, version controlled in `database/macros/`
10. **Mac Training Only** - All training on Mac M4, no cloud training
11. **Source Prefixing** - All columns prefixed with source (`databento_`, `fred_`, etc.)
12. **Close Prices Only** - Do not reference Open/High/Low/Volume for price features

---

## HIGH PRIORITY (Should Always Follow)

### Pre-Work Validation
- Check existing resources before creating/modifying
- Validate naming conventions (`{source}_{symbol}_{indicator}_{param}_{transform}`)
- Verify schema compatibility before merging/joining

### Post-Work Validation
- Run data quality checks
- Test queries/scripts before declaring success
- Run: `python scripts/setup_database.py --both`
- Run: `bash scripts/system_status.sh`

### Code Quality
- Test all code before committing
- Document complex logic (explain why, not just what)
- Follow naming conventions (source prefixes)
- No hardcoded values (use config/env variables)

---

## MEDIUM PRIORITY (Best Practices)

### Data Engineering
- Idempotent pipelines (safe to re-run)
- Preserve source data (never modify raw layer)
- Validate transformations (test with known inputs/outputs)

### Model Development
- Pre-training validation (run data quality checks)
- Local training only (Mac M4, NOT cloud)
- Post-training validation (evaluate on holdout set)
- Save metadata (version, hyperparameters, performance metrics)

### Integration & Deployment
- Pre-integration checks (run audit framework)
- Test in staging before production
- Rollback planning (always have rollback plan)

### Monitoring & Maintenance
- Monitor data quality (daily automated checks)
- Clean up resources (temporary files, test data)
- Update documentation (when code changes)

---

## Naming Conventions

### Volatility vs Volume (CRITICAL)
| Concept | Pattern | Examples | NEVER USE |
|---------|---------|----------|-----------|
| **Volatility** (price variance) | `volatility_*` | `volatility_zl_21d`, `volatility_bucket_score` | `vol_*` alone |
| **Volume** (trading activity) | `volume_*` | `volume_zl_21d`, `open_interest_zl` | `vol_*` alone |

### Feature Naming Pattern
`{source}_{symbol}_{indicator}_{param}_{transform}`
- ✅ `databento_zl_close`, `volatility_vix_close`, `cftc_zl_managed_money_net_pct`
- ❌ `vol_zl_21d` (ambiguous), `volat_regime` (inconsistent)

---

## Workflow Checklist

### Before Starting Work
- [ ] Read `docs/architecture/MASTER_PLAN.md`
- [ ] Check existing resources (tables, datasets, files)
- [ ] Research best practices for the task
- [ ] Verify naming conventions

### During Work
- [ ] Follow existing patterns in codebase
- [ ] Use source prefixes for columns
- [ ] Document complex logic
- [ ] Test code as you write

### After Work
- [ ] Run data quality checks
- [ ] Audit for errors (nulls, duplicates, gaps)
- [ ] Test queries/scripts
- [ ] Update documentation

---

**Last Updated**: December 9, 2025
