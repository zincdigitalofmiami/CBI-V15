# CBI-V15 Best Practices

**Date:** November 28, 2025  
**Status:** Production Standards

---

## CRITICAL RULES (Must Always Follow)

### Data Quality
1. **NO FAKE DATA** - NEVER use placeholders, synthetic data, or fake values
2. **ALWAYS CHECK BEFORE CREATING** - Verify tables/datasets/files exist before creating
3. **ALWAYS AUDIT AFTER WORK** - Run data quality checks after any data modification

### Cost & Resource Management
4. **us-central1 ONLY** - ALL BigQuery datasets, GCS buckets, GCP resources MUST be in us-central1
5. **NO COSTLY RESOURCES WITHOUT APPROVAL** - Do NOT create paid GCP resources without explicit approval (>$5/month)

### Research & Validation
6. **RESEARCH BEST PRACTICES** - ALWAYS research online for best practices before implementing
7. **RESEARCH QUANT FINANCE** - For modeling features, research quant finance best practices

### Security
8. **API KEYS** - macOS Keychain (Mac) or Secret Manager (GCP scheduler), NEVER hardcoded
9. **Configuration** - YAML/JSON files, environment variables, never hardcoded

### Architecture
10. **Dataform First** - All ETL transformations in Dataform, version controlled
11. **Mac Training Only** - All training on Mac M4, no cloud training
12. **Source Prefixing** - All columns prefixed with source (`databento_`, `fred_`, etc.)

---

## HIGH PRIORITY (Should Always Follow)

### Pre-Work Validation
- Check existing resources before creating/modifying
- Validate naming conventions (`{asset}_{function}_{scope}_{regime}_{horizon}`)
- Verify schema compatibility before merging/joining

### Post-Work Validation
- Run data quality checks (`scripts/validation/data_quality_checks.py`)
- Test queries/scripts before declaring success
- Validate BigQuery views/tables are accessible
- Run Dataform compile (`cd dataform && dataform compile`)

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
- Review costs (monthly GCP billing audits)
- Update documentation (when code changes)

---

**Last Updated**: November 28, 2025

