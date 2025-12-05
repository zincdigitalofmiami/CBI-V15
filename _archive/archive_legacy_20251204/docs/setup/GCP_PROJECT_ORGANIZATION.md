# GCP Project Organization - CBI-V15

**Date**: November 28, 2025  
**Question**: Should CBI-V15 be organized under "App Development" folder in GCP?

---

## Current GCP Project Structure

### Project: `cbi-v15`
- **Location**: `us-central1` ONLY (critical - no multi-region)
- **Purpose**: Soybean Oil Forecasting Platform (ZL futures)
- **Type**: Data Science / ML Platform

---

## GCP Folder Organization Options

### Option 1: Standalone Project (Current)
- **Project**: `cbi-v15`
- **No folder**: Directly under organization
- **Pros**: Simple, direct access
- **Cons**: No organizational grouping

### Option 2: Under "App Development" Folder
- **Folder**: `App Development` (or similar)
- **Project**: `cbi-v15` under this folder
- **Pros**: Organized with other app development projects
- **Cons**: May not fit if this is more "Data Science" than "App Development"

### Option 3: Under "Data Science" or "ML" Folder
- **Folder**: `Data Science` or `Machine Learning`
- **Project**: `cbi-v15` under this folder
- **Pros**: Better categorization (this is ML/data science)
- **Cons**: Requires creating folder structure

---

## Recommendation

**This project is primarily a Data Science/ML platform**, not a traditional "App Development" project:

- ✅ **Data Science Focus**: ML models, feature engineering, forecasting
- ✅ **Data Pipeline**: ETL, BigQuery, Dataform
- ✅ **ML Training**: Mac M4 training, model inference
- ⚠️ **Dashboard**: Secondary (Next.js dashboard is just visualization)

**Suggested Organization**:
- **Folder**: `Data Science` or `ML Platforms` (if folders exist)
- **OR**: Keep standalone if no folder structure exists

---

## BigQuery Dataset Organization (Within Project)

**Current Structure** (8 datasets):
- `raw` - Source data
- `staging` - Staged data
- `features` - Engineered features
- `training` - Training data
- `forecasts` - Predictions
- `api` - Dashboard views
- `reference` - Reference tables
- `ops` - Operations monitoring

**This structure is correct** - no changes needed.

---

## Action Required

**If you want CBI-V15 under "App Development" folder**:

1. **Create folder in GCP** (if it doesn't exist):
   ```bash
   gcloud resource-manager folders create \
     --display-name="App Development" \
     --organization=YOUR_ORG_ID
   ```

2. **Move project to folder**:
   ```bash
   gcloud projects move cbi-v15 \
     --folder=FOLDER_ID
   ```

**If keeping standalone**: No action needed - current structure is fine.

---

## Verification

Check current project organization:
```bash
gcloud projects describe cbi-v15 --format="get(parent)"
```

If output is empty, project is standalone (no folder).

---

**Last Updated**: November 28, 2025
