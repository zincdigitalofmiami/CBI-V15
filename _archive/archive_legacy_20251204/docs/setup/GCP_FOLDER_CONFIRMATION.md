# GCP Folder Location Confirmation - CBI-V15

**Date**: November 28, 2025  
**Question**: Should CBI-V15 be created under "App Development" folder in GCP?

---

## Current GCP Organization

### Folders Available:
- ✅ **App Development** (ID: `568609080192`)
- IT-Dev
- Proof of Concept
- Web-Dev
- Shared
- Human-Asset
- system-gsuite
- gcp-internal-cloud-setup

### Existing Projects:
- `cbi-v15` - Current project (need to check its folder)
- `google-mpf-568609080192` - Already under "App Development" folder

---

## Analysis: Is "App Development" the Right Location?

### CBI-V15 Project Type:
- **Primary Function**: Data Science / ML Platform
- **Components**:
  - ✅ BigQuery data warehouse
  - ✅ Dataform ETL pipelines
  - ✅ ML model training (Mac M4)
  - ✅ Forecasting engine
  - ⚠️ Dashboard (Next.js) - secondary component

### "App Development" Folder Purpose:
- Typically for: Web apps, mobile apps, application development
- CBI-V15 is more: Data pipeline, ML platform, forecasting system

---

## Recommendation

### Option 1: Under "App Development" ✅ (If that's your standard)
**Pros**:
- Consistent with your organization structure
- Easy to find with other projects
- If you categorize all projects there, it's fine

**Cons**:
- Not technically "app development" (more data science)
- Might be confusing later

**Action**: Create project under App Development folder

### Option 2: Standalone (No folder) ⚠️
**Pros**:
- Simple, direct
- No organizational overhead

**Cons**:
- Less organized
- Harder to manage with many projects

**Action**: Create project without folder assignment

### Option 3: Create "Data Science" Folder (Future)
**Pros**:
- Better categorization
- More accurate description

**Cons**:
- Requires creating new folder
- More setup

**Action**: Create new folder, then create project

---

## Decision Needed

**Please confirm**:
1. ✅ **YES** - Create CBI-V15 under "App Development" folder (`568609080192`)
2. ⚠️ **NO** - Create CBI-V15 as standalone (no folder)
3. 🔄 **OTHER** - Create under different folder (specify which)

---

## If YES (App Development Folder):

**Updated project creation command**:
```bash
gcloud projects create cbi-v15 \
  --name="CBI-V15 Soybean Oil Forecasting" \
  --folder=568609080192
```

**Or move existing project**:
```bash
gcloud projects move cbi-v15 --folder=568609080192
```

---

## Current Status

- ✅ Folder "App Development" exists: `568609080192`
- ⚠️ Project `cbi-v15` may not exist yet (or not accessible)
- ⚠️ Need confirmation on folder placement

---

**Please confirm**: Should CBI-V15 be created under "App Development" folder?

---

**Last Updated**: November 28, 2025

