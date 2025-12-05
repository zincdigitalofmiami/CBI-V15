# IAM Permissions Guide - CBI-V15

**Date**: November 28, 2025  
**Folder**: App Development (`568609080192`)  
**Project**: `cbi-v15`

---

## Overview

CBI-V15 requires specific IAM permissions at multiple levels:
1. **Folder-level** (App Development folder)
2. **Project-level** (cbi-v15 project)
3. **Dataset-level** (BigQuery datasets)
4. **Service Account** permissions

---

## Service Accounts

### 1. Dataform ETL Service Account
- **Name**: `cbi-v15-dataform@cbi-v15.iam.gserviceaccount.com`
- **Purpose**: Dataform ETL, Cloud Scheduler
- **Permissions**:
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - `roles/secretmanager.secretAccessor`
  - `roles/dataform.worker`

### 2. Cloud Functions Service Account
- **Name**: `cbi-v15-functions@cbi-v15.iam.gserviceaccount.com`
- **Purpose**: Data ingestion (Cloud Functions)
- **Permissions**:
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - `roles/secretmanager.secretAccessor`

### 3. Cloud Run Service Account
- **Name**: `cbi-v15-run@cbi-v15.iam.gserviceaccount.com`
- **Purpose**: Dashboard (Cloud Run)
- **Permissions**:
  - `roles/bigquery.dataViewer` (read-only)
  - `roles/bigquery.jobUser`

---

## Folder-Level Permissions (App Development)

**Required for**: Project visibility, folder-level BigQuery access

**Who can grant**: Organization Admin

**Required Roles**:
- `roles/resourcemanager.folderViewer` - View projects in folder
- `roles/bigquery.admin` (optional) - If folder-level BigQuery access needed

**How to grant** (Org Admin):
```bash
# Grant folder viewer to user
gcloud resource-manager folders add-iam-policy-binding 568609080192 \
  --member="user:YOUR_EMAIL@domain.com" \
  --role="roles/resourcemanager.folderViewer"

# Grant folder BigQuery admin (if needed)
gcloud resource-manager folders add-iam-policy-binding 568609080192 \
  --member="user:YOUR_EMAIL@domain.com" \
  --role="roles/bigquery.admin"
```

---

## Project-Level Permissions

**Automated via**: `scripts/setup/setup_iam_permissions.sh`

**What it does**:
- Creates 3 service accounts
- Grants project-level IAM roles
- Sets up BigQuery dataset permissions
- Configures Cloud Scheduler permissions

---

## Dataset-Level Permissions (BigQuery)

**Automated via**: `scripts/setup/setup_iam_permissions.sh`

**Datasets** (8 total):
- `raw` - Dataform SA: Editor, Functions SA: Editor
- `staging` - Dataform SA: Editor, Functions SA: Editor
- `features` - Dataform SA: Editor, Functions SA: Editor
- `training` - Dataform SA: Editor, Functions SA: Editor
- `forecasts` - Dataform SA: Editor, Functions SA: Editor
- `api` - Dataform SA: Editor, Functions SA: Editor, Run SA: Viewer
- `reference` - Dataform SA: Editor, Functions SA: Editor
- `ops` - Dataform SA: Editor, Functions SA: Editor

---

## Setup Instructions

### Step 1: Run IAM Setup Script

```bash
cd /Volumes/Satechi Hub/CBI-V15
./scripts/setup/setup_iam_permissions.sh
```

**What it does**:
- ✅ Creates 3 service accounts
- ✅ Grants project-level permissions
- ✅ Sets up dataset-level permissions
- ⚠️  Notes folder-level permissions (requires org admin)

### Step 2: Request Folder-Level Permissions

**If you don't have folder access**, request from org admin:

```
Subject: IAM Permission Request - App Development Folder

I need the following permissions for CBI-V15 project:
- roles/resourcemanager.folderViewer (for App Development folder: 568609080192)
- roles/bigquery.admin (optional, if folder-level BigQuery access needed)

Project: cbi-v15
Folder: App Development (568609080192)
User: YOUR_EMAIL@domain.com
```

### Step 3: Verify Permissions

```bash
# Check project-level permissions
gcloud projects get-iam-policy cbi-v15

# Check folder-level permissions (if you have access)
gcloud resource-manager folders get-iam-policy 568609080192

# Check dataset-level permissions
bq show --format=prettyjson cbi-v15:raw | grep -A 10 "access"
```

---

## Troubleshooting

### Error: "Permission denied on folder"
**Solution**: Request folder-level permissions from org admin

### Error: "Service account not found"
**Solution**: Run `setup_iam_permissions.sh` to create service accounts

### Error: "BigQuery dataset access denied"
**Solution**: Run `setup_iam_permissions.sh` to grant dataset permissions

### Error: "Cannot create project in folder"
**Solution**: Ensure you have `roles/resourcemanager.projectCreator` on folder

---

## Security Best Practices

1. ✅ **Least Privilege**: Service accounts have minimum required permissions
2. ✅ **Separate Accounts**: Different SAs for different purposes
3. ✅ **Read-Only Where Possible**: Cloud Run SA is read-only
4. ✅ **Secret Manager**: API keys stored in Secret Manager, not code
5. ✅ **Folder Organization**: Projects organized by purpose (App Development)

---

## Summary

**Automated Setup**:
- ✅ Service accounts (3)
- ✅ Project-level permissions
- ✅ Dataset-level permissions

**Manual Setup Required**:
- ⚠️  Folder-level permissions (org admin)

**Run**: `./scripts/setup/setup_iam_permissions.sh`

---

**Last Updated**: November 28, 2025

