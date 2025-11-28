# CBI-V15 Execution Checklist

**Date**: November 28, 2025  
**Status**: ✅ **READY** - All prerequisites met

---

## ✅ Pre-Execution Checklist

### Prerequisites
- [ ] ✅ Google Cloud SDK installed (`gcloud`)
- [ ] ✅ BigQuery CLI installed (`bq`)
- [ ] ✅ Python 3.9+ installed
- [ ] ✅ Node.js 18+ installed
- [ ] ✅ GCP billing account enabled
- [ ] ✅ Authenticated: `gcloud auth login`
- [ ] ✅ Organization admin access (for folder permissions)

### Documentation Review
- [ ] ✅ Read `COMPLETE_SETUP_EXECUTION_GUIDE.md`
- [ ] ✅ Read `docs/setup/IAM_PERMISSIONS_GUIDE.md`
- [ ] ✅ Read `README_BIGQUERY_SETUP.md`

---

## 🚀 Execution Steps

### Step 1: Pre-Flight Check
- [ ] Run `./scripts/setup/pre_flight_check.sh`
- [ ] All checks pass ✅

### Step 2: GCP Project Setup
- [ ] Run `./scripts/setup/setup_gcp_project.sh`
- [ ] Project created under App Development folder
- [ ] APIs enabled
- [ ] Datasets created
- [ ] Billing account linked (manual)

### Step 3: IAM Permissions
- [ ] Run `./scripts/setup/setup_iam_permissions.sh`
- [ ] 3 service accounts created
- [ ] Project-level permissions granted
- [ ] Dataset-level permissions granted
- [ ] Folder-level permissions requested (if needed)

### Step 4: BigQuery Skeleton
- [ ] Run `./scripts/setup/setup_bigquery_skeleton.sh`
- [ ] 42 tables created
- [ ] Reference tables initialized
- [ ] Verification passes

### Step 5: API Keys
- [ ] Run `./scripts/setup/store_api_keys.sh` OR manual setup
- [ ] Keys stored in macOS Keychain
- [ ] Keys stored in Secret Manager

### Step 6: Verification
- [ ] Run `python3 scripts/setup/verify_connections.py`
- [ ] All checks pass ✅

### Step 7: Dataform
- [ ] Run `cd dataform && npm install && dataform compile`
- [ ] Dataform compiles without errors

---

## ✅ Post-Setup Verification

### BigQuery
- [ ] All 8 datasets exist in `us-central1`
- [ ] All 42 tables exist
- [ ] Tables partitioned by date
- [ ] Tables clustered appropriately
- [ ] Reference tables populated

### IAM
- [ ] 3 service accounts exist
- [ ] Permissions granted correctly
- [ ] Folder-level permissions (if applicable)

### APIs
- [ ] All required APIs enabled
- [ ] API keys accessible (Keychain/Secret Manager)

### Dataform
- [ ] Dataform compiles successfully
- [ ] No errors in compilation

---

## 📋 Next Steps (After Setup)

- [ ] Test data ingestion (one source)
- [ ] Build first feature table
- [ ] Export training data
- [ ] Train LightGBM baselines

---

## 🐛 Troubleshooting Log

**Issues Encountered**:
- [ ] Issue 1: ________________
  - Solution: ________________
- [ ] Issue 2: ________________
  - Solution: ________________

---

**Status**: Ready to execute ✅

**Last Updated**: November 28, 2025

