# Connect Dataform to GitHub - Complete Guide

**Two Methods**: SSH (Recommended) or HTTPS (Simpler)

---

## Method 1: SSH Connection (Recommended - More Secure)

### Step 1: Run Setup Script
```bash
./scripts/setup/setup_dataform_github_ssh.sh
```

This script will:
1. Generate SSH key pair (if needed)
2. Display public key to add to GitHub
3. Store private key in Secret Manager
4. Grant Dataform service account access

### Step 2: Add Public Key to GitHub
1. Go to: https://github.com/settings/ssh/new
2. Title: `Dataform CBI-V15`
3. Key: Paste the public key shown by the script
4. Click **"Add SSH key"**

### Step 3: Connect in Dataform UI
1. Go to: [Google Cloud Console → Dataform](https://console.cloud.google.com/dataform)
2. Click **"Create Repository"** or select existing
3. Repository name: `CBI-V15`
4. Location: `us-central1`
5. Click **"Create"**
6. Go to **Settings** → **Git Remote Settings**
7. Click **"Connect to GitHub"**
8. Select **SSH** connection method
9. Enter:
   - **SSH URL**: `git@github.com:zincdigitalofmiami/CBI-V15.git`
   - **Secret**: `dataform-github-ssh-key` (from Secret Manager)
   - **Branch**: `main`
   - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
10. Click **"Connect"**

---

## Method 2: HTTPS Connection (Simpler - Uses Personal Access Token)

### Step 1: Create GitHub Personal Access Token
1. Go to: https://github.com/settings/tokens/new
2. Note: `Dataform CBI-V15`
3. Expiration: Choose duration
4. Scopes: Check `repo` (Full control of private repositories)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

### Step 2: Store Token in Secret Manager
```bash
echo "YOUR_GITHUB_TOKEN" | gcloud secrets create dataform-github-token \
    --data-file=- \
    --project=cbi-v15 \
    --replication-policy="automatic"
```

### Step 3: Grant Dataform Access
```bash
PROJECT_NUMBER=$(gcloud projects describe cbi-v15 --format='value(projectNumber)')
SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@gcp-sa-dataform.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding dataform-github-token \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project=cbi-v15
```

### Step 4: Connect in Dataform UI
1. Go to: [Google Cloud Console → Dataform](https://console.cloud.google.com/dataform)
2. Create or select repository `CBI-V15`
3. Go to **Settings** → **Git Remote Settings**
4. Click **"Connect to GitHub"**
5. Select **HTTPS** connection method
6. Enter:
   - **HTTPS URL**: `https://github.com/zincdigitalofmiami/CBI-V15.git`
   - **Secret**: `dataform-github-token` (from Secret Manager)
   - **Branch**: `main`
   - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
7. Click **"Connect"**

---

## ⚠️ Critical: Root Directory

**MUST be set to**: `dataform/`

**Why**: Your `dataform.json` file is in the `dataform/` subdirectory, not at the repository root.

**If you forget this**: Dataform won't find your files and will show errors.

---

## Verification

After connecting:

1. **Check Repository Status**:
   - Go to Dataform → Your Repository
   - You should see your SQL files listed
   - "Create repository" button should disappear

2. **Test Compilation**:
   - Click **"Compile"** in Dataform UI
   - Should show 18+ actions compiled
   - No errors (except 2 non-critical UDF warnings)

3. **Verify Files Visible**:
   - Navigate to `definitions/01_raw/`
   - Should see: `databento_daily.sqlx`, `fred_macro.sqlx`, etc.
   - Navigate to `definitions/02_staging/`
   - Should see: `market_daily.sqlx`, `fred_macro_clean.sqlx`, etc.

---

## Troubleshooting

### "Repository not found"
- Verify repository name: `zincdigitalofmiami/CBI-V15`
- Check GitHub access permissions
- Verify SSH key is added to GitHub (for SSH method)

### "Secret not found"
- Run the setup script: `./scripts/setup/setup_dataform_github_ssh.sh`
- Or create secret manually in Secret Manager

### "Files not visible"
- **Check Root Directory**: Must be `dataform/`
- Refresh the Dataform page
- Verify files exist in GitHub repository

### "Compilation errors"
- Check Root Directory is `dataform/`
- Verify `dataform.json` exists in `dataform/` directory
- Check file paths in SQL files

---

## Quick Reference

**SSH Method** (Recommended):
```bash
./scripts/setup/setup_dataform_github_ssh.sh
# Then follow UI steps above
```

**HTTPS Method** (Simpler):
- Create GitHub PAT
- Store in Secret Manager
- Connect via UI with HTTPS URL

**Root Directory**: `dataform/` ⚠️ **CRITICAL**

---

**After Connection**: You can compile and run Dataform transformations from the UI!

